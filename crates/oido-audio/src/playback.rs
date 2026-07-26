use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};
use rubato::Resampler as _;

use crate::{AudioChunk, AudioError, PlaybackSink};

/// Capacidad del canal `PlaybackCommand`. Pequeña a propósito: si es
/// grande, el worker TTS produce docenas de chunks (cada uno ~1.5s de
/// audio) antes de que cpal los consuma, y el usuario oye una "cola"
/// larguísima de audio que se reproduce minutos después de que él ya
/// dejó de importarle. Con cap=2, el worker TTS se bloquea tras
/// encolar el chunk actual + 1 prefetch → backpressure natural desde
/// el dispositivo de salida hacia la síntesis.
const WORKER_QUEUE_CAP: usize = 2;
const WORKER_THREAD_NAME: &str = "oido-playback-worker";

#[derive(Debug)]
enum PlaybackCommand {
    Enqueue(AudioChunk),
    CancelAndPlay(AudioChunk),
    Stop,
}

pub struct CpalPlayback {
    tx: Sender<PlaybackCommand>,
    sample_rate: u32,
}

impl std::fmt::Debug for CpalPlayback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalPlayback")
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

impl CpalPlayback {
    pub fn new() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| AudioError::Playback("no hay dispositivo de salida".into()))?;
        let supported = device.default_output_config().map_err(|e| AudioError::Playback(format!("configuración de salida: {e}")))?;
        let rate = supported.sample_rate();
        let channels = if supported.channels() >= 1 { supported.channels() } else { 1 };
        let config = cpal::StreamConfig { channels, sample_rate: rate, buffer_size: cpal::BufferSize::Default };
        let (tx, rx) = bounded(WORKER_QUEUE_CAP);
        thread::Builder::new().name(WORKER_THREAD_NAME.into()).spawn(move || run_worker(device, config, rx, rate))
            .map_err(|e| AudioError::Playback(format!("thread spawn: {e}")))?;
        Ok(Self { tx, sample_rate: rate })
    }
}

impl PlaybackSink for CpalPlayback {
    fn enqueue(&self, chunk: AudioChunk) -> Result<(), AudioError> { self.tx.send(PlaybackCommand::Enqueue(chunk)).map_err(|_| AudioError::Playback("worker muerto".into())) }
    fn cancel_and_play(&self, chunk: AudioChunk) -> Result<(), AudioError> { self.tx.send(PlaybackCommand::CancelAndPlay(chunk)).map_err(|_| AudioError::Playback("worker muerto".into())) }
    fn stop(&self) -> Result<(), AudioError> { self.tx.send(PlaybackCommand::Stop).map_err(|_| AudioError::Playback("worker muerto".into())) }
    fn is_playing(&self) -> bool { false }
    fn device_sample_rate_hz(&self) -> u32 { self.sample_rate }
}

fn run_worker(device: cpal::Device, config: cpal::StreamConfig, rx: Receiver<PlaybackCommand>, rate: u32) {
    // Cola interna de muestras PCM mono. Capacidad ~2s de audio a la
    // tasa del dispositivo (suficiente para absorber bursts del worker
    // sin bloquear el callback de cpal).
    let (sample_tx, sample_rx) = bounded::<f32>((rate as usize).max(8000) * 2);
    let channels = config.channels as usize;
    let callback_rx = sample_rx.clone();

    tracing::info!(
        device_rate = rate,
        channels,
        "CpalPlayback: abriendo stream de salida"
    );

    // El stream SIEMPRE activo: cuando no hay muestras para reproducir,
    // el callback rellena con silencio (0.0). Esto evita el bug de
    // stream pausado que nunca arranca el primer `play()`, y simplifica
    // el ciclo de vida: no gestionamos pause/play desde el worker loop.
    let stream = match device.build_output_stream::<f32, _, _>(config, move |data, _| {
        for frame in data.chunks_mut(channels) {
            let sample = callback_rx.try_recv().unwrap_or(0.0);
            frame.fill(sample);
        }
    }, |e| tracing::error!(?e, "error en stream de salida"), None) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(?e, "CpalPlayback: no se pudo abrir stream de salida");
            return;
        }
    };

    // Arrancar el stream INMEDIATAMENTE. Desde este momento el callback
    // corre emitiendo silencio hasta que `sample_tx` reciba muestras.
    if let Err(e) = stream.play() {
        tracing::error!(?e, "CpalPlayback: stream.play() inicial falló");
        return;
    }
    tracing::info!("CpalPlayback: stream activo (callback emitiendo silencio hasta que llegue audio)");

    // Sin cola intermedia: procesamos cada chunk conforme llega. El
    // canal `rx` (cap=2) ya provee backpressure hacia el worker TTS —
    // si cpal no está drenando, `rx` se llena y el TTS worker se
    // bloquea en `enqueue()`. Sin esta eliminación de la VecDeque
    // teníamos 16 chunks acumulados ~24s de audio en buffer.
    let mut enqueued_total: u64 = 0;
    loop {
        let chunk_opt: Option<AudioChunk> = match rx.recv() {
            Ok(PlaybackCommand::Enqueue(chunk)) => {
                tracing::debug!(
                    chunk_samples = chunk.samples.len(),
                    chunk_rate = chunk.sample_rate_hz,
                    "CpalPlayback: Enqueue recibido"
                );
                Some(chunk)
            }
            Ok(PlaybackCommand::CancelAndPlay(chunk)) => {
                // Flush del buffer del callback para que el chunk nuevo
                // suene inmediatamente, no detrás de 1s de audio viejo.
                while sample_rx.try_recv().is_ok() {}
                Some(chunk)
            }
            Ok(PlaybackCommand::Stop) => {
                while sample_rx.try_recv().is_ok() {}
                None
            }
            Err(_) => return,
        };
        if let Some(chunk) = chunk_opt {
            let samples = resample(&chunk, rate);
            let n = samples.len();
            for sample in samples {
                // `send` bloquea cuando el callback no ha drenado
                // (buffer lleno) — eso es exactamente la backpressure
                // que queremos: el worker TTS no produce el siguiente
                // chunk hasta que el actual se esté reproduciendo.
                if sample_tx.send(sample).is_err() {
                    tracing::warn!("CpalPlayback: sample_tx cerrado; saliendo del worker");
                    return;
                }
            }
            enqueued_total += n as u64;
            tracing::debug!(
                enqueued_samples = n,
                enqueued_total,
                target_rate = rate,
                "CpalPlayback: chunk resampled y encolado al callback"
            );
        }
    }
}

fn resample(chunk: &AudioChunk, target: u32) -> Vec<f32> {
    if chunk.samples.is_empty() || chunk.sample_rate_hz == 0 || chunk.sample_rate_hz == target { return chunk.samples.clone(); }
    let params = rubato::SincInterpolationParameters { sinc_len: 128, f_cutoff: 0.95, interpolation: rubato::SincInterpolationType::Linear, oversampling_factor: 256, window: rubato::WindowFunction::BlackmanHarris2 };
    let ratio = f64::from(target) / f64::from(chunk.sample_rate_hz);
    let size = 512;
    let mut r = match rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, size, 1) { Ok(r) => r, Err(_) => return Vec::new() };
    let mut out = Vec::new();
    let mut input = chunk.samples.clone();
    let rem = input.len() % size;
    if rem != 0 { input.extend(std::iter::repeat_n(0.0, size - rem)); }
    for part in input.chunks(size) { if let Ok(v) = r.process(&[part.to_vec()], None) { out.extend(v.into_iter().flatten()); } }
    let expected = (chunk.samples.len() as f64 * f64::from(target) / f64::from(chunk.sample_rate_hz)).round() as usize;
    if out.len() < expected { out.resize(expected, 0.0); } else { out.truncate(expected); }
    out
}

const _: fn() = || { fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<CpalPlayback>(); };

#[cfg(test)]
mod tests {
    use super::*;
    fn length(from: u32, to: u32, n: usize) -> usize { resample(&AudioChunk { samples: vec![0.0; n], sample_rate_hz: from }, to).len() }
    #[test] fn resampler_22050_to_48000_produces_correct_length() { assert!((length(22050, 48000, 1024) as i64 - 2229).abs() < 40); }
    #[test] fn resampler_24000_to_48000_produces_correct_length() { assert!((length(24000, 48000, 1024) as i64 - 2048).abs() < 40); }
    #[test] fn sine_wave_440hz_at_48khz_has_expected_length_after_resampling() { let n=22050*200/1000; let samples=(0..n).map(|i| (2.0*std::f32::consts::PI*440.0*i as f32/22050.0).sin()).collect(); let out=resample(&AudioChunk{samples,sample_rate_hz:22050},48000); assert!((out.len() as i64-9600).abs()<40); }
}
