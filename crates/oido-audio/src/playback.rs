use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};

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
        let device = host
            .default_output_device()
            .ok_or_else(|| AudioError::Playback("no hay dispositivo de salida".into()))?;
        let supported = device
            .default_output_config()
            .map_err(|e| AudioError::Playback(format!("configuración de salida: {e}")))?;
        let rate = supported.sample_rate();
        let channels = if supported.channels() >= 1 {
            supported.channels()
        } else {
            1
        };
        let config = cpal::StreamConfig {
            channels,
            sample_rate: rate,
            buffer_size: cpal::BufferSize::Default,
        };
        let (tx, rx) = bounded(WORKER_QUEUE_CAP);
        thread::Builder::new()
            .name(WORKER_THREAD_NAME.into())
            .spawn(move || run_worker(device, config, rx, rate))
            .map_err(|e| AudioError::Playback(format!("thread spawn: {e}")))?;
        Ok(Self {
            tx,
            sample_rate: rate,
        })
    }
}

impl PlaybackSink for CpalPlayback {
    fn enqueue(&self, chunk: AudioChunk) -> Result<(), AudioError> {
        self.tx
            .send(PlaybackCommand::Enqueue(chunk))
            .map_err(|_| AudioError::Playback("worker muerto".into()))
    }
    fn cancel_and_play(&self, chunk: AudioChunk) -> Result<(), AudioError> {
        self.tx
            .send(PlaybackCommand::CancelAndPlay(chunk))
            .map_err(|_| AudioError::Playback("worker muerto".into()))
    }
    fn stop(&self) -> Result<(), AudioError> {
        self.tx
            .send(PlaybackCommand::Stop)
            .map_err(|_| AudioError::Playback("worker muerto".into()))
    }
    fn is_playing(&self) -> bool {
        false
    }
    fn device_sample_rate_hz(&self) -> u32 {
        self.sample_rate
    }
}

fn run_worker(
    device: cpal::Device,
    config: cpal::StreamConfig,
    rx: Receiver<PlaybackCommand>,
    rate: u32,
) {
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
    let stream = match device.build_output_stream::<f32, _, _>(
        config,
        move |data, _| {
            for frame in data.chunks_mut(channels) {
                let sample = callback_rx.try_recv().unwrap_or(0.0);
                frame.fill(sample);
            }
        },
        |e| tracing::error!(?e, "error en stream de salida"),
        None,
    ) {
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
    tracing::info!(
        "CpalPlayback: stream activo (callback emitiendo silencio hasta que llegue audio)"
    );

    // Resampler STATEFUL: vive todo el ciclo de vida del worker y
    // mantiene el filtro sinc "caliente" entre chunks. Antes teníamos
    // un `SincFixedIn` construido por-chunk, que reseteaba el estado
    // del filtro en cada frontera de AudioChunk → transientes de
    // "ring-in" audibles como clicks/zipper que sonaban robóticos en
    // AMBOS engines (porque ambos pasan por este mismo `resample`).
    // Espejo del `Resampler` del lado de captura (`capture.rs:512-617`),
    // que sí mantiene un `pending: Vec<f32>` entre llamadas.
    let mut resampler = PlaybackResampler::new(rate);

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
                // También descartamos el pending del resampler: son
                // muestras del audio cancelado que aún no completaban un
                // chunk del filtro. Si las dejáramos, sonarían detrás
                // del chunk nuevo.
                resampler.flush();
                Some(chunk)
            }
            Ok(PlaybackCommand::Stop) => {
                while sample_rx.try_recv().is_ok() {}
                resampler.flush();
                None
            }
            Err(_) => return,
        };
        if let Some(chunk) = chunk_opt {
            let samples = resampler.process(&chunk);
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

/// Resampler stateful para playback: lleva PCM desde el `sample_rate_hz`
/// del engine (24 kHz Kokoro, 22.05 kHz Piper) hasta el rate nativo del
/// dispositivo (típicamente 44.1/48 kHz).
///
/// **Diferencia clave vs. la función libre `resample()` anterior**: el
/// `SincFixedIn` se construye UNA vez y se reutiliza entre chunks. Un
/// filtro sinc necesita ~`sinc_len` (128) muestras de "ring-in" antes
/// de emitir output correcto; resetearlo por-chunk mete un transiente
/// en cada frontera de `AudioChunk` que suena como click/zipper
/// ("robótico"). Mantener el filtro caliente entre chunks elimina ese
/// artefacto. Es el mismo patrón que `capture::Resampler`.
///
/// `pending` acumula el "resto" de un chunk que no completa un bloque
/// de `chunk_in` muestras; se difiere al siguiente `process()` para que
/// rubato siempre reciba bloques exactos (sin padding con ceros, que
/// introduciría una caída a silencio brusca al final de cada chunk).
struct PlaybackResampler {
    /// `None` = identidad (input rate == target), sin filtro.
    /// `Some` = filtro activo para `input_rate` → `target`.
    inner: Option<rubato::SincFixedIn<f32>>,
    /// Tamaño de bloque que `rubato::SincFixedIn` espera por llamada a
    /// `process`. Cacheado para no pedirlo al filtro en cada chunk.
    chunk_in: usize,
    /// Acumulador entre llamadas: samples que no completan un bloque de
    /// `chunk_in` se difieren al siguiente `process`.
    pending: Vec<f32>,
    /// `sample_rate_hz` para el que se construyó `inner`. Si llega un
    /// chunk con otro rate (engine switch Kokoro↔Piper), hay que
    /// reconstruir el filtro porque el ratio cambió.
    input_rate: u32,
    /// Rate nativo del dispositivo (fijo para la vida del worker).
    target: u32,
    /// Cuota de seguridad anti-desbordamiento de `pending`. Espejo del
    /// `max_pending` del resampler de captura.
    max_pending: usize,
}

impl PlaybackResampler {
    /// Crea un resampler que llevará cualquier `sample_rate_hz` de
    /// entrada a `target` (el rate del dispositivo). El filtro concreto
    /// se materializa perezosamente en el primer `process()` (cuando
    /// sabemos el `input_rate` del primer chunk).
    fn new(target: u32) -> Self {
        Self {
            inner: None,
            chunk_in: 512,
            pending: Vec::new(),
            input_rate: 0,
            target,
            max_pending: 512 * 128,
        }
    }

    /// Procesa un chunk y devuelve las muestras equivalentes a
    /// `target` Hz. Acumula internamente entre llamadas: si llega menos
    /// de `chunk_in` muestras, las difiere al siguiente `process`.
    fn process(&mut self, chunk: &AudioChunk) -> Vec<f32> {
        // Casos triviales: chunk vacío, rate cero, o identidad.
        if chunk.samples.is_empty() || chunk.sample_rate_hz == 0 {
            return Vec::new();
        }
        if chunk.sample_rate_hz == self.target {
            // Identidad: no hay filtro. Si teníamos un filtro de un
            // rate anterior (engine switch), lo descartamos junto con su
            // pending (que era de otro rate).
            if self.inner.is_some() {
                self.inner = None;
                self.pending.clear();
                self.input_rate = 0;
            }
            return chunk.samples.clone();
        }

        // Si cambió el input rate respecto al filtro actual (engine
        // switch Kokoro↔Piper), reconstruimos. Es raro y caro, pero
        // correcto. Descartamos el pending viejo: era del rate anterior.
        if chunk.sample_rate_hz != self.input_rate {
            let params = rubato::SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                interpolation: rubato::SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: rubato::WindowFunction::BlackmanHarris2,
            };
            let ratio = f64::from(self.target) / f64::from(chunk.sample_rate_hz);
            match rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, self.chunk_in, 1) {
                Ok(r) => {
                    self.inner = Some(r);
                    self.input_rate = chunk.sample_rate_hz;
                    self.pending.clear();
                    tracing::debug!(
                        input_rate = chunk.sample_rate_hz,
                        target = self.target,
                        "PlaybackResampler: filtro reconstruido (engine switch?)"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        ?e,
                        input_rate = chunk.sample_rate_hz,
                        target = self.target,
                        "PlaybackResampler: no se pudo construir el filtro; pasando crudo"
                    );
                    return chunk.samples.clone();
                }
            }
        }

        let Some(ref mut inner) = self.inner else {
            return chunk.samples.clone();
        };

        self.pending.extend_from_slice(&chunk.samples);

        // Cuota anti-desbordamiento (espejo del capture-side).
        if self.pending.len() > self.max_pending {
            tracing::warn!(
                pending = self.pending.len(),
                max = self.max_pending,
                "PlaybackResampler.pending desbordó la cuota; descartando lo más viejo"
            );
            let drop_n = self.pending.len() - self.max_pending;
            self.pending.drain(..drop_n);
        }

        use rubato::Resampler as _;
        let mut out = Vec::new();
        while self.pending.len() >= self.chunk_in {
            let block: Vec<f32> = self.pending.drain(..self.chunk_in).collect();
            match inner.process(&[block], None) {
                Ok(v) => out.extend(v.into_iter().flatten()),
                Err(e) => {
                    tracing::warn!(?e, "PlaybackResampler.process falló; descartando bloque");
                }
            }
        }
        // Lo que quede en `pending` (< chunk_in) se difiere al siguiente
        // `process()`. NO paddeamos con ceros: cualquier chunk futuro lo
        // completa y rubato procesa exactamente `chunk_in`. El padding
        // anterior era la otra fuente de "plop" al final de cada chunk.
        out
    }

    /// Descarta el `pending` acumulado. Lo llama el worker ante
    /// `Stop` / `CancelAndPlay` para que muestras del audio cancelado
    /// no suenen detrás del siguiente chunk.
    fn flush(&mut self) {
        self.pending.clear();
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CpalPlayback>();
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Resamplea un chunk único de `from` Hz a `to` Hz con un
    /// `PlaybackResampler` fresco y devuelve el nº de muestras de
    /// salida. Helper para los tests de longitud.
    ///
    /// NOTA: con un solo `process()` el output siempre es MENOR que
    /// `n * (to/from)` porque (a) rubato retiene un delay interno de
    /// ring-in (~`sinc_len/2` muestras) y (b) el último bloque de
    /// `<chunk_in` muestras queda en `pending` sin emitirse. En
    /// producción esto no importa: los chunks siguientes drenan el
    /// pending y el delay se paga una sola vez al inicio del utterance.
    /// Los tests de longitud usan tolerancias amplias para reflejar
    /// esto; el test `split_chunks_produce_same_total_as_single_chunk`
    /// valida la continuidad real entre chunks.
    fn length(from: u32, to: u32, n: usize) -> usize {
        let mut r = PlaybackResampler::new(to);
        let chunk = AudioChunk {
            samples: vec![0.0; n],
            sample_rate_hz: from,
        };
        r.process(&chunk).len()
    }

    #[test]
    fn resampler_22050_to_48000_produces_correct_length() {
        // 22050→48000 ratio ≈ 2.177. Tolerancia amplia: cubre el delay
        // interno de rubato + el pending del último bloque.
        let got = length(22050, 48000, 8192);
        let expected = (8192.0 * 48000.0 / 22050.0) as i64;
        assert!(
            (got as i64 - expected).abs() < 600,
            "esperaba ~{expected} (±600), obtuve {got}"
        );
    }

    #[test]
    fn resampler_24000_to_48000_produces_correct_length() {
        // 24000→48000 ratio = 2.0 exacto.
        let got = length(24000, 48000, 8192);
        assert!(
            (got as i64 - 16384).abs() < 600,
            "esperaba ~16384 (±600), obtuve {got}"
        );
    }

    /// Un chunk grande de onda senoidal 440 Hz @ 22050 Hz, resampleado
    /// a 48000 Hz, debe acercarse a la duración esperada (~200 ms).
    /// Tolerancia amplia: con ~4400 muestras de entrada, el delay
    /// interno de rubato (~sinc_len/2 en el dominio de salida) + el
    /// pending del último bloque pesan una fracción no trivial. En
    /// producción esto se compensa cuando los chunks siguientes drenan
    /// el pending.
    #[test]
    fn sine_wave_440hz_preserves_duration_after_resampling() {
        let n = 22050 * 200 / 1000;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let mut r = PlaybackResampler::new(48000);
        let out = r.process(&AudioChunk {
            samples,
            sample_rate_hz: 22050,
        });
        // 200ms @ 48kHz = 9600 muestras esperadas. El delay interno
        // roba ~800-900; permitimos hasta 1000 de diferencia.
        assert!(
            (out.len() as i64 - 9600).abs() < 1000,
            "esperaba ~9600 muestras (200ms @ 48kHz, ±1000), obtuve {}",
            out.len()
        );
    }

    /// **Caso clave del bug robótico**: procesar el mismo audio en
    /// DOS chunks separados debe producir el MISMO total de muestras
    /// que procesarlo en UN solo chunk. Antes, el `resample()` por-chunk
    /// reseteaba el filtro y producía longitudes ligeramente distintas
    /// (por el truncamiento/padding por chunk), causando glitches.
    #[test]
    fn split_chunks_produce_same_total_as_single_chunk() {
        let n = 22050 * 300 / 1000; // 300ms @ 22050
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 22050.0).sin())
            .collect();
        let mid = n / 2;

        // Un solo chunk.
        let mut r1 = PlaybackResampler::new(48000);
        let single = r1.process(&AudioChunk {
            samples: samples.clone(),
            sample_rate_hz: 22050,
        });

        // Dos chunks con el MISMO resampler (stateful).
        let mut r2 = PlaybackResampler::new(48000);
        let first = r2.process(&AudioChunk {
            samples: samples[..mid].to_vec(),
            sample_rate_hz: 22050,
        });
        let second = r2.process(&AudioChunk {
            samples: samples[mid..].to_vec(),
            sample_rate_hz: 22050,
        });
        let combined: Vec<f32> = first.into_iter().chain(second).collect();

        assert_eq!(
            single.len(),
            combined.len(),
            "split en 2 chunks debe producir el mismo total que un solo chunk"
        );
        // Y el contenido debe ser (casi) idéntico: el filtro stateful
        // preserva continuidad, así que las muestras deben coincidir
        // salvo error de redondeo del filtro en la frontera.
        let max_diff = single
            .iter()
            .zip(combined.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 0.01,
            "diferencia máxima entre single y combined demasiado alta: {max_diff}"
        );
    }

    /// Rate identidad (chunk.sample_rate_hz == target) devuelve el
    /// input sin tocar, incluso si antes había un filtro activo.
    #[test]
    fn identity_rate_passes_through() {
        let mut r = PlaybackResampler::new(48000);
        // Primero un chunk 24000 (activa el filtro).
        let _ = r.process(&AudioChunk {
            samples: vec![0.5; 1024],
            sample_rate_hz: 24000,
        });
        // Ahora un chunk 48000 (identidad) → debe pasar crudo.
        let out = r.process(&AudioChunk {
            samples: vec![0.7; 100],
            sample_rate_hz: 48000,
        });
        assert_eq!(out, vec![0.7; 100]);
    }

    /// Engine switch (cambio de input_rate entre chunks) reconstruye el
    /// filtro sin pánico y sigue produciendo output.
    #[test]
    fn engine_switch_rebuilds_filter() {
        let mut r = PlaybackResampler::new(48000);
        // Kokoro 24kHz.
        let a = r.process(&AudioChunk {
            samples: vec![0.1; 2048],
            sample_rate_hz: 24000,
        });
        assert!(!a.is_empty());
        // Piper 22.05kHz — distinto ratio, debe reconstruir.
        let b = r.process(&AudioChunk {
            samples: vec![0.2; 2048],
            sample_rate_hz: 22050,
        });
        assert!(!b.is_empty());
    }
}
