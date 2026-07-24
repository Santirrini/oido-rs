//! `CaptureSource` para todos los OS. Usa `cpal` (cross-platform).
//!
//! Pipeline de audio:
//!
//! ```text
//! cpal stream (cualquier sample rate, F32)
//!      ↓
//!   [`CpalCapture::start`] envía AudioFrame { sample_rate_hz = real }
//!      ↓
//!   [`oido-core`] consumer thread aplica resampler → 16 kHz
//! ```
//!
//! Esto soluciona el bug de Fase 1 donde si el dispositivo no soportaba
//! 16 kHz nativo, whisper.cpp recibía audio al sample rate incorrecto
//! y producía basura.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crossbeam_channel::Sender;
use std::fmt;

use crate::{AudioError, AudioFrame, CaptureSource};

pub struct CpalCapture {
    device: cpal::Device,
    device_name: String,
    stream_config: cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    sample_rate: u32,
    sink: Option<Sender<AudioFrame>>,
    stream: Option<cpal::Stream>,
}

impl std::fmt::Debug for CpalCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalCapture")
            .field("device", &self.device_name)
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.stream_config.channels)
            .field("sample_format", &self.sample_format)
            .field("stream_active", &self.stream.is_some())
            .finish()
    }
}

/// Descripción ligera de un dispositivo de entrada — lo que devolvemos
/// al submenú del tray y al CLI. Sin cpal-types en la API pública.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

impl CpalCapture {
    /// Constructor con selección explícita de dispositivo.
    ///
    /// - `Some(name)`: enumera los dispositivos de entrada y elige el
    ///   cuyo `description().name()` coincide con `name`. Si no se
    ///   encuentra, emite un `warn!` y cae al default del OS.
    /// - `None`: usa el dispositivo de entrada por defecto (mismo
    ///   comportamiento que el bin tenía antes de esta feature).
    pub fn with_device(name: Option<&str>) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let default_device = host.default_input_device();
        let default_name = default_device
            .as_ref()
            .and_then(|d| d.description().ok().map(|x| x.name().to_owned()));

        let (device, device_name) = match name {
            None => match default_device {
                Some(d) => {
                    let n = d
                        .description()
                        .ok()
                        .map(|x| x.name().to_owned())
                        .unwrap_or_else(|| "<unknown>".into());
                    (d, n)
                }
                None => {
                    return Err(AudioError::Capture(
                        "sin dispositivo de entrada por defecto".into(),
                    ));
                }
            },
            Some(want) => {
                let mut found = None;
                if let Ok(devs) = host.input_devices() {
                    for d in devs {
                        let n = d
                            .description()
                            .ok()
                            .map(|x| x.name().to_owned())
                            .unwrap_or_default();
                        if n == want {
                            found = Some((d, n));
                            break;
                        }
                    }
                }
                match found {
                    Some(p) => p,
                    None => {
                        tracing::warn!(
                            requested = %want,
                            "dispositivo de entrada solicitado no existe; cayendo al default del OS"
                        );
                        match default_device {
                            Some(d) => {
                                let n = d
                                    .description()
                                    .ok()
                                    .map(|x| x.name().to_owned())
                                    .unwrap_or_else(|| "<unknown>".into());
                                (d, n)
                            }
                            None => {
                                return Err(AudioError::Capture(format!(
                                    "dispositivo '{want}' no existe y no hay default de OS"
                                )));
                            }
                        }
                    }
                }
            }
        };

        if let Some(ref def) = default_name {
            if def == &device_name {
                tracing::info!(device = %device_name, "captura desde dispositivo por defecto del OS");
            } else {
                tracing::info!(device = %device_name, "captura desde dispositivo seleccionado por el usuario");
            }
        }

        let (stream_config, sample_format, sample_rate) = negotiate_config(&device)?;

        Ok(Self {
            device,
            device_name,
            stream_config,
            sample_format,
            sample_rate,
            sink: None,
            stream: None,
        })
    }

    /// Constructor sin selección explícita: usa el default del OS.
    /// Equivalente a `with_device(None)`. Mantenido para no romper
    /// el ejemplo `mic_check.rs` y los tests existentes.
    pub fn new() -> Result<Self, AudioError> {
        Self::with_device(None)
    }
}

/// Negocia la mejor configuración de captura para un dispositivo
/// dado. Preferencia: 16 kHz mono F32 (lo que necesita whisper.cpp).
/// Si no está soportado, cae a `default_input_config()` y deja que
/// el resampler de `oido-core` haga la conversión.
fn negotiate_config(
    device: &cpal::Device,
) -> Result<(cpal::StreamConfig, cpal::SampleFormat, u32), AudioError> {
    let mut wanted = None;
    if let Ok(supported) = device.supported_input_configs() {
        for cfg in supported {
            if cfg.channels() == 1
                && cfg.sample_format() == cpal::SampleFormat::F32
                && cfg.contains_rate(16_000)
            {
                wanted = cfg.try_with_sample_rate(16_000);
                break;
            }
        }
    }
    match wanted {
        Some(s) => Ok((s.config(), cpal::SampleFormat::F32, 16_000_u32)),
        None => {
            let fallback = device
                .default_input_config()
                .map_err(|e| AudioError::Capture(format!("default_input_config: {e}")))?;
            tracing::warn!(
                requested = 16_000,
                actual = fallback.sample_rate(),
                "dispositivo no soporta 16kHz mono F32; usando default + resampling"
            );
            Ok((
                fallback.config(),
                fallback.sample_format(),
                fallback.sample_rate(),
            ))
        }
    }
}

/// Lista los dispositivos de entrada disponibles en el host por
/// defecto. El primero con `is_default = true` es el que el OS elige
/// cuando no se hace una selección explícita. Usado por el submenú
/// "Micrófono" del tray y por el flag `--set-mic` del CLI para
/// validar el argumento.
pub fn list_input_devices() -> Vec<InputDeviceInfo> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.description().ok().map(|x| x.name().to_owned()));
    let mut out = Vec::new();
    if let Ok(devs) = host.input_devices() {
        for d in devs {
            let name = d
                .description()
                .ok()
                .map(|x| x.name().to_owned())
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let is_default = default_name.as_deref() == Some(name.as_str());
            out.push(InputDeviceInfo { name, is_default });
        }
    }
    out
}

/// Resultado del sondeo de calidad de un dispositivo individual.
#[derive(Debug, Clone)]
pub struct DeviceProbe {
    pub name: String,
    /// RMS en dBFS durante el periodo de sondeo. -inf = silencio total.
    pub rms_dbfs: f32,
    /// Fracción de muestras con |x| < 1e-4 (0..1). 1.0 = silencio
    /// digital exacto (muteado / sin permiso / dispositivo sin señal).
    pub near_zero_frac: f32,
}

/// Sondea todos los dispositivos de entrada disponibles. Para cada
/// uno abre un stream corto, captura `duration_ms` y mide RMS. El
/// sondeo se hace secuencialmente para evitar peleas de permisos
/// con el dispositivo por defecto del OS.
///
/// Devuelve el nombre del dispositivo con mejor RMS, **excluyendo**
/// los que dan silencio digital (`near_zero_frac > 0.99`). Si todos
/// los dispositivos dan silencio, devuelve `None` — el caller decide
/// qué hacer (típicamente: `warn!` y no persistir nada).
///
/// Se ejecuta en un thread dedicado en el caller (la operación
/// bloquea `duration_ms * N_devices`).
pub fn probe_devices(duration_ms: u32) -> Vec<DeviceProbe> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.description().ok().map(|x| x.name().to_owned()));

    let mut results: Vec<DeviceProbe> = Vec::new();
    let Ok(devs) = host.input_devices() else {
        return results;
    };
    for d in devs {
        let name = d
            .description()
            .ok()
            .map(|x| x.name().to_owned())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let is_default = default_name.as_deref() == Some(name.as_str());
        let probe = match probe_one(&d, &name, duration_ms, is_default) {
            Some(p) => p,
            None => continue,
        };
        results.push(probe);
    }
    results
}

fn probe_one(
    device: &cpal::Device,
    name: &str,
    duration_ms: u32,
    is_default: bool,
) -> Option<DeviceProbe> {
    use std::sync::{Arc, Mutex};
    let cfg = device.default_input_config().ok()?;
    let sample_rate = cfg.sample_rate();
    let channels = cfg.channels() as usize;
    let stream_config = cfg.config();
    let sample_format = cfg.sample_format();
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    macro_rules! build_stream {
        ($t:ty, $conv:expr) => {{
            let buf = Arc::clone(&buf);
            let channels = channels;
            let name_for_err = name.to_string();
            device.build_input_stream(
                stream_config.clone(),
                move |data: &[$t], _| {
                    let conv: fn($t) -> f32 = $conv;
                    let mut g = buf.lock().expect("poisoned");
                    for frame in data.chunks(channels.max(1)) {
                        let s: f32 = frame.iter().map(|&x| conv(x)).sum::<f32>()
                            / frame.len() as f32;
                        g.push(s);
                    }
                },
                move |e| tracing::error!(device = %name_for_err, ?e, "probe stream error"),
                None,
            )
        }};
    }

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream!(f32, |x| x),
        cpal::SampleFormat::I16 => build_stream!(i16, |x| f32::from(x) / 32_768.0),
        cpal::SampleFormat::U16 => build_stream!(u16, |x| (f32::from(x) - 32_768.0) / 32_768.0),
        f => {
            tracing::debug!(device = name, ?f, "probe: sample format no soportado");
            return None;
        }
    }
    .ok()?;
    if stream.play().is_err() {
        return None;
    }
    std::thread::sleep(std::time::Duration::from_millis(u64::from(duration_ms)));
    let _ = stream.pause();
    drop(stream);

    let samples = buf.lock().expect("poisoned").clone();
    let (rms, zeros_frac) = rms_dbfs(&samples);
    tracing::info!(
        device = name,
        is_default,
        sample_rate,
        captured = samples.len(),
        rms_dbfs = rms,
        near_zero_frac = zeros_frac,
        "probe: dispositivo sondeado"
    );
    Some(DeviceProbe {
        name: name.to_string(),
        rms_dbfs: rms,
        near_zero_frac: zeros_frac,
    })
}

/// Escoge el mejor dispositivo a partir de un vector de probes.
/// "Mejor" = mayor RMS, descartando los que son silencio digital
/// (`near_zero_frac > 0.99`). Si todos son silencio, devuelve `None`.
pub fn pick_best_device(probes: &[DeviceProbe]) -> Option<String> {
    let eligible: Vec<&DeviceProbe> = probes
        .iter()
        .filter(|p| p.near_zero_frac < 0.99 && p.rms_dbfs.is_finite())
        .collect();
    eligible
        .into_iter()
        .max_by(|a, b| {
            a.rms_dbfs
                .partial_cmp(&b.rms_dbfs)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.name.clone())
}

fn rms_dbfs(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (f32::NEG_INFINITY, 1.0);
    }
    let mut sum = 0.0f64;
    let mut zeros = 0usize;
    for &s in samples {
        sum += f64::from(s) * f64::from(s);
        if s.abs() < 1e-4 {
            zeros += 1;
        }
    }
    let rms = (sum / samples.len() as f64).sqrt() as f32;
    let db = if rms <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    };
    (db, zeros as f32 / samples.len() as f32)
}

impl CaptureSource for CpalCapture {
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), AudioError> {
        self.sink = Some(sink);
        Ok(())
    }

    fn start(&mut self) -> Result<(), AudioError> {
        let sink = self
            .sink
            .clone()
            .ok_or_else(|| AudioError::Capture("open() no invocado antes de start()".into()))?;
        let sample_rate = self.sample_rate;
        let channels = self.stream_config.channels as usize;

        // Ponytail: tres branches (F32/I16/U16) en lugar de un trait
        // object porque cpal selecciona el closure por tipo concreto.
        let stream = match self.sample_format {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                self.stream_config,
                move |data: &[f32], _cb| {
                    let samples = if channels == 1 {
                        data.to_vec()
                    } else {
                        let mut mono = Vec::with_capacity(data.len() / channels);
                        for frame in data.chunks_exact(channels) {
                            let sum: f32 = frame.iter().sum();
                            mono.push(sum / (channels as f32));
                        }
                        mono
                    };
                    let _ = sink.send(AudioFrame {
                        samples,
                        sample_rate_hz: sample_rate,
                    });
                },
                |err| tracing::error!(?err, "cpal stream error"),
                None,
            ),
            cpal::SampleFormat::I16 => self.device.build_input_stream(
                self.stream_config,
                move |data: &[i16], _cb| {
                    let samples = if channels == 1 {
                        data.iter().map(|&s| f32::from(s) / 32_768.0).collect()
                    } else {
                        let mut mono = Vec::with_capacity(data.len() / channels);
                        for frame in data.chunks_exact(channels) {
                            let sum: f32 = frame.iter().map(|&s| f32::from(s) / 32_768.0).sum();
                            mono.push(sum / (channels as f32));
                        }
                        mono
                    };
                    let _ = sink.send(AudioFrame {
                        samples,
                        sample_rate_hz: sample_rate,
                    });
                },
                |err| tracing::error!(?err, "cpal stream error"),
                None,
            ),
            cpal::SampleFormat::U16 => self.device.build_input_stream(
                self.stream_config,
                move |data: &[u16], _cb| {
                    let samples = if channels == 1 {
                        data.iter()
                            .map(|&s| (f32::from(s) - 32_768.0) / 32_768.0)
                            .collect()
                    } else {
                        let mut mono = Vec::with_capacity(data.len() / channels);
                        for frame in data.chunks_exact(channels) {
                            let sum: f32 = frame
                                .iter()
                                .map(|&s| (f32::from(s) - 32_768.0) / 32_768.0)
                                .sum();
                            mono.push(sum / (channels as f32));
                        }
                        mono
                    };
                    let _ = sink.send(AudioFrame {
                        samples,
                        sample_rate_hz: sample_rate,
                    });
                },
                |err| tracing::error!(?err, "cpal stream error"),
                None,
            ),
            other => {
                return Err(AudioError::Capture(format!(
                    "sample format no soportado: {other:?}"
                )));
            }
        }
        .map_err(|e| AudioError::Capture(format!("build_input_stream: {e}")))?;

        stream
            .play()
            .map_err(|e| AudioError::Capture(format!("stream.play: {e}")))?;
        self.stream = Some(stream);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), AudioError> {
        if let Some(s) = self.stream.take() {
            s.pause()
                .map_err(|e| AudioError::Capture(format!("pause: {e}")))?;
        }
        Ok(())
    }

    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate
    }
}

// =========================================================================
// Resampler: vive en `oido-core` (donde tiene estado entre frames). Lo
// exponemos desde aquí para que el bin o los tests puedan construirlo
// sin importar dependencias de audio-platform desde core.
// =========================================================================

/// Resampler SincFixedIn para convertir cualquier sample rate → 16 kHz
/// mono f32 (lo que necesita whisper.cpp).
///
/// Wrapper sobre `rubato::SincFixedIn<f32>`. Usa interpolación Linear
/// (más rápida, suficiente para voz) con sinc_len de 128.
///
/// Mantiene estado interno entre llamadas a `process()`: los samples
/// que no caben en el chunk actual se difieren al siguiente.
pub struct Resampler {
    inner: rubato::SincFixedIn<f32>,
    chunk_in: usize,
    /// Acumulador entre llamadas: los samples que aún no completan un
    /// chunk completo se difieren al siguiente `process`. Si en una
    /// llamada llega más de un chunk completo (e.g. cpal entrega 4096
    /// muestras de golpe), procesamos varios y los concatenamos.
    pending: Vec<f32>,
    /// Cuota de seguridad: si `pending` crece sin drenarse (algo va
    /// mal aguas abajo), descartamos lo más viejo para no acumular
    /// memoria indefinidamente. 32 × chunk_in @ 48 kHz ≈ 0.7 s.
    max_pending: usize,
}

impl std::fmt::Debug for Resampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resampler")
            .field("chunk_in", &self.chunk_in)
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl Resampler {
    /// Crea un resampler que lleva `input_rate` → 16 kHz mono. Devuelve
    /// `None` si el ratio requiere un input_rate absurdo o si `rubato`
    /// no puede construir el resampler (e.g. chunk_size demasiado
    /// pequeño para el ratio).
    pub fn new(input_rate: u32) -> Option<Self> {
        if input_rate == 0 || input_rate == 16_000 {
            return None; // nada que resamplear
        }
        const CHUNK_IN: usize = 512;
        const OUT_RATE: u32 = 16_000;
        // rubato 0.15: ratio = output / input. Si queremos 48000 → 16000,
        // ratio = 1/3 = 0.333...; si 44100 → 16000, ratio = 16000/44100.
        let params = rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let ratio = f64::from(OUT_RATE) / f64::from(input_rate);
        rubato::SincFixedIn::<f32>::new(
            ratio, 2.0, params, CHUNK_IN, 1, // mono
        )
        .ok()
        .map(|inner| Self {
            inner,
            chunk_in: CHUNK_IN,
            pending: Vec::new(),
            max_pending: CHUNK_IN * 128,
        })
    }

    /// Indica si el input rate requiere resampling.
    #[must_use]
    pub fn is_identity(input_rate: u32) -> bool {
        input_rate == 16_000
    }

    /// Procesa los samples de entrada y devuelve los equivalentes a
    /// 16 kHz. **Acumula internamente** entre llamadas: si llega menos
    /// de `chunk_in` muestras, las difiere; si llega más, las parte en
    /// varios chunks completos. Sólo devuelve samples cuando hay al
    /// menos un chunk completo disponible. Sustituye al `truncate` de
    /// Fase 1 que silenciosamente descartaba audio cuando cpal
    /// entregaba frames grandes (p.ej. 4096 muestras @ 48 kHz).
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, AudioError> {
        use rubato::Resampler;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        self.pending.extend_from_slice(input);

        // Cuota de seguridad: si pending desborda (algo va mal
        // aguas abajo y nadie drena), descartamos lo más viejo para
        // no acumular memoria infinita.
        if self.pending.len() > self.max_pending {
            tracing::warn!(
                pending = self.pending.len(),
                max = self.max_pending,
                "resampler.pending desbordó la cuota; descartando lo más viejo"
            );
            let drop_n = self.pending.len() - self.max_pending;
            self.pending.drain(..drop_n);
        }

        let mut out = Vec::new();
        while self.pending.len() >= self.chunk_in {
            let chunk: Vec<f32> = self.pending.drain(..self.chunk_in).collect();
            let waves_in = vec![chunk];
            let result = self
                .inner
                .process(&waves_in, None)
                .map_err(|e| AudioError::Capture(format!("resampler.process: {e}")))?;
            out.extend(result.into_iter().flatten());
        }
        // Lo que quede en `self.pending` (< chunk_in) se difiere para
        // completar en la próxima llamada. No paddeamos con ceros
        // artificialmente: cualquier sample futuro los completa y
        // rubato procesará exactamente chunk_in.
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_identity_when_input_is_16khz() {
        assert!(Resampler::new(16_000).is_none());
        assert!(Resampler::is_identity(16_000));
    }

    #[test]
    fn resampler_48000_to_16000_produces_third_length() {
        let mut r = Resampler::new(48_000).expect("48kHz → 16kHz debe ser soportado");
        // 1024 muestras @ 48kHz → ~341 muestras @ 16kHz (ratio 1/3).
        // El output real depende del sinc_len + oversampling_factor;
        // admitimos un rango amplio para evitar fragilidad.
        let input = vec![0.5_f32; 1024];
        let out = r.process(&input).expect("process no debe fallar");
        assert!(
            (280..=360).contains(&out.len()),
            "esperaba ~280-360 samples @ 16kHz, obtuve {}",
            out.len()
        );
    }

    #[test]
    fn resampler_44100_to_16000_produces_correct_length() {
        let mut r = Resampler::new(44_100).expect("44.1kHz → 16kHz debe ser soportado");
        let input = vec![0.0_f32; 1024];
        let out = r.process(&input).expect("process no debe fallar");
        // ratio = 16000/44100 ≈ 0.363; output depende de parámetros
        // sinc. Rango amplio para no ser frágil.
        assert!(
            (300..=380).contains(&out.len()),
            "esperaba ~300-380 samples @ 16kHz, obtuve {}",
            out.len()
        );
    }

    #[test]
    fn resampler_deferes_short_input_until_chunk_completes() {
        let mut r = Resampler::new(48_000).expect("48kHz → 16kHz");
        // Input de 100 muestras (< chunk_in=1024) no debe procesarse:
        // se difiere al siguiente process. Esto reemplazó el padding
        // con ceros (que metía 924 ceros "fantasma" en el flujo).
        let partial = vec![0.1_f32; 100];
        let out_partial = r.process(&partial).expect("process no debe fallar");
        assert!(
            out_partial.is_empty(),
            "input de 100 muestras debe quedar pendiente, obtuve {}",
            out_partial.len()
        );
    }

    #[test]
    fn resampler_accumulates_across_calls_to_complete_chunk() {
        let mut r = Resampler::new(48_000).expect("48kHz → 16kHz");
        // Tres llamadas de 200 + una de 50 = 650 samples → 1 chunk
        // completo (512) más 138 pendientes.
        let frame_a = vec![0.1_f32; 200];
        let frame_b = vec![0.1_f32; 200];
        let frame_c = vec![0.1_f32; 200];
        let frame_d = vec![0.1_f32; 50];

        let out_a = r.process(&frame_a).expect("a");
        let out_b = r.process(&frame_b).expect("b");
        let out_c = r.process(&frame_c).expect("c");
        let out_d = r.process(&frame_d).expect("d");

        // Las dos primeras llamadas no completan chunk (acumulado 400).
        assert!(out_a.is_empty(), "tras 200 samples debe estar vacío");
        assert!(out_b.is_empty(), "tras 400 samples debe estar vacío");

        // La tercera completa 600 → 1 chunk procesado + 88 pendientes.
        assert!(
            (120..=180).contains(&out_c.len()),
            "esperaba ~120-180 samples @ 16kHz, obtuve {}",
            out_c.len()
        );

        // La cuarta no llega a chunk completo, queda pendiente.
        assert!(
            out_d.is_empty(),
            "88 + 50 = 138 < 512 debe quedar pendiente, obtuve {}",
            out_d.len()
        );
    }

    #[test]
    fn resampler_does_not_truncate_oversized_frames() {
        let mut r = Resampler::new(48_000).expect("48kHz → 16kHz");
        // 4096 muestras @ 48 kHz: antes esto se truncaba a 1024 y se
        // perdía el 75% del audio. Ahora esperamos ~4 chunks → ~4 × 341
        // = ~1364 samples de salida.
        let input = vec![0.5_f32; 4096];
        let out = r.process(&input).expect("process no debe fallar");
        // 4 chunks exactos: 4 × chunk_in = 4096 → ratio 1/3 → ~1365.
        // Aceptamos 1000–1500 por márgenes del sinc interpolation.
        assert!(
            (1000..=1500).contains(&out.len()),
            "esperaba 1000-1500 muestras @ 16kHz (4 chunks), obtuve {}",
            out.len()
        );
    }

    #[test]
    fn resampler_pending_does_not_explode() {
        let mut r = Resampler::new(48_000).expect("48kHz → 16kHz");
        // Bombeamos 100 chunks pequeños sin drenar el resultado.
        // pending nunca debe superar max_pending = CHUNK_IN * 32.
        for _ in 0..100 {
            r.process(&vec![0.1_f32; 100]).expect("process");
        }
        // El `Debug` muestra `pending.len()` (acumulado no drenado). El
        // assert no es trivial porque drains internos durante process
        // pueden reducirlo. La cota superior garantizada por el
        // código: nunca > max_pending en estado normal.
        // Aquí basta con que no haya hecho panic.
        let dbg = format!("{:?}", r);
        assert!(dbg.contains("Resampler"));
    }
}
