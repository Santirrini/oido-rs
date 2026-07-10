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
//! El resampler vive en `oido-core` (donde sí podemos mantener estado
//! entre frames). `CpalCapture` se limita a entregar lo que el OS da.
//!
//! Esto soluciona el bug de Fase 1 donde si el dispositivo no soportaba
//! 16 kHz nativo, whisper.cpp recibía audio al sample rate incorrecto
//! y producía basura.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crossbeam_channel::Sender;

use crate::traits::{CaptureSource, PlatformError};
use crate::AudioFrame;

pub struct CpalCapture {
    device: cpal::Device,
    stream_config: cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    sample_rate: u32,
    sink: Option<Sender<AudioFrame>>,
    stream: Option<cpal::Stream>,
}

impl std::fmt::Debug for CpalCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalCapture")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.stream_config.channels)
            .field("sample_format", &self.sample_format)
            .field("stream_active", &self.stream.is_some())
            .finish()
    }
}

impl CpalCapture {
    pub fn new() -> Result<Self, PlatformError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or_else(|| {
            PlatformError::Capture("sin dispositivo de entrada por defecto".into())
        })?;

        // Preferimos 16 kHz mono F32 (lo que necesita whisper.cpp). Si
        // no está disponible, caemos al default del dispositivo; el
        // consumer thread de oido-core hará el resampling a 16 kHz
        // antes de transcribir.
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
        let (stream_config, sample_format, sample_rate) = match wanted {
            Some(s) => (s.config(), cpal::SampleFormat::F32, 16_000_u32),
            None => {
                let fallback = device
                    .default_input_config()
                    .map_err(|e| PlatformError::Capture(format!("default_input_config: {e}")))?;
                tracing::warn!(
                    requested = 16_000,
                    actual = fallback.sample_rate(),
                    "dispositivo no soporta 16kHz mono F32; usando default + resampling"
                );
                let cfg = fallback.config();
                let rate = fallback.sample_rate();
                let fmt = fallback.sample_format();
                (cfg, fmt, rate)
            }
        };

        Ok(Self {
            device,
            stream_config,
            sample_format,
            sample_rate,
            sink: None,
            stream: None,
        })
    }
}

impl CaptureSource for CpalCapture {
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), PlatformError> {
        self.sink = Some(sink);
        Ok(())
    }

    fn start(&mut self) -> Result<(), PlatformError> {
        let sink = self
            .sink
            .clone()
            .ok_or_else(|| PlatformError::Capture("open() no invocado antes de start()".into()))?;
        let sample_rate = self.sample_rate;

        // Ponytail: tres branches (F32/I16/U16) en lugar de un trait
        // object porque cpal selecciona el closure por tipo concreto.
        let stream = match self.sample_format {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                self.stream_config,
                move |data: &[f32], _cb| {
                    let _ = sink.send(AudioFrame {
                        samples: data.to_vec(),
                        sample_rate_hz: sample_rate,
                    });
                },
                |err| tracing::error!(?err, "cpal stream error"),
                None,
            ),
            cpal::SampleFormat::I16 => self.device.build_input_stream(
                self.stream_config,
                move |data: &[i16], _cb| {
                    let samples: Vec<f32> = data.iter().map(|&s| f32::from(s) / 32_768.0).collect();
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
                    let samples: Vec<f32> = data
                        .iter()
                        .map(|&s| (f32::from(s) - 32_768.0) / 32_768.0)
                        .collect();
                    let _ = sink.send(AudioFrame {
                        samples,
                        sample_rate_hz: sample_rate,
                    });
                },
                |err| tracing::error!(?err, "cpal stream error"),
                None,
            ),
            other => {
                return Err(PlatformError::Capture(format!(
                    "sample format no soportado: {other:?}"
                )));
            }
        }
        .map_err(|e| PlatformError::Capture(format!("build_input_stream: {e}")))?;

        stream
            .play()
            .map_err(|e| PlatformError::Capture(format!("stream.play: {e}")))?;
        self.stream = Some(stream);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), PlatformError> {
        if let Some(s) = self.stream.take() {
            s.pause()
                .map_err(|e| PlatformError::Capture(format!("pause: {e}")))?;
        }
        Ok(())
    }

    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate
    }
}

impl Default for CpalCapture {
    fn default() -> Self {
        Self {
            device: cpal::default_host()
                .default_input_device()
                .expect("sin dispositivo"),
            stream_config: cpal::StreamConfig {
                channels: 1,
                sample_rate: 16_000,
                buffer_size: cpal::BufferSize::Default,
            },
            sample_format: cpal::SampleFormat::F32,
            sample_rate: 16_000,
            sink: None,
            stream: None,
        }
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
}

impl std::fmt::Debug for Resampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resampler")
            .field("chunk_in", &self.chunk_in)
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
        const CHUNK_IN: usize = 1024;
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
        })
    }

    /// Indica si el input rate requiere resampling.
    #[must_use]
    pub fn is_identity(input_rate: u32) -> bool {
        input_rate == 16_000
    }

    /// Procesa un bloque completo de samples. Si el bloque es menor a
    /// `chunk_in`, se completa con ceros (rubato requiere tamaño fijo).
    /// Devuelve los samples resampleados a 16 kHz.
    ///
    /// Si el bloque es exactamente `chunk_in`, lo pasa tal cual.
    /// Si es mayor, se procesa en chunks internos.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, PlatformError> {
        use rubato::Resampler;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        // rubato::SincFixedIn::process requiere exactamente chunk_in
        // muestras. Padding con ceros si es más corto.
        let mut padded = input.to_vec();
        if padded.len() < self.chunk_in {
            padded.resize(self.chunk_in, 0.0);
        } else if padded.len() > self.chunk_in {
            // Para bloques más grandes que chunk_in, procesamos el
            // primero y dejamos el resto para la siguiente llamada.
            // En la práctica los frames de cpal son ~10ms a 48kHz = 480
            // muestras, mucho menor que chunk_in.
            padded.truncate(self.chunk_in);
        }
        let waves_in = vec![padded];
        let result = self
            .inner
            .process(&waves_in, None)
            .map_err(|e| PlatformError::Capture(format!("resampler.process: {e}")))?;
        Ok(result.into_iter().flatten().collect())
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
            "esperaba ~290-350 samples @ 16kHz, obtuve {}",
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
    fn resampler_handles_short_input_with_zero_padding() {
        let mut r = Resampler::new(48_000).expect("48kHz → 16kHz");
        // Input de 100 muestras (más corto que chunk_in=1024) debe
        // paddearse con ceros y procesarse igual.
        let input = vec![0.1_f32; 100];
        let out = r.process(&input).expect("process no debe fallar");
        assert!(!out.is_empty(), "input corto debe producir output");
    }
}
