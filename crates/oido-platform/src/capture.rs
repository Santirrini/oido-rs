//! `CaptureSource` para todos los OS. Usa `cpal` (cross-platform).
//!
//! MVP: pedimos 16 kHz mono F32 al dispositivo; si no lo soporta,
//! caemos al default y loggeamos una advertencia. Fase 2 añade un
//! resampler (`rubato`) para mezclar bien cualquier sample rate con
//! whisper.cpp que requiere 16 kHz estricto.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};

use crossbeam_channel::Sender;

use crate::traits::{CaptureSource, PlatformError};
use crate::AudioFrame;

pub struct CpalCapture {
    device: cpal::Device,
    stream_config: StreamConfig,
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
        let device = host
            .default_input_device()
            .ok_or_else(|| PlatformError::Capture("sin dispositivo de entrada por defecto".into()))?;

        // Buscar una config 16 kHz mono F32. Si no, fallback al default.
        let mut wanted: Option<StreamConfig> = None;
        let mut sample_format = cpal::SampleFormat::F32;
        if let Ok(supported) = device.supported_input_configs() {
            for cfg in supported.flatten() {
                if cfg.channels() == 1 && cfg.sample_format() == cpal::SampleFormat::F32 {
                    if let Ok(picked) = cfg.with_sample_rate(SampleRate(16_000)) {
                        wanted = Some(picked.config());
                        sample_format = cpal::SampleFormat::F32;
                        break;
                    }
                }
            }
        }
        let (stream_config, sample_format, sample_rate) = match wanted {
            Some(c) => (c, cpal::SampleFormat::F32, 16_000),
            None => {
                let fallback = device
                    .default_input_config()
                    .map_err(|e| PlatformError::Capture(format!("default_input_config: {e}")))?;
                tracing::warn!(
                    requested = 16_000,
                    actual = fallback.sample_rate().0,
                    "dispositivo no soporta 16kHz mono F32; usando default; Fase 2 añade resampler"
                );
                let cfg = fallback.config();
                let rate = fallback.sample_rate().0;
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
        // Mensajes idénticos; el coste en líneas es menor que la
        // alternativa de abstraer el sample-format.
        let stream = match self.sample_format {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                &self.stream_config,
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
                &self.stream_config,
                move |data: &[i16], _cb| {
                    let samples: Vec<f32> =
                        data.iter().map(|&s| f32::from(s) / 32_768.0).collect();
                    let _ = sink.send(AudioFrame {
                        samples,
                        sample_rate_hz: sample_rate,
                    });
                },
                |err| tracing::error!(?err, "cpal stream error"),
                None,
            ),
            cpal::SampleFormat::U16 => self.device.build_input_stream(
                &self.stream_config,
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
        // fallback usado sólo para que el bin compile en tests sin
        // micrófono. La apertura real falla en `new()` si no hay
        // dispositivo.
        Self {
            device: cpal::default_host()
                .default_input_device()
                .expect("sin dispositivo"),
            stream_config: StreamConfig {
                channels: 1,
                sample_rate: SampleRate(16_000),
                buffer_size: cpal::BufferSize::Default,
            },
            sample_format: cpal::SampleFormat::F32,
            sample_rate: 16_000,
            sink: None,
            stream: None,
        }
    }
}
