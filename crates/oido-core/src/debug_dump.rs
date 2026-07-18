//! Dump de audio post-resample para diagnóstico.
//!
//! Dos canales independientes, ambos opt-in por env y con **cero costo
//! cuando están apagados** (la env se lee una sola vez y se cachea en
//! `OnceLock`):
//!
//! 1. `OIDO_DEBUG_PCM_DUMP=<path>` — appendea cada frame que entra al
//!    buffer de grabación como PCM crudo **f32 little-endian mono
//!    16 kHz**. Sirve para correlacionar timestamps con glitches.
//!    Convertir a WAV para inspección:
//!    ```text
//!    ffmpeg -f f32le -ar 16000 -ac 1 -i dump.pcm dump.wav
//!    ```
//!
//! 2. `OIDO_DEBUG_WAV_DIR=<dir>` — al soltar el hotkey, guarda el
//!    dictado completo como WAV f32 16 kHz mono (`dictado-NNN.wav`).
//!    Sirve para **escuchar** exactamente lo que whisper recibió: si el
//!    WAV suena bien pero la transcripción es mala, el problema es el
//!    modelo; si el WAV suena mal (bajo, cortado, con ruido), el
//!    problema es la captura.
//!
//! Además, `analyze` calcula stats de nivel por dictado (RMS en dBFS,
//! pico, % de casi-ceros) que se loguean siempre con el dictado — un
//! RMS muy bajo (<-45 dBFS) indica señal débil, causa común de
//! alucinaciones de whisper.
//!
//! Nada de esto afecta al pipeline si falla: sólo loguea. Es código de
//! diagnóstico, no de producción.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Stats de nivel por dictado
// ---------------------------------------------------------------------------

/// Stats de nivel de un buffer de audio f32.
#[derive(Debug, Clone, Copy)]
pub struct AudioStats {
    /// RMS en dBFS (0 dBFS = full scale). -inf si el buffer es mudo.
    pub rms_dbfs: f32,
    /// Pico absoluto en dBFS.
    pub peak_dbfs: f32,
    /// Fracción de samples con |x| < 1e-4 (casi-cero). ~1.0 = mudo.
    pub near_zero_frac: f32,
}

/// Calcula RMS/pico/casi-ceros de `samples`. O(n), sin asignaciones.
pub fn analyze(samples: &[f32]) -> AudioStats {
    if samples.is_empty() {
        return AudioStats {
            rms_dbfs: f32::NEG_INFINITY,
            peak_dbfs: f32::NEG_INFINITY,
            near_zero_frac: 1.0,
        };
    }
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f32;
    let mut near_zero = 0usize;
    for &s in samples {
        let a = s.abs();
        sum_sq += f64::from(s) * f64::from(s);
        if a > peak {
            peak = a;
        }
        if a < 1e-4 {
            near_zero += 1;
        }
    }
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    let to_db = |x: f32| {
        if x <= 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * x.log10()
        }
    };
    AudioStats {
        rms_dbfs: to_db(rms),
        peak_dbfs: to_db(peak),
        near_zero_frac: near_zero as f32 / samples.len() as f32,
    }
}

// ---------------------------------------------------------------------------
// Dump PCM continuo (por frame)
// ---------------------------------------------------------------------------

pub struct PcmDumper {
    writer: Mutex<BufWriter<std::fs::File>>,
}

static PCM_DUMPER: OnceLock<Option<PcmDumper>> = OnceLock::new();

/// Devuelve el dumper activo si `OIDO_DEBUG_PCM_DUMP` estaba seteada al
/// primer llamado. Llamadas posteriores retornan la misma instancia (la
/// env se lee una sola vez: cambiarla en caliente no tiene efecto).
pub fn pcm_dumper() -> Option<&'static PcmDumper> {
    PCM_DUMPER
        .get_or_init(|| {
            let path = std::env::var("OIDO_DEBUG_PCM_DUMP").ok()?;
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => {
                    tracing::info!(%path, "PCM dump activo (f32le mono 16kHz)");
                    Some(PcmDumper {
                        writer: Mutex::new(BufWriter::new(f)),
                    })
                }
                Err(e) => {
                    tracing::warn!(%path, ?e, "no pude abrir PCM dump; desactivado");
                    None
                }
            }
        })
        .as_ref()
}

impl PcmDumper {
    /// Appendea `samples` (f32 16 kHz mono) al dump como little-endian.
    /// No propaga errores: sólo loguea (un dump parcial sigue siendo
    /// útil para diagnóstico).
    pub fn write(&self, samples: &[f32]) {
        let mut w = self.writer.lock();
        let mut buf = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        if let Err(e) = w.write_all(&buf) {
            tracing::warn!(?e, "PCM dump write falló");
        }
    }
}

// ---------------------------------------------------------------------------
// Dump WAV por dictado (al soltar el hotkey)
// ---------------------------------------------------------------------------

pub struct WavDumper {
    dir: std::path::PathBuf,
    counter: AtomicU64,
}

static WAV_DUMPER: OnceLock<Option<WavDumper>> = OnceLock::new();

/// Devuelve el dumper WAV activo si `OIDO_DEBUG_WAV_DIR` estaba seteada
/// al primer llamado (el dir se crea si no existe).
pub fn wav_dumper() -> Option<&'static WavDumper> {
    WAV_DUMPER
        .get_or_init(|| {
            let dir = std::env::var("OIDO_DEBUG_WAV_DIR").ok()?;
            let dir = std::path::PathBuf::from(dir);
            if let Err(e) = std::fs::create_dir_all(&dir) {
                tracing::warn!(dir = ?dir, ?e, "no pude crear WAV dump dir; desactivado");
                return None;
            }
            tracing::info!(dir = ?dir, "WAV dump por dictado activo (f32 16kHz mono)");
            Some(WavDumper {
                dir,
                counter: AtomicU64::new(0),
            })
        })
        .as_ref()
}

impl WavDumper {
    /// Guarda el dictado completo como WAV f32 16 kHz mono. El nombre
    /// es `dictado-NNN.wav` con un contador creciente por proceso.
    pub fn write_dictation(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let path = self.dir.join(format!("dictado-{n:03}.wav"));
        match write_wav_f32(&path, samples, 16_000) {
            Ok(()) => tracing::info!(path = ?path, "dictado WAV guardado"),
            Err(e) => tracing::warn!(path = ?path, ?e, "falló guardar dictado WAV"),
        }
    }
}

/// Escribe un WAV RIFF con formato IEEE float (3), mono, `sample_rate`.
fn write_wav_f32(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let data_len = (samples.len() * 4) as u32;
    let mut w = BufWriter::new(std::fs::File::create(path)?);

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&(36u32 + data_len).to_le_bytes())?; // chunk size
    w.write_all(b"WAVE")?;

    // fmt chunk (16 bytes, PCM float)
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    w.write_all(&3u16.to_le_bytes())?; // audio format = IEEE float
    w.write_all(&1u16.to_le_bytes())?; // channels = 1 (mono)
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&(sample_rate * 4).to_le_bytes())?; // byte rate
    w.write_all(&4u16.to_le_bytes())?; // block align (1 ch * 4 bytes)
    w.write_all(&32u16.to_le_bytes())?; // bits per sample

    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_len.to_le_bytes())?;
    let mut buf = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    w.write_all(&buf)?;
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_empty_is_mute() {
        let s = analyze(&[]);
        assert!(s.rms_dbfs.is_infinite() && s.rms_dbfs.is_sign_negative());
        assert_eq!(s.near_zero_frac, 1.0);
    }

    #[test]
    fn analyze_full_scale_sine_near_zero_dbfs() {
        // Seno a amplitud 1.0: RMS = 1/sqrt(2) ≈ -3.01 dBFS, pico = 0 dBFS.
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let s = analyze(&samples);
        assert!(
            (s.rms_dbfs - (-3.01)).abs() < 0.1,
            "RMS esperado ~-3 dBFS, obtuve {}",
            s.rms_dbfs
        );
        assert!(
            s.peak_dbfs.abs() < 0.1,
            "pico esperado ~0 dBFS, obtuve {}",
            s.peak_dbfs
        );
        assert!(s.near_zero_frac < 0.01);
    }

    #[test]
    fn analyze_silence_is_mute() {
        let samples = vec![0.0_f32; 16_000];
        let s = analyze(&samples);
        assert!(s.rms_dbfs.is_infinite() && s.rms_dbfs.is_sign_negative());
        assert_eq!(s.near_zero_frac, 1.0);
    }

    #[test]
    fn write_wav_f32_produces_valid_riff_header() {
        let dir = std::env::temp_dir().join("oido-debug-dump-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let samples = vec![0.5_f32, -0.5, 0.25, -0.25];
        write_wav_f32(&path, &samples, 16_000).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(&bytes[36..40], b"data");
        // 4 samples * 4 bytes = 16 bytes de datos.
        let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data_len, 16);
        assert_eq!(bytes.len(), 44 + 16);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
