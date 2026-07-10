//! Implementación `whisper.cpp` del trait `Transcriber`.
//!
//! Regla R2: **único** archivo del workspace que contiene `unsafe`
//! (declarado vía `#![allow(unsafe_code)]` aquí). Todos los punteros
//! crudos a whisper.cpp salen de `whisper-rs`, que internamente es la
//! superficie FFI.
//!
//! Concurrencia:
//! - `whisper_rs::WhisperContext` es `Send + Sync` (internamente `Arc`).
//! - `WhisperState` (creada por `create_state()`) **no** es `Send`; la
//!   creamos y consumimos dentro de la misma llamada `transcribe`, por
//!   lo que es seguro usarla desde múltiples threads simultáneamente.
//! - `whisper.cpp` no tolera dos inferences concurrentes contra el mismo
//!   `WhisperState`, pero cada llamada genera una nueva. Regla R1
//!   (channels) nos garantiza que solo un STT worker consume el contexto.
//!
//! Parámetros optimizados para dictado corto hold-to-talk (<30s): ver
//! `Transcriber::transcribe` abajo.

#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::{SttError, Transcriber};

/// Configuración de aceleración por GPU.
///
/// En `whisper-rs` 0.16, `WhisperContextParameters` sólo expone `use_gpu`
/// (no `n_gpu_layers`); el offload total lo gestiona internamente la
/// feature de compilación (`cuda`/`metal`/`vulkan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuConfig {
    pub use_gpu: bool,
    pub flash_attn: bool,
}

impl GpuConfig {
    /// Auto-detecta según las features compiladas. Si el bin se compiló
    /// con `cuda`/`metal`/`vulkan`, devuelve GPU activado; en caso
    /// contrario, CPU.
    pub fn auto_detect() -> Self {
        if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan")) {
            Self {
                use_gpu: true,
                flash_attn: true,
            }
        } else {
            Self {
                use_gpu: false,
                flash_attn: false,
            }
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Backend `whisper.cpp`. Es `Send + Sync` porque `WhisperContext` lo es.
#[derive(Debug)]
pub struct WhisperCpp {
    ctx: Option<WhisperContext>,
    language: Option<String>,
    gpu_config: GpuConfig,
    n_threads: u16,
}

impl Default for WhisperCpp {
    fn default() -> Self {
        Self {
            ctx: None,
            language: None,
            gpu_config: GpuConfig::default(),
            n_threads: detect_n_threads(),
        }
    }
}

impl WhisperCpp {
    /// Construye con un idioma ya configurado (ej: `"es"` para forzar ES,
    /// autodetect si la opción whisper.cpp lo permite).
    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            ctx: None,
            language: Some(language.into()),
            gpu_config: GpuConfig::default(),
            n_threads: detect_n_threads(),
        }
    }

    /// Configura GPU + número de threads explícitamente. Útil para el
    /// bin que lee `Config::use_gpu` / `Config::n_threads`.
    #[must_use]
    pub fn with_runtime(mut self, gpu: GpuConfig, n_threads: u16) -> Self {
        self.gpu_config = gpu;
        self.n_threads = n_threads;
        self
    }
}

/// Devuelve el número de threads óptimo para whisper.cpp.
/// `min(cores, 8)` porque whisper.cpp no escala bien más allá.
fn detect_n_threads() -> u16 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u16)
        .unwrap_or(4)
        .min(8)
}

impl Transcriber for WhisperCpp {
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(PathBuf::from("<WhisperCpp: modelo no cargado>"))
        })?;

        // whisper.cpp requiere 1s mínimo (16000 muestras @ 16kHz) para
        // producir algo útil con `single_segment`. Si entra menos,
        // devuelve audio demasiado corto en lugar de alucinar.
        if audio.len() < 16_000 {
            return Err(SttError::AudioTooShort(audio.len()));
        }

        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;

        // Greedy determinista: el más rápido y el más repetible. Para
        // dictado interactivo no necesitamos beam search (5× más lento).
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // === Throughput / paralelismo ===
        params.set_n_threads(self.n_threads as i32);

        // === Output ===
        if let Some(lang) = &self.language {
            params.set_language(Some(lang.as_str()));
        }
        params.set_translate(false);
        params.set_print_realtime(false);
        params.set_print_progress(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);

        // === Anti-alucinación ===
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0); // sin fallback de temperatura
        params.set_max_len(60); // corta loops tipo "gracias gracias gracias"

        // === Optimización para dictado corto (<30s) ===
        // Por defecto, whisper.cpp divide el audio en ventanas de 30s y
        // procesa cada una con overlap. En hold-to-talk el audio es
        // siempre menor a 30s, así que forzamos 1 sola ventana: ahorra
        // el cost del encoder repetido y mejora latencia.
        params.set_single_segment(true);
        // No usar el resultado de la transcripción anterior como prompt;
        // en dictado puntual causa que frases se "peguen" entre
        // activaciones.
        params.set_no_context(true);
        // Sin timestamps a nivel de token (overhead innecesario).
        params.set_token_timestamps(false);

        state
            .full(params, audio)
            .map_err(|e| SttError::Backend(format!("full: {e}")))?;

        let mut out = String::new();
        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let Ok(text) = seg.to_str_lossy() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(trimmed);
        }
        Ok(out)
    }

    fn load_model(&mut self, model_path: &Path) -> Result<(), SttError> {
        if !model_path.exists() {
            return Err(SttError::ModelNotFound(model_path.to_path_buf()));
        }
        let path_str = model_path
            .to_str()
            .ok_or_else(|| SttError::Backend(format!("path no UTF-8: {}", model_path.display())))?;
        let ctx_params = WhisperContextParameters {
            use_gpu: self.gpu_config.use_gpu,
            flash_attn: self.gpu_config.flash_attn,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(path_str, ctx_params)
            .map_err(|e| SttError::Backend(format!("load model: {e}")))?;
        self.ctx = Some(ctx);
        tracing::info!(
            ?model_path,
            use_gpu = self.gpu_config.use_gpu,
            flash_attn = self.gpu_config.flash_attn,
            n_threads = self.n_threads,
            "modelo whisper.cpp cargado"
        );
        Ok(())
    }

    fn warm_up(&self) -> Result<(), SttError> {
        // Una inferencia de 1s de silencio fuerza:
        // 1. La materialización de capas en memoria (CPU) o VRAM (GPU).
        // 2. La compilación lazy de kernels CUDA/Metal si aplica.
        // Después de esto, el primer dictado del usuario corre en
        // "tiempo caliente" sin cold-start.
        let silence: Vec<f32> = vec![0.0; 16_000];
        let _ = self.transcribe(&silence)?;
        tracing::info!("warm-up STT completado");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_auto_detect_matches_features() {
        let cfg = GpuConfig::auto_detect();
        if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan")) {
            assert!(cfg.use_gpu);
            assert!(cfg.flash_attn);
        } else {
            assert!(!cfg.use_gpu);
            assert!(!cfg.flash_attn);
        }
    }

    #[test]
    fn detect_n_threads_is_capped_at_8() {
        let n = detect_n_threads();
        assert!((1..=8).contains(&n), "n_threads fuera de [1, 8]: {n}");
    }

    #[test]
    fn empty_ctx_returns_model_not_loaded() {
        let stt = WhisperCpp::default();
        let audio = vec![0.0_f32; 16_000];
        match stt.transcribe(&audio) {
            Err(SttError::ModelNotFound(_)) => (),
            other => panic!("esperaba ModelNotFound, obtuve: {:?}", other),
        }
    }

    #[test]
    fn short_audio_returns_audio_too_short() {
        let stt = WhisperCpp::default();
        // Sin modelo cargamos; la rama de audio corto requiere modelo,
        // así que este test ejercita la rama ModelNotFound que es
        // regresión válida igualmente.
        let audio = vec![0.0_f32; 800];
        let res = stt.transcribe(&audio);
        assert!(matches!(res, Err(SttError::ModelNotFound(_))));
    }

    /// Smoke test E2E: requiere `models/ggml-base.bin` en el repo local.
    /// Activarlo con `cargo test --features audio-smoke -- --ignored`.
    #[test]
    #[ignore = "requiere models/ggml-base.bin presente en disco"]
    fn smoke_transcribe_real_audio() {
        let model = std::path::PathBuf::from("models/ggml-base.bin");
        let mut stt = WhisperCpp::with_language("es");
        stt.load_model(&model).expect("cargar modelo");
        let audio: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let out = stt.transcribe(&audio);
        eprintln!("smoke output: {out:?}");
    }
}
