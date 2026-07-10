//! Motor de streaming de audio STT usando el algoritmo LocalAgreement-2.
//!
//! Regla R2: Toda la superficie expuesta y lógica de este archivo es 100% safe Rust.
//! El FFI inseguro se aísla mediante las llamadas seguras provistas por `whisper-rs`.

use std::path::Path;
use std::sync::Arc;
use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};
use crate::SttError;

/// Representa el resultado parcial de una transcripción por streaming.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PartialTranscript {
    /// Texto confirmado estable que ya puede ser inyectado/tipeado.
    pub confirmed: String,
    /// Texto tentativo e inestable de la ventana actual (preview).
    pub unconfirmed: String,
}

/// Contrato para transcriptores en streaming. Permite desacoplar y mockear
/// la etapa de inferencia incremental en los tests de integración.
pub trait Streamer: Send + std::fmt::Debug {
    fn process(&mut self, audio: &[f32]) -> Result<PartialTranscript, SttError>;
    fn flush_final(&mut self) -> Result<PartialTranscript, SttError>;
    fn reset(&mut self);
}

/// Transcriptor en streaming basado en LocalAgreement-2.
/// Mantiene el estado de los tokens de pasadas previas para confirmar
/// prefijos estables de forma incremental.
#[derive(Debug)]
pub struct LocalAgreementStreamer {
    ctx: Option<Arc<WhisperContext>>,
    /// WhisperState persistente (Bug 2). Se crea UNA sola vez por streamer y se
    /// reutiliza en todas las pasadas, evitando el coste de `whisper_init_state`
    /// (~330 MB de KV cache/encoder/decoder) en cada tick.
    /// Es `Option` para permitir `Clone` sin error (el state no es `Clone`).
    state: Option<WhisperState>,
    language: Option<String>,
    gpu_config: crate::GpuConfig,
    n_threads: u16,

    // Estado interno del algoritmo LocalAgreement-2
    prev_tokens: Vec<i32>,
    confirmed_count: usize,
}

impl Clone for LocalAgreementStreamer {
    fn clone(&self) -> Self {
        // WhisperState aloja ~330 MB de buffers (KV cache, encoder, decoder) y NO
        // implementa Clone. El trait Clone no puede fallar, así que dejamos el
        // state en None y se crea perezosamente en el primer `process()` del clon.
        // Compartimos el WhisperContext (Arc) para no recargar el modelo de disco.
        Self {
            ctx: self.ctx.clone(),
            state: None,
            language: self.language.clone(),
            gpu_config: self.gpu_config,
            n_threads: self.n_threads,
            prev_tokens: Vec::new(),
            confirmed_count: 0,
        }
    }
}

impl LocalAgreementStreamer {
    /// Crea un nuevo streamer.
    pub fn new(language: Option<String>, gpu_config: crate::GpuConfig, n_threads: u16) -> Self {
        Self {
            ctx: None,
            state: None,
            language,
            gpu_config,
            n_threads,
            prev_tokens: Vec::new(),
            confirmed_count: 0,
        }
    }

    /// Carga el modelo Whisper en memoria y crea el WhisperState persistente.
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), SttError> {
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

        // Bug 2: creamos el WhisperState UNA sola vez aquí. Antes se recreaba
        // en cada process() (~330 MB/pasada). WhisperState es Send + Sync.
        let state = ctx
            .create_state()
            .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;

        self.ctx = Some(Arc::new(ctx));
        self.state = Some(state);
        Ok(())
    }

    /// Precalienta el modelo realizando una pasada de inferencia sobre silencio.
    pub fn warm_up(&self) -> Result<(), SttError> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(std::path::PathBuf::from("<LocalAgreementStreamer: modelo no cargado>"))
        })?;

        // 30 segundos de silencio para inicializar
        let silence = vec![0.0_f32; 16_000 * 30];
        let mut state = ctx.create_state().map_err(|e| SttError::Backend(format!("create_state: {e}")))?;
        let params = crate::whisper_cpp::build_streaming_params(self.language.as_deref(), self.n_threads);
        state.full(params, &silence).map_err(|e| SttError::Backend(format!("full: {e}")))?;

        Ok(())
    }

    /// Procesa una ventana de audio acumulada. Devuelve la diferencia
    /// recién confirmada y el texto tentativo restante.
    pub fn process(&mut self, audio: &[f32]) -> Result<PartialTranscript, SttError> {
        if audio.len() < 4_800 {
            // whisper.cpp requiere al menos 300 ms de audio para procesar
            return Ok(PartialTranscript {
                confirmed: String::new(),
                unconfirmed: self.get_unconfirmed_text()?,
            });
        }

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(std::path::PathBuf::from("<LocalAgreementStreamer: modelo no cargado>"))
        })?;

        // Bug 2: reutilizar el WhisperState persistente. En clones recién creados
        // (state == None) se inicializa perezosamente en el primer tick.
        if self.state.is_none() {
            let state = ctx
                .create_state()
                .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;
            self.state = Some(state);
        }
        let state = self.state.as_mut().expect("state inicializado arriba");

        let mut params = crate::whisper_cpp::build_streaming_params(self.language.as_deref(), self.n_threads);

        // Ajustar dinámicamente la ventana del contexto de audio
        let audio_secs = audio.len() as f32 / 16_000.0;
        let ctx_frames = (((audio_secs + 2.0) * 50.0) as i32).clamp(100, 1500);
        params.set_audio_ctx(ctx_frames);

        state.full(params, audio).map_err(|e| SttError::Backend(format!("full: {e}")))?;

        // Bug 1: filtrar tokens especiales. Los tokens de texto real son [0, eot_id).
        // Todo token >= eot_id es especial ([_EOT_], [_BEG_], [_TT_xxx]) y se descarta.
        // Antes se usaba n_vocab (51865 en multilingüe), pero los especiales
        // (50257..50364+) son menores y pasaban el filtro.
        let eot_id = ctx.token_eot();
        let mut curr_tokens = Vec::new();
        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            for j in 0..seg.n_tokens() {
                if let Some(token) = seg.get_token(j) {
                    let tid = token.token_id();
                    if tid >= 0 && tid < eot_id {
                        curr_tokens.push(tid);
                    }
                }
            }
        }

        // Calcular el prefijo común más largo (LCP) con la pasada anterior
        let agreed_len = longest_common_prefix(&curr_tokens, &self.prev_tokens);

        let confirmed_text = if agreed_len > self.confirmed_count {
            let new_tokens = &curr_tokens[self.confirmed_count..agreed_len];
            let txt = tokens_to_string(ctx, new_tokens)?;
            self.confirmed_count = agreed_len;
            txt
        } else {
            String::new()
        };

        // Guardar para la próxima iteración
        self.prev_tokens = curr_tokens;

        let unconfirmed_text = if self.prev_tokens.len() > self.confirmed_count {
            tokens_to_string(ctx, &self.prev_tokens[self.confirmed_count..])?
        } else {
            String::new()
        };

        Ok(PartialTranscript {
            confirmed: confirmed_text,
            unconfirmed: unconfirmed_text,
        })
    }

    /// Confirma todos los tokens restantes y limpia el estado para el siguiente dictado.
    pub fn flush_final(&mut self) -> Result<PartialTranscript, SttError> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(std::path::PathBuf::from("<LocalAgreementStreamer: modelo no cargado>"))
        })?;

        // En la pasada final confirmamos absolutamente todo lo que queda
        let confirmed_text = if self.prev_tokens.len() > self.confirmed_count {
            let remaining = &self.prev_tokens[self.confirmed_count..];
            tokens_to_string(ctx, remaining)?
        } else {
            String::new()
        };

        self.reset();

        Ok(PartialTranscript {
            confirmed: confirmed_text,
            unconfirmed: String::new(),
        })
    }

    /// Resetea el estado del algoritmo LA-2 (tokens acumulados y contadores).
    /// El WhisperState persistente NO se destruye (Bug 2): se reutiliza en el
    /// siguiente dictado, evitando realocar ~330 MB de KV cache.
    pub fn reset(&mut self) {
        self.prev_tokens.clear();
        self.confirmed_count = 0;
    }

    fn get_unconfirmed_text(&self) -> Result<String, SttError> {
        let Some(ref ctx) = self.ctx else {
            return Ok(String::new());
        };
        if self.prev_tokens.len() > self.confirmed_count {
            tokens_to_string(ctx, &self.prev_tokens[self.confirmed_count..])
        } else {
            Ok(String::new())
        }
    }
}

impl Streamer for LocalAgreementStreamer {
    fn process(&mut self, audio: &[f32]) -> Result<PartialTranscript, SttError> {
        self.process(audio)
    }

    fn flush_final(&mut self) -> Result<PartialTranscript, SttError> {
        self.flush_final()
    }

    fn reset(&mut self) {
        self.reset();
    }
}

/// Calcula la longitud del prefijo común más largo entre dos rodajas de enteros.
pub fn longest_common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Convierte una secuencia de token IDs a un String UTF-8 válido usando el WhisperContext.
fn tokens_to_string(ctx: &WhisperContext, tokens: &[i32]) -> Result<String, SttError> {
    let mut bytes = Vec::new();
    for &tok in tokens {
        if let Ok(b) = ctx.token_to_bytes(tok) {
            bytes.extend_from_slice(b);
        }
    }
    String::from_utf8(bytes).map_err(|e| SttError::Backend(format!("invalid utf8 from tokens: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest_common_prefix() {
        assert_eq!(longest_common_prefix(&[], &[]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[]), 0);
        assert_eq!(longest_common_prefix(&[], &[1, 2, 3]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(longest_common_prefix(&[1, 2, 3, 5], &[1, 2, 3, 4]), 3);
    }

    #[test]
    fn test_streamer_reset() {
        let mut streamer = LocalAgreementStreamer::new(None, crate::GpuConfig { use_gpu: false, flash_attn: false }, 1);
        streamer.prev_tokens = vec![1, 2, 3];
        streamer.confirmed_count = 2;
        streamer.reset();
        assert!(streamer.prev_tokens.is_empty());
        assert_eq!(streamer.confirmed_count, 0);
    }
}
