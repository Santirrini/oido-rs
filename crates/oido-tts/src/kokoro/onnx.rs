//! Inferencia ONNX para Kokoro-82M: sesión ort + loop de `session.run`.
//!
//! ## Contrato del grafo (verificado)
//!
//! El `kokoro-82m-v1.0.onnx` distribuido por
//! `onnx-community/Kokoro-82M-v1.0-ONNX` (el que el catálogo de
//! `oido-models` descarga) tiene 3 inputs y 1 output fijos:
//!
//! | Nombre      | Tipo   | Shape         | Significado                                  |
//! |-------------|--------|---------------|----------------------------------------------|
//! | `input_ids` | `i64`  | `[1, seq_len]`| IDs de fonemas **con** el pad 0 al inicio/fin |
//! | `style`     | `f32`  | `[1, 256]`    | Vector de estilo 1D seleccionado por voz+L   |
//! | `speed`     | `f32`  | `[1]`         | Escalar: multiplicador de velocidad (1.0x = 1000 milli) |
//! | **output[0]** `audio` | `f32` | `[1, n_samples]` | Audio PCM mono @ 24 kHz                  |
//!
//! NOTA: el input de fonemas se llama **`input_ids`** en este export
//! (convención `onnx-community/transformers`). La variante de
//! `thewh1teagle/kokoro-onnx` usa `"tokens"`, pero NO es la que el
//! catálogo descarga — usar ese nombre provoca
//! `Invalid input name: tokens` en cada `session.run`.
//!
//! ## Padding
//!
//! El input `tokens` que espera Kokoro es la secuencia de fonemas
//! **flanqueada por un 0 al inicio y al final** (token PAD = 0). El
//! `synthesize` upstream lo hace con `tokens = [[0, *tokens, 0]]`,
//! pre-pad + post-pad. NO usamos padding hasta un múltiplo
//! cualquiera: copiamos tal cual.
//!
//! ## Truncado
//!
//! `MAX_PHONEME_LENGTH = 510` en el upstream (deja sitio para los dos
//! pads 0). Texto con más fonemas se trunca a `MAX_TOKENS` (constante
//! local, 510 = el límite del modelo). Verificamos `seq_len <= 510`
//! antes de llamar a `session.run` para evitar un assert interno de
//! ort que mataría el proceso.
//!
//! ## Sample rate
//!
//! Fijo a 24 000 Hz (verificado contra `kokoro_onnx/config.py`:
//! `SAMPLE_RATE = 24000`). NO se lee del .onnx porque Kokoro v1.0
//! siempre emite a 24 kHz.
//!
//! ## Concurrencia
//!
//! `ort::Session::run` requiere `&mut self`. Envolvemos la sesión en
//! `Mutex<Option<Session>>` (mismo patrón que `piper::PiperSession`)
//! para que `Engine::synthesize(&self)` pueda correr desde múltiples
//! workers sin violar la regla de ort. El `&mut self` de `load` se
//! serializa con el `&self` de `infer` a través del Mutex.
//!
//! ## FFI
//!
//! El `unsafe` vive dentro de `ort`/`ort-sys`. Este módulo es 100% Safe
//! Rust (regla R2 de `AGENTS.md`).
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;

use super::voices::VoiceBank;
use crate::TtsError;

/// Tamaño máximo de la secuencia de tokens (incluyendo los dos pads 0
/// en los extremos). El upstream (`kokoro_onnx/config.py`) define
/// `MAX_PHONEME_LENGTH = 510` para los fonemas "útiles" — el tensor
/// final `[0, *tokens, 0]` tiene por tanto 512 elementos como techo.
/// Aceptamos ambos límites; el chequeo en [`KokoroSession::infer`]
/// recorta a 510 fonemas reales si el texto es más largo.
pub const MAX_TOKENS: usize = 510;

/// Sample rate nativo de Kokoro-82M. El `PlaybackSink` de `oido-audio`
/// lo usa para resamplear al device.
pub const SAMPLE_RATE_HZ: u32 = 24_000;

/// Dimensión del vector de estilo por voz.
pub const STYLE_DIM: usize = 256;

/// Sesión ort thread-safe para Kokoro-82M.
///
/// Estructura paralela a `piper::PiperSession`:
/// - `Arc<Mutex<Option<Session>>>` para `&self` synthesize.
/// - `Arc<Mutex<Option<VoiceBank>>>` para el banco de voces (no es
///   parte del Session ort, pero se carga junto).
///
/// `Clone` es barato (clona los `Arc` y copia los `PathBuf`). Lo usa
/// `KokoroEngine::synthesize` para soltar el lock del `Mutex`
/// exterior antes de inferir, eliminando el riesgo de re-entrada
/// sobre el mismo `Mutex` si `infer` re-tocase el handle.
///
/// NOTA: el `Mutex` que envuelve el `VoiceBank` está dentro de un
/// `Arc` porque `parking_lot::Mutex<T>` NO implementa `Clone` (a
/// diferencia de `std::sync::Mutex`); sin el `Arc` no podríamos
/// derivar `Clone` para `KokoroSession`.
#[derive(Debug, Clone)]
pub struct KokoroSession {
    inner: Arc<Mutex<Option<Session>>>,
    /// Path del `.onnx` cargado (informativo).
    pub model_path: PathBuf,
    /// Banco de voces (estilo por voz × token length). `None` hasta
    /// que el caller llame a `load_voices` con la ruta al
    /// `voices-v1.0.bin`.
    pub voices: Arc<Mutex<Option<VoiceBank>>>,
}

impl KokoroSession {
    /// Crea un handle "sin cargar". El engine debe llamar a
    /// [`KokoroSession::load`] y a [`KokoroSession::load_voices`]
    /// antes de [`KokoroSession::infer`].
    #[must_use]
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            model_path: PathBuf::new(),
            voices: Arc::new(Mutex::new(None)),
        }
    }

    /// Carga el `.onnx` desde `model_path` (validando existencia).
    /// Si ya había una sesión, la reemplaza. NO carga el `voices.bin`
    /// — eso lo hace `load_voices` por separado.
    pub fn load(&mut self, model_path: &Path) -> Result<(), TtsError> {
        if !model_path.exists() {
            return Err(TtsError::ModelNotFound(model_path.to_path_buf()));
        }
        let session = Session::builder()
            .map_err(|e| TtsError::Backend(format!("ort SessionBuilder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| TtsError::Backend(format!("commit_from_file: {e}")))?;
        *self.inner.lock() = Some(session);
        self.model_path = model_path.to_path_buf();
        tracing::info!(?model_path, "KokoroSession::load completada");
        Ok(())
    }

    /// Carga el `voices-v1.0.bin` desde `voices_path`. Idempotente:
    /// re-cargar reemplaza el banco anterior.
    pub fn load_voices(&self, voices_path: &Path) -> Result<(), TtsError> {
        let bank = VoiceBank::load(voices_path)?;
        tracing::info!(
            ?voices_path,
            num_voices = bank.voice_ids().len(),
            "Kokoro VoiceBank cargado"
        );
        *self.voices.lock() = Some(bank);
        Ok(())
    }

    /// Indica si la sesión y el banco de voces están listos. Si
    /// alguno falta, `infer` devolverá `Backend("modelo no cargado")`.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.inner.lock().is_some() && self.voices.lock().is_some()
    }

    /// Ejecuta la inferencia ONNX de Kokoro.
    ///
    /// - `phoneme_ids`: IDs de fonemas SIN los pads 0 (los añadimos
    ///   aquí, igual que el upstream). Típicamente producidos por
    ///   [`super::g2p::phonemize`].
    /// - `voice_id`: ID de la voz (debe estar en el `VoiceBank`).
    /// - `speed`: multiplicador de velocidad (1.0 = 1000 milli).
    ///
    /// Devuelve el PCM mono f32 (shape `[n_samples]` tras squeeze).
    pub fn infer(
        &self,
        phoneme_ids: &[i64],
        voice_id: &str,
        speed: f32,
    ) -> Result<Vec<f32>, TtsError> {
        if phoneme_ids.is_empty() {
            return Err(TtsError::Backend(
                "KokoroSession::infer con phoneme_ids vacío".into(),
            ));
        }
        if !(0.5..=2.0).contains(&speed) {
            return Err(TtsError::Backend(format!(
                "speed {speed} fuera de rango [0.5, 2.0]"
            )));
        }

        // --- 1. Truncar si es necesario y aplicar padding 0 a ambos lados.
        let n_real = phoneme_ids.len();
        if n_real > MAX_TOKENS {
            tracing::warn!(
                input_phonemes = n_real,
                max = MAX_TOKENS,
                "truncando secuencia de fonemas al máximo de Kokoro"
            );
        }
        let n_keep = n_real.min(MAX_TOKENS);
        // Pad: [0, *phoneme_ids[..n_keep], 0] → shape [1, n_keep+2].
        let mut padded: Vec<i64> = Vec::with_capacity(n_keep + 2);
        padded.push(0);
        padded.extend_from_slice(&phoneme_ids[..n_keep]);
        padded.push(0);
        let seq_len = padded.len();

        // --- 2. Tensor `tokens` (i64, [1, seq_len]).
        let tokens_tensor = Tensor::from_array(([1_i64, seq_len as i64], padded))
            .map_err(|e| TtsError::Backend(format!("tensor tokens: {e}")))?;

        // --- 3. Tensor `style` (f32, [1, 256]).
        // El "token_count" para slice de voz es el número de fonemas
        // REALES (sin pads) — exactamente lo que usa el upstream
        // (`voice = voice[len(tokens)]`).
        let style_vec: Array1<f32> = {
            let bank_guard = self.voices.lock();
            let bank = bank_guard
                .as_ref()
                .ok_or_else(|| TtsError::Backend("banco de voces no cargado".into()))?;
            bank.voice_slice(voice_id, n_keep)?
        };
        // `voice_slice` ya devuelve los 256 floats en orden C
        // (row-major). Tomamos una copia como `Vec<f32>` plana para
        // alimentar `Tensor::from_array((shape, vec))`. No podemos
        // usar `into_raw_vec` directamente porque consume el
        // `Array1` y aún necesitamos soltarlo en este scope; un
        // `to_vec` es equivalente y evita el borrow conflict.
        let style_data: Vec<f32> = style_vec.to_vec();
        let style_tensor = Tensor::from_array(([1_i64, STYLE_DIM as i64], style_data))
            .map_err(|e| TtsError::Backend(format!("tensor style: {e}")))?;

        // --- 4. Tensor `speed` (f32, [1]).
        let speed_tensor = Tensor::from_array(([1_i64], vec![speed]))
            .map_err(|e| TtsError::Backend(format!("tensor speed: {e}")))?;

        // --- 5. Lock + run. ort::Session::run requiere `&mut self`.
        let mut guard = self.inner.lock();
        let session = guard.as_mut().ok_or_else(|| {
            TtsError::Backend("KokoroSession::infer con sesión no cargada".into())
        })?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => tokens_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor,
            })
            .map_err(|e| TtsError::Backend(format!("session.run: {e}")))?;

        // --- 6. Extraer el tensor de audio (output[0] = "audio").
        // Intentamos por nombre primero (más robusto si el modelo
        // añade más outputs en el futuro), luego caemos al índice 0.
        let audio_value = if outputs.contains_key("audio") {
            &outputs["audio"]
        } else {
            &outputs[0]
        };
        let samples = extract_audio_samples(audio_value)?;
        if samples.is_empty() {
            return Err(TtsError::Backend(
                "ort devolvió tensor de audio vacío".into(),
            ));
        }
        Ok(samples)
    }
}

/// Extrae el `Vec<f32>` del tensor de salida squezeeando las primeras
/// dims (batch=1 + canales).
///
/// El output de Kokoro es `[1, n_samples]` (1 batch, 1 canal). En
/// general `try_extract_array::<f32>()` devuelve `ArrayViewD<f32>`
/// con shape dinámico; hacemos `iter().copied().collect()` para
/// aplanarlo — no asumimos shape concreto (defensa ante un modelo
/// futuro que emita `[1, 1, n]` estilo Piper).
fn extract_audio_samples(value: &ort::value::DynValue) -> Result<Vec<f32>, TtsError> {
    let view = value
        .try_extract_array::<f32>()
        .map_err(|e| TtsError::Backend(format!("try_extract_array<f32>: {e}")))?;
    Ok(view.iter().copied().collect())
}
