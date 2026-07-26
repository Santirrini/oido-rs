//! Backend Kokoro-82M (Apache-2.0, English) para `oido-tts`.
//!
//! Implementación real del trait [`crate::Engine`] usando:
//!
//! - [`misaki-rs`] 0.3 (MIT, **sin** espeak-ng = sin GPL) para
//!   grapheme → fonema en inglés. Ver [`g2p`]. La elección de
//!   `misaki-rs` es por licencia: el binding a espeak-ng es GPL-3.0,
//!   incompatible con el binario MIT/Apache de oido. v1.0 sólo
//!   sintetiza **inglés** en Kokoro.
//! - [`ort`] 2.0.0-rc.12 (runtime ONNX, FFI aislado) para inferencia
//!   del grafo `kokoro-v1.0.onnx`. Ver [`onnx`].
//! - [`voices`] para cargar el `voices-v1.0.bin` (estilo por voz ×
//!   token length) y resolver la fila correcta en cada inferencia.
//!
//! ## Estructura del flujo `synthesize`
//!
//! ```text
//! texto (ASCII EN)
//!      │
//!      ▼  g2p::phonemize
//! phoneme_ids: Vec<i64>
//!      │
//!      ▼  onnx::KokoroSession::infer
//!      │     ├── pad [0, *ids, 0]  → tensor tokens [1, n+2] i64
//!      │     ├── voice_slice(voice, n) → tensor style [1, 256] f32
//!      │     ├── speed (1.0x) → tensor speed [1] f32
//!      │     └── ort Session.run()
//!      ▼
//! Vec<f32> PCM mono @ 24 kHz
//!      │
//!      ▼  AudioChunk { samples, sample_rate_hz: 24_000 }
//! ```
//!
//! ## Concurrencia
//!
//! `ort::Session::run` requiere `&mut self`. Envolvemos la sesión
//! (igual que Piper) en `Arc<Mutex<Option<Session>>>` para que
//! `synthesize(&self)` sea seguro desde múltiples workers. El
//! `parking_lot::Mutex` serializa las inferencias.
//!
//! ## `Send + Sync`
//!
//! - `ort::Session` es `Send + Sync` (declarado en el crate upstream).
//! - `parking_lot::Mutex<T>` es `Send + Sync` cuando `T: Send`.
//! - `misaki_rs::G2P` no es `Sync` (contiene un `Regex` interno), por
//!   lo que el cache de G2P usa `Mutex<Option<Arc<G2P>>>` en
//!   [`g2p`], garantizando que el `&self` de synthesize no toque el
//!   G2P más que para clonarlo.
//!
//! Resultado: `KokoroEngine: Send + Sync + Debug`, exigido por el
//! trait `Engine`.

mod g2p;
mod onnx;
mod voices;

use std::path::{Path, PathBuf};

use oido_config::TtsEngineKind;
use parking_lot::Mutex;

pub use onnx::{KokoroSession, SAMPLE_RATE_HZ};

use crate::{AudioChunk, Engine, TtsError, VoiceDescriptor};

/// Catálogo embebido de voces Kokoro conocidas (F0).
///
/// En F3 seguimos con el catálogo estático para que el submenú del
/// tray tenga contenido sin depender de la presencia del
/// `voices-v1.0.bin` (que el usuario puede no haber descargado
/// todavía). F4 lo reemplaza por descubrimiento dinámico del `.bin`:
/// `voice_ids()` del `VoiceBank` cargado se cruza con este catálogo
/// para añadir `display_name` y `language`.
const KOKORO_VOICES: &[(&str, &str, &str)] = &[
    ("af_heart", "Heart (en-US, female)", "en-US"),
    ("am_michael", "Michael (en-US, male)", "en-US"),
    ("bf_emma", "Emma (en-GB, female)", "en-GB"),
];

/// Backend TTS Kokoro-82M.
///
/// Carga lazy: hasta que `load()` no corra, `synthesize()` devuelve
/// `TtsError::Backend("modelo no cargado")` sin panicar. El
/// `KokoroSession` también requiere `load_voices()` con la ruta al
/// `voices-v1.0.bin` — sin él, `infer` devuelve
/// `Backend("banco de voces no cargado")`.
#[derive(Debug)]
pub struct KokoroEngine {
    /// ID canónico de la voz (debe matchear un `id` de
    /// `KOKORO_VOICES` o uno presente en el `VoiceBank`).
    voice_id: String,
    /// Sesión ort + banco de voces. `None` hasta `load()` +
    /// `load_voices()`. Encapsulado en un `KokoroSession` propio para
    /// que `infer` y `is_loaded` vivan en el mismo módulo.
    session: Mutex<Option<KokoroSession>>,
    /// Path del `.onnx` (informativo, para logs).
    model_path: Option<PathBuf>,
    /// Multiplicador de velocidad (1.0× = 1000 milli, viene de
    /// `Config::tts.speed_milli`). F4 lo conectará al menú
    /// "Velocidad" del tray.
    speed_milli: u16,
}

impl KokoroEngine {
    /// Construye un engine sin modelo cargado. La sesión ort se
    /// materializa en `load(model_path)` y el banco de voces en
    /// `load_voices(voices_path)`.
    #[must_use]
    pub fn new(voice: impl Into<String>) -> Self {
        Self {
            voice_id: voice.into(),
            session: Mutex::new(None),
            model_path: None,
            speed_milli: 1000, // 1.0× default; mismo default que Config::tts.
        }
    }

    /// Setter de la voz activa. NO recarga el modelo: se usa en el
    /// siguiente `synthesize`. Igual que
    /// `PiperEngine::set_speed_milli`.
    pub fn set_voice(&mut self, voice: impl Into<String>) {
        self.voice_id = voice.into();
    }

    /// Setter runtime del multiplicador de velocidad.
    ///
    /// `0` se satura a `1` (1000 milli = 1.0×) para evitar un tensor
    /// `speed=0` que Kokoro interpretaría como silencio (NaN/inf en
    /// el decoder ISTFTNet).
    pub fn set_speed_milli(&mut self, speed_milli: u16) {
        self.speed_milli = if speed_milli == 0 { 1 } else { speed_milli };
    }

    /// Multiplicador de velocidad actual, derivado de `speed_milli`
    /// (1.0× = 1000). Útil para logging.
    #[must_use]
    pub fn speed(&self) -> f32 {
        f32::from(self.speed_milli) / 1000.0
    }

    /// Carga el banco de voces desde `voices_path` (típicamente
    /// `voices-v1.0.bin`, ~25 MB). No es parte del trait `Engine` —
    /// lo invoca `load()` automáticamente cuando detecta el banco
    /// junto al `.onnx`. Devuelve `TtsError::Backend` si el archivo
    /// no se puede parsear.
    ///
    /// Si el `KokoroSession` aún no se ha materializado (no se llamó
    /// a `load` todavía), creamos un `KokoroSession::empty()` como
    /// carrier provisional. El `load` posterior rellenará el
    /// `inner` ort sin tocar el `voices` Mutex, que ya contiene el
    /// banco cargado.
    pub fn load_voices(&self, voices_path: &Path) -> Result<(), TtsError> {
        let mut session_guard = self.session.lock();
        if session_guard.is_none() {
            *session_guard = Some(KokoroSession::empty());
        }
        let session = session_guard
            .as_ref()
            .expect("KokoroEngine::load_voices: Option poblado arriba");
        session.load_voices(voices_path)?;
        tracing::info!(?voices_path, "banco de voces Kokoro cargado");
        Ok(())
    }

    /// Path del `.onnx` (informativo, `None` si no se cargó).
    #[must_use]
    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    /// ID de la voz activa.
    #[must_use]
    pub fn voice_id(&self) -> &str {
        &self.voice_id
    }
}

impl Default for KokoroEngine {
    fn default() -> Self {
        Self::new("af_heart")
    }
}

impl Engine for KokoroEngine {
    fn synthesize(&self, text: &str) -> Result<AudioChunk, TtsError> {
        // 1. Clonamos el `KokoroSession` completo (es barato: sus
        //    campos son `Arc<Mutex<...>>` + `PathBuf`) para soltar el
        //    `Mutex<Option<KokoroSession>>` exterior antes de inferir.
        //    Sin esto, mantener el lock exterior durante `infer` (que
        //    también bloquea sobre el `Mutex<Option<Session>>`
        //    interior) sería self-deadlock puro si re-entrase por
        //    cualquier motivo.
        let session = {
            let session_guard = self.session.lock();
            let session = session_guard.as_ref().ok_or_else(|| {
                TtsError::Backend("modelo no cargado".to_string())
            })?;
            if !session.is_loaded() {
                return Err(TtsError::Backend(
                    "modelo o banco de voces no cargado".into(),
                ));
            }
            session.clone()
        };

        // 2. G2P: texto → IDs. Rechaza vacío.
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(TtsError::TextTooShort);
        }
        let phoneme_ids = g2p::phonemize(trimmed)?;
        if phoneme_ids.is_empty() {
            return Err(TtsError::Phonemization(format!(
                "G2P no produjo fonemas para '{trimmed}'"
            )));
        }

        // 3. Inferencia ONNX.
        let speed = f32::from(self.speed_milli) / 1000.0;
        let samples = session.infer(&phoneme_ids, &self.voice_id, speed)?;

        // 4. Empaquetar como AudioChunk.
        Ok(AudioChunk {
            samples,
            sample_rate_hz: SAMPLE_RATE_HZ,
        })
    }

    fn load(&mut self, model_path: &Path) -> Result<(), TtsError> {
        if !model_path.exists() {
            return Err(TtsError::ModelNotFound(model_path.to_path_buf()));
        }
        // Construir (o reusar) el `KokoroSession`. Si ya había uno
        // con el banco de voces cargado, lo preservamos.
        let mut session_guard = self.session.lock();
        if session_guard.is_none() {
            *session_guard = Some(KokoroSession::empty());
        }
        let session = session_guard
            .as_mut()
            .expect("KokoroEngine::load: Option poblado arriba");
        session.load(model_path)?;
        self.model_path = Some(model_path.to_path_buf());

        // Carga automática del banco de voces si está junto al `.onnx`.
        // Convención: el bin descarga `kokoro-82m-v1.0.onnx` y
        // `voices-v1.0.bin` al mismo directorio (`models_dir`). Sin el
        // banco, `infer` devolvería `Backend("banco de voces no cargado")`.
        // Buscamos `voices-v1.0.bin` en el directorio padre del modelo.
        if let Some(parent) = model_path.parent() {
            let voices_path = parent.join("voices-v1.0.bin");
            if voices_path.is_file() {
                if let Err(e) = session.load_voices(&voices_path) {
                    tracing::warn!(
                        ?e,
                        ?voices_path,
                        "banco de voces Kokoro presente pero no se pudo cargar"
                    );
                }
            } else {
                tracing::warn!(
                    ?voices_path,
                    "TTS Kokoro: voices-v1.0.bin no encontrado junto al modelo; \
                     la síntesis fallará hasta descargarlo con \
                     `oido --tts-download kokoro`"
                );
            }
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.session
            .lock()
            .as_ref()
            .is_some_and(KokoroSession::is_loaded)
    }

    fn warm_up(&self) -> Result<(), TtsError> {
        // Calentamiento: una inferencia corta para forzar la
        // materialización de pesos ort y la compilación lazy de
        // kernels CUDA/Metal (cuando aplica). Sin esto, el primer
        // `synthesize` real paga ~500-1500 ms de cold-path.
        //
        // Si el engine aún no está cargado, NO es error: el patrón
        // del bin puede llamar a `warm_up` antes de `load` (igual
        // que en `WhisperCpp::warm_up`).
        if !self.is_loaded() {
            return Ok(());
        }
        let _ = self.synthesize("hello")?;
        tracing::debug!(voice = %self.voice_id, "Kokoro warm-up completado");
        Ok(())
    }

    fn sample_rate_hz(&self) -> u32 {
        // Kokoro-82M emite fijo a 24 kHz (verificado contra
        // `kokoro_onnx/config.py`).
        SAMPLE_RATE_HZ
    }

    fn voices(&self) -> Vec<VoiceDescriptor> {
        KOKORO_VOICES
            .iter()
            .map(|(id, display, lang)| VoiceDescriptor {
                id: (*id).to_string(),
                display_name: (*display).to_string(),
                language: (*lang).to_string(),
                engine: TtsEngineKind::Kokoro,
            })
            .collect()
    }

    fn engine_kind(&self) -> TtsEngineKind {
        TtsEngineKind::Kokoro
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `KokoroEngine::sample_rate_hz()` debe ser siempre 24 000 (Kokoro
    /// emite fijo a 24 kHz). Si en el futuro el modelo cambia, este
    /// test alertará para actualizar el `PlaybackSink` y el resampling.
    #[test]
    fn kokoro_sample_rate_is_24000() {
        let engine = KokoroEngine::new("af_heart");
        assert_eq!(engine.sample_rate_hz(), 24_000);
        assert_eq!(SAMPLE_RATE_HZ, 24_000);
    }

    /// `Engine::engine_kind()` debe devolver `TtsEngineKind::Kokoro`
    /// para que el tray pueda etiquetar "Leyendo con Kokoro — af_heart".
    #[test]
    fn kokoro_engine_kind_is_kokoro() {
        let engine = KokoroEngine::new("af_heart");
        assert_eq!(engine.engine_kind(), TtsEngineKind::Kokoro);
    }

    /// El catálogo embebido de voces debe incluir al menos las 3
    /// voces canónicas que el submenú muestra por default.
    #[test]
    fn kokoro_voices_catalog_has_three_entries() {
        let engine = KokoroEngine::new("af_heart");
        let voices = engine.voices();
        assert_eq!(voices.len(), 3, "catálogo F0 debe tener 3 voces");
        for v in &voices {
            assert_eq!(v.engine, TtsEngineKind::Kokoro);
            assert!(!v.id.is_empty());
            assert!(!v.display_name.is_empty());
            assert!(!v.language.is_empty());
        }
        // El ID pasado a `new` debe estar en el catálogo (sanity
        // check: si F4 cambia el catálogo, esto rompe a propósito).
        assert!(
            voices.iter().any(|v| v.id == "af_heart"),
            "af_heart debe estar en el catálogo"
        );
    }

    /// `synthesize` sobre un engine sin modelo cargado debe devolver
    /// `Err(TtsError::Backend("modelo no cargado"))` y NO panicar.
    /// Garantiza que el guard lazy-load funciona (F3 entrega un
    /// engine que se comporta como los stubs F0 pero ya tiene la
    /// estructura interna completa).
    #[test]
    fn kokoro_synthesize_on_unloaded_returns_backend_error() {
        let engine = KokoroEngine::new("af_heart");
        match engine.synthesize("hello world") {
            Err(TtsError::Backend(msg)) => {
                assert!(
                    msg.contains("no cargado"),
                    "mensaje debería mencionar 'no cargado', obtuve: {msg}"
                );
            }
            Err(other) => panic!(
                "esperaba TtsError::Backend(\"modelo no cargado\"), obtuve: {other:?}"
            ),
            Ok(_) => panic!("synthesize sin modelo cargado NO debe devolver Ok"),
        }
    }

    /// `synthesize("")` o whitespace devuelve `TextTooShort` (defensa
    /// contra selecciones vacías). Como el guard de "no cargado" corre
    /// antes, aceptamos ambos errores como señal de guard funcionando.
    #[test]
    fn kokoro_synthesize_empty_text_returns_guard_error() {
        let engine = KokoroEngine::new("af_heart");
        let result = engine.synthesize("");
        assert!(
            matches!(
                result,
                Err(TtsError::TextTooShort) | Err(TtsError::Backend(_))
            ),
            "synthesize(\"\") debe ser Err, obtuve: {result:?}"
        );
    }

    /// `set_speed_milli(0)` satura a 1 (evita speed=0 → NaN en
    /// Kokoro). El speed efectivo leído vía `speed()` debe ser 0.001
    /// (= 1/1000), NUNCA 0.0.
    #[test]
    fn kokoro_set_speed_milli_zero_saturates_to_one() {
        let mut engine = KokoroEngine::new("af_heart");
        engine.set_speed_milli(0);
        assert_eq!(engine.speed_milli, 1);
        assert!((engine.speed() - 0.001).abs() < f32::EPSILON);
    }

    /// `set_speed_milli(1500)` produce 1.5× (rango Kokoro válido).
    #[test]
    fn kokoro_set_speed_milli_scales_correctly() {
        let mut engine = KokoroEngine::new("af_heart");
        engine.set_speed_milli(1500);
        assert!((engine.speed() - 1.5).abs() < f32::EPSILON);
    }

    /// `is_loaded()` antes de `load` debe ser `false`. Sin modelo no
    /// se puede sintetizar.
    #[test]
    fn kokoro_is_loaded_false_before_load() {
        let engine = KokoroEngine::new("af_heart");
        assert!(!engine.is_loaded());
    }
}
