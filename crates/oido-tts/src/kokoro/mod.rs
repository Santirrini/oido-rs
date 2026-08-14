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
    // === American English (en-US, female) ===
    ("af_heart", "Heart (en-US, female)", "en-US"),
    ("af_alloy", "Alloy (en-US, female)", "en-US"),
    ("af_aoede", "Aoede (en-US, female)", "en-US"),
    ("af_bella", "Bella (en-US, female)", "en-US"),
    ("af_jessica", "Jessica (en-US, female)", "en-US"),
    ("af_kore", "Kore (en-US, female)", "en-US"),
    ("af_nicole", "Nicole (en-US, female)", "en-US"),
    ("af_nova", "Nova (en-US, female)", "en-US"),
    ("af_river", "River (en-US, female)", "en-US"),
    ("af_sarah", "Sarah (en-US, female)", "en-US"),
    ("af_sky", "Sky (en-US, female)", "en-US"),
    // === American English (en-US, male) ===
    ("am_adam", "Adam (en-US, male)", "en-US"),
    ("am_echo", "Echo (en-US, male)", "en-US"),
    ("am_eric", "Eric (en-US, male)", "en-US"),
    ("am_fenrir", "Fenrir (en-US, male)", "en-US"),
    ("am_liam", "Liam (en-US, male)", "en-US"),
    ("am_michael", "Michael (en-US, male)", "en-US"),
    ("am_onyx", "Onyx (en-US, male)", "en-US"),
    ("am_puck", "Puck (en-US, male)", "en-US"),
    ("am_santa", "Santa (en-US, male)", "en-US"),
    // === British English (en-GB, female) ===
    ("bf_alice", "Alice (en-GB, female)", "en-GB"),
    ("bf_emma", "Emma (en-GB, female)", "en-GB"),
    ("bf_isabella", "Isabella (en-GB, female)", "en-GB"),
    ("bf_lily", "Lily (en-GB, female)", "en-GB"),
    // === British English (en-GB, male) ===
    ("bm_daniel", "Daniel (en-GB, male)", "en-GB"),
    ("bm_fable", "Fable (en-GB, male)", "en-GB"),
    ("bm_george", "George (en-GB, male)", "en-GB"),
    ("bm_lewis", "Lewis (en-GB, male)", "en-GB"),
    // === Spanish (es, female & male) ===
    ("ef_dora", "Dora (es, female)", "es"),
    ("em_alex", "Alex (es, male)", "es"),
    ("em_santa", "Santa (es, male)", "es"),
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
            let session = session_guard
                .as_ref()
                .ok_or_else(|| TtsError::Backend("modelo no cargado".to_string()))?;
            if !session.is_loaded() {
                return Err(TtsError::Backend(
                    "modelo o banco de voces no cargado".into(),
                ));
            }
            session.clone()
        };

        // 2. G2P: texto → IDs. El G2P se enruta por el idioma de la voz
        //    (prefijo `ef_*`/`em_*` = español, resto = inglés), no por
        //    el contenido del texto: el español ASCII sin tildes
        //    ("Simulacion Medica") pasaría por el G2P inglés y el modelo
        //    deletrearía. Ver `g2p::KokoroLang::from_voice`.
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(TtsError::TextTooShort);
        }
        let phoneme_ids = g2p::phonemize(trimmed, &self.voice_id)?;
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

    fn set_voice(&mut self, voice: &str) {
        self.voice_id = voice.to_string();
    }
}

// `KOKORO_VOICES` (privado) es la fuente de verdad dentro de
// `oido-tts`. La **única** lista canónica consumida por el resto del
// workspace vive en `oido_models::tts_models::KOKORO_VOICES` (espejo
// del upstream `onnx-community/Kokoro-82M-v1.0-ONNX` README).
// `Engine::voices()` (línea 343) y los tests de este módulo usan la
// copia local; `oido-tray::TtsSection::build_voices_submenu` consume
// la de `oido-models`.

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

    /// El catálogo embebido de voces debe incluir las 28 voces
    /// canónicas del `voices-v1.0.bin` de Kokoro v1.0. Si cambia
    /// el `.bin` upstream, este test alertará para actualizar
    /// `KOKORO_VOICES` (en este módulo) y
    /// `oido_models::tts_models::KOKORO_VOICES` (la copia pública).
    #[test]
    fn kokoro_voices_catalog_has_all_entries() {
        let engine = KokoroEngine::new("af_heart");
        let voices = engine.voices();
        assert_eq!(voices.len(), 31, "catálogo Kokoro debe tener 31 voces");
        for v in &voices {
            assert_eq!(v.engine, TtsEngineKind::Kokoro);
            assert!(!v.id.is_empty());
            assert!(!v.display_name.is_empty());
            assert!(!v.language.is_empty());
        }
        // Sanity checks: voces que NO pueden faltar.
        assert!(
            voices.iter().any(|v| v.id == "af_heart"),
            "af_heart debe estar en el catálogo"
        );
        assert!(
            voices.iter().any(|v| v.id == "am_michael"),
            "am_michael debe estar en el catálogo"
        );
        assert!(
            voices.iter().any(|v| v.id == "bf_emma"),
            "bf_emma debe estar en el catálogo"
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
            Err(other) => {
                panic!("esperaba TtsError::Backend(\"modelo no cargado\"), obtuve: {other:?}")
            }
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

    /// `set_voice` (método del trait `Engine`) actualiza el `voice_id`
    /// en caliente sin recargar el modelo. Garantiza que el handler
    /// `SyncTtsRuntime` del control loop pueda cambiar de voz Kokoro
    /// sin pagar la recarga del `.onnx` (326 MB).
    #[test]
    fn kokoro_set_voice_updates_voice_id() {
        let mut engine = KokoroEngine::new("af_heart");
        assert_eq!(engine.voice_id(), "af_heart");
        // Cambio en caliente vía el trait (no el constructor).
        engine.set_voice("am_michael");
        assert_eq!(engine.voice_id(), "am_michael");
        // Y un segundo cambio para confirmar que no es one-shot.
        engine.set_voice("bf_emma");
        assert_eq!(engine.voice_id(), "bf_emma");
    }

    /// `is_loaded()` antes de `load` debe ser `false`. Sin modelo no
    /// se puede sintetizar.
    #[test]
    fn kokoro_is_loaded_false_before_load() {
        let engine = KokoroEngine::new("af_heart");
        assert!(!engine.is_loaded());
    }
}
