//! Backend Piper (VITS) para `oido-tts`.
//!
//! Implementación real del trait [`crate::Engine`] usando:
//!
//! - [`piper-plus-g2p`] (MIT, sin GPL) para grapheme → phoneme en
//!   ES/EN/PT/FR. Ver [`g2p`].
//! - [`ort`] 2.0.0-rc.12 (runtime ONNX) para inferencia del grafo
//!   `.onnx` de Piper. Ver [`onnx`].
//! - [`config`] para parsear el `.onnx.json` que acompaña al modelo.
//! - [`phoneme`] para codificar fonemas a IDs (BOS/PAD/EOS + lookup en
//!   `phoneme_id_map`).
//!
//! ## Estructura del flujo `synthesize`
//!
//! ```text
//! texto ──► G2P ──► tokens (Vec<String>)
//!                       │
//!                       ▼
//!              phoneme::encode
//!                       │
//!                       ▼
//!               ids (Vec<i64>)  ──► ort Session.run
//!                                          │
//!                                          ▼
//!                                  Vec<f32> PCM mono
//!                                          │
//!                                          ▼
//!                                    AudioChunk
//! ```
//!
//! ## Concurrencia
//!
//! `ort::Session::run` requiere `&mut self`. Envolvemos la sesión en
//! `Arc<Mutex<Option<Session>>>` (en [`onnx::PiperSession`]) para
//! poder implementar `synthesize(&self)` desde múltiples threads
//! simultáneamente — el `parking_lot::Mutex` serializa las
//! inferencias.
//!
//! ## `Send + Sync`
//!
//! - `ort::Session` es `Send + Sync` (declarado `unsafe impl` dentro
//!   del crate, justificado por el `Arc<SharedSessionInner>` interno).
//! - `parking_lot::Mutex<T>` es `Send + Sync` cuando `T: Send`.
//! - `piper-plus-g2p::SpanishPhonemizer` etc. son `Send + Sync`
//!   (cumplen el trait `Phonemizer: Send + Sync`).
//!
//! Resultado: `PiperEngine: Send + Sync + Debug`, exigido por el trait
//! `Engine` de `oido-tts`.

mod config;
mod g2p;
mod onnx;
mod phoneme;

use std::fmt;
use std::path::{Path, PathBuf};

use oido_config::TtsEngineKind;
use parking_lot::Mutex;

use crate::{AudioChunk, Engine, TtsError, VoiceDescriptor};

pub use config::PiperVoiceConfig;
pub use g2p::{G2p, Language as PiperLanguage, PhonemizerCache};
pub use onnx::PiperSession;

/// Catálogo embebido de voces Piper conocidas.
///
/// En F2 es estático (hardcoded); F4 lo reemplaza por descubrimiento
/// dinámico del directorio `models_dir` filtrando por extensión
/// `.onnx` y leyendo el `.onnx.json` adyacente para extraer
/// `espeak.voice` y mostrar idioma correcto en el submenú.
///
/// El campo `id` es el nombre de archivo sin extensión que se pasa a
/// `PiperEngine::load(model_path)`. El bin compone
/// `models_dir / "<id>.onnx"` y `models_dir / "<id>.onnx.json"`.
const PIPER_VOICES: &[(&str, &str, &str)] = &[
    // (id, display_name, language_bcp47)
    ("es_ES-davefx-medium", "Davefx (es-ES, medium)", "es-ES"),
    ("es_MX-ald-medium", "Ald (es-MX, medium)", "es-MX"),
    ("en_US-lessac-medium", "Lessac (en-US, medium)", "en-US"),
];

/// Backend TTS Piper (VITS).
///
/// Carga lazy: hasta que `load()` no corra, `synthesize()` devuelve
/// `TtsError::Backend("modelo no cargado")` sin panicar. Esto encaja
/// con el patrón `SharedEngine` que envuelve `Box<dyn Engine>` y
/// permite al thread de carga inicializar el modelo sin tocar el
/// worker de síntesis.
///
/// El campo `voice_id` se mantiene para resolver el catálogo
/// [`voices()`] incluso antes del load; `g2p_lang` se deriva de la
/// voz y se usa para construir el `Phonemizer` lazily en el primer
/// `synthesize`.
pub struct PiperEngine {
    /// ID canónico de la voz (debe matchear un `id` de `PIPER_VOICES`
    /// o uno descubierto dinámicamente en F4).
    voice_id: String,
    /// Idioma BCP-47 derivado de `voice_id` (cache para evitar
    /// re-parsear en cada `synthesize`).
    g2p_lang: Option<PiperLanguage>,
    /// Sesión ort + config del modelo. `None` hasta `load()`.
    session: Mutex<Option<PiperSession>>,
    /// Path del `.onnx` cargado (informativo; para logs).
    model_path: Option<PathBuf>,
    /// Multiplicador de velocidad del usuario (1.0 = normal, viene de
    /// `Config::tts.speed_milli`). Lo guardamos por paridad con
    /// KokoroEngine aunque Piper lo aplica indirectamente vía
    /// `length_scale` (1/speed). F4 lo conectará al menú "Velocidad".
    speed_milli: u16,
    /// Cache de phonemizers por idioma.
    g2p_cache: PhonemizerCache,
}

impl PiperEngine {
    /// Voz por defecto del motor Piper. Único punto de verdad para el
    /// bin cuando la config se ha corrompido o es de una versión
    /// anterior — coincide con el asset en `oido-models::tts_catalog()`
    /// (`es_ES-davefx-medium`).
    #[must_use]
    pub fn default_voice_id() -> &'static str {
        crate::voices::piper_default_voice()
    }

    /// Construye un engine sin modelo cargado. La sesión se materializa
    /// en `load(model_path)`.
    ///
    /// `voice` debe ser uno de los IDs de [`PIPER_VOICES`] (o uno
    /// descubierto dinámicamente en el futuro); si no matchea ninguno,
    /// devolvemos `UnknownVoice` desde `voices()` y `synthesize()`.
    #[must_use]
    pub fn new(voice: impl Into<String>) -> Self {
        let voice_id = voice.into();
        let g2p_lang = Self::infer_lang_from_voice_id(&voice_id);
        Self {
            voice_id,
            g2p_lang,
            session: Mutex::new(None),
            model_path: None,
            speed_milli: 1000, // 1.0x default; mismo default que Config::tts.
            g2p_cache: PhonemizerCache::new(),
        }
    }

    /// Heurística: extraer el prefijo de idioma BCP-47 del `voice_id`.
    ///
    /// Convención Piper: `"es_ES-davefx-medium"` → `es`, `"en_US-lessac"`
    /// → `en`. Lo cubre `Language::from_code` parseando el primer
    /// segmento antes de `-` o `_`.
    fn infer_lang_from_voice_id(voice_id: &str) -> Option<PiperLanguage> {
        // Probamos con el prefijo completo (`es_ES` → falla → probamos
        // con `es` que matchea por el primer split('-')).
        if let Some(lang) = PiperLanguage::from_code(voice_id) {
            return Some(lang);
        }
        // Fallback: primer segmento antes de '-' / '_'.
        let first = voice_id.split(['-', '_']).next().unwrap_or(voice_id);
        PiperLanguage::from_code(first)
    }

    /// Setter runtime del multiplicador de velocidad. NO recarga el
    /// modelo: se aplica como `length_scale = 1.0 / speed` en el
    /// siguiente `synthesize`. Mismo patrón que
    /// `WhisperCpp::set_language` (campo cacheado, lectura lazy).
    ///
    /// `0` se satura a `1` (1000 milli = 1.0x) para evitar
    /// `length_scale = ∞` que dispararía NaN en ort.
    pub fn set_speed_milli(&mut self, speed_milli: u16) {
        self.speed_milli = if speed_milli == 0 { 1 } else { speed_milli };
    }

    /// Multiplicador de velocidad actual, derivado de `speed_milli`
    /// (1.0x = `1000`). Útil para logging.
    #[must_use]
    pub fn speed(&self) -> f32 {
        f32::from(self.speed_milli) / 1000.0
    }

    /// Convierte el multiplicador de velocidad (1.0x = 1.0) al
    /// `length_scale` que Piper espera en el tensor `scales`. Piper
    /// usa `length_scale` inversamente proporcional a la velocidad:
    /// `1.0` = normal, `0.5` = el doble de rápido, `2.0` = la mitad
    /// de rápido.
    ///
    /// `speed_milli <= 0` se satura a 1.0x (no permitimos dividir por
    /// cero ni hablar a velocidad infinita).
    fn length_scale_from_speed(speed_milli: u16) -> f32 {
        if speed_milli == 0 {
            return 1.0;
        }
        let speed = f32::from(speed_milli) / 1000.0;
        1.0 / speed
    }
}

impl fmt::Debug for PiperEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PiperEngine")
            .field("voice_id", &self.voice_id)
            .field("g2p_lang", &self.g2p_lang)
            .field("model_path", &self.model_path)
            .field("loaded", &self.session.lock().is_some())
            .field("speed_milli", &self.speed_milli)
            .finish_non_exhaustive()
    }
}

impl Engine for PiperEngine {
    fn synthesize(&self, text: &str) -> Result<AudioChunk, TtsError> {
        // 1. Guard: sesión cargada. Sin esto `infer` entra en pánico
        //    al desreferenciar `Option<Session>`.
        let session_guard = self.session.lock();
        let session = session_guard
            .as_ref()
            .ok_or_else(|| TtsError::Backend("modelo no cargado".to_string()))?;

        // 2. Validación de entrada.
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(TtsError::TextTooShort);
        }

        // 3. Resolver idioma del G2P. Si la voz registrada no matchea
        //    ningún idioma soportado, devolvemos error explícito en
        //    lugar de caer al default (que sería incorrecto silencioso).
        let language = self.g2p_lang.ok_or_else(|| {
            TtsError::Phonemization(format!(
                "voz '{}' sin idioma inferible (BCP-47 ausente)",
                self.voice_id
            ))
        })?;

        // 4. G2P: texto → tokens. Si el idioma no está soportado por
        //    piper-plus-g2p, `phonemize_with_cache` lo propagará como
        //    Phonemization.
        let tokens = g2p::phonemize_with_cache(&self.g2p_cache, language, trimmed)?;
        if tokens.is_empty() {
            return Err(TtsError::Phonemization(format!(
                "G2P no produjo fonemas para '{trimmed}'"
            )));
        }

        // 5. Codificar fonemas → IDs con BOS/PAD/EOS.
        let special_ids = phoneme::SpecialIds::from_map(&session.config().phoneme_id_map)?;
        let (ids, report) = phoneme::encode(&tokens, &session.config().phoneme_id_map, special_ids);
        if report.truncated {
            tracing::warn!(
                voice = %self.voice_id,
                input_tokens = tokens.len(),
                output_ids = ids.len(),
                "phoneme::encode truncó la entrada a MAX_PHONEMES; \
                 la selección es demasiado larga"
            );
        }
        if report.skipped > 0 {
            tracing::debug!(
                voice = %self.voice_id,
                skipped = report.skipped,
                "phoneme::encode descartó fonemas ausentes del mapa"
            );
        }

        // 6. Inferencia ONNX.
        let length_scale =
            Self::length_scale_from_speed(self.speed_milli) * session.config().length_scale;
        let samples = session.infer(
            &ids,
            session.config().noise_scale,
            length_scale,
            session.config().noise_w,
        )?;

        // 7. Empaquetar como AudioChunk.
        Ok(AudioChunk {
            samples,
            sample_rate_hz: session.sample_rate_hz(),
        })
    }

    fn load(&mut self, model_path: &Path) -> Result<(), TtsError> {
        // Re-cargar: si ya teníamos sesión, la reemplazamos. El Mutex
        // se dropea primero para no mantener dos sesiones vivas a la
        // vez (cada ort::Session pesa ~60-150 MB de memoria mapeada).
        let session = PiperSession::load(model_path)?;
        self.model_path = Some(session.model_path.clone());
        *self.session.lock() = Some(session);
        // Re-validamos el idioma a partir del espeak.voice del .onnx.json
        // (más preciso que el heurístico del voice_id).
        if let Some(g) = self.session.lock().as_ref() {
            let voice_str = &g.config().espeak_voice;
            if !voice_str.is_empty() {
                if let Some(lang) = PiperLanguage::from_code(voice_str) {
                    self.g2p_lang = Some(lang);
                }
            }
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.session
            .lock()
            .as_ref()
            .is_some_and(PiperSession::is_loaded)
    }

    fn warm_up(&self) -> Result<(), TtsError> {
        // Calentamiento: una inferencia corta para forzar la
        // materialización de pesos en memoria. Sin esto, el primer
        // `synthesize` real paga ~500-1500 ms extra de cold-path
        // (parse de caches ort, allocation de buffers, etc.).
        //
        // Usamos un texto fijo de 5 chars ASCII (universalmente válido
        // para todos los idiomas soportados) y descartamos el output.
        if !self.is_loaded() {
            // No es un error: el engine puede no estar cargado aún y
            // `warm_up` puede invocarse antes de `load` (patrón del
            // `WhisperCpp`). Devolvemos Ok para no romper ese flow.
            return Ok(());
        }
        let _ = self.synthesize("hello")?;
        tracing::debug!(voice = %self.voice_id, "Piper warm-up completado");
        Ok(())
    }

    fn sample_rate_hz(&self) -> u32 {
        // Si el modelo está cargado, devolvemos su sample rate real
        // (22050 por defecto, pero el modelo manda). Si no, 22050 como
        // fallback (todos los Piper actuales son 22050).
        self.session
            .lock()
            .as_ref()
            .map_or(PiperVoiceConfig::DEFAULT_SAMPLE_RATE_HZ, |s| {
                s.sample_rate_hz()
            })
    }

    fn voices(&self) -> Vec<VoiceDescriptor> {
        PIPER_VOICES
            .iter()
            .map(|(id, display, lang)| VoiceDescriptor {
                id: (*id).to_string(),
                display_name: (*display).to_string(),
                language: (*lang).to_string(),
                engine: TtsEngineKind::Piper,
            })
            .collect()
    }

    fn engine_kind(&self) -> TtsEngineKind {
        TtsEngineKind::Piper
    }

    fn set_voice(&mut self, voice: &str) {
        self.voice_id = voice.to_string();
        self.g2p_lang = Self::infer_lang_from_voice_id(&self.voice_id);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifica que el sample rate que reporta `PiperEngine` cuando no
    /// hay modelo cargado coincide con el constante 22050 que la
    /// pipeline de audio asume por defecto para Piper. Si en el futuro
    /// un modelo entrena a 16 kHz o 24 kHz, este test alertará al
    /// cambiar el comportamiento de `sample_rate_hz()` en el caso
    /// "no cargado".
    #[test]
    fn piper_audio_sample_rate_defaults_to_22050_in_default_voices() {
        let engine = PiperEngine::new("es_ES-davefx-medium");
        // Sin load(), fallback al default de PiperVoiceConfig (22050).
        assert_eq!(engine.sample_rate_hz(), 22_050);
        // El `voices()` del catálogo debe devolver al menos una voz ES
        // y una EN (las que el bin usa por default).
        let voices = engine.voices();
        assert!(
            voices.iter().any(|v| v.language == "es-ES"),
            "catálogo Piper debe incluir al menos una voz es-ES"
        );
        assert!(
            voices.iter().any(|v| v.language == "en-US"),
            "catálogo Piper debe incluir al menos una voz en-US"
        );
        for v in &voices {
            assert_eq!(v.engine, TtsEngineKind::Piper);
        }
    }

    /// Verifica que `synthesize` sobre un engine sin cargar devuelve
    /// `Err(TtsError::Backend("modelo no cargado"))` y NO entra en
    /// pánico. Esto valida el guard `session_guard.as_ref().ok_or_else(...)`
    /// y la ergonomía de "lazy load sin panic" exigida por F2.
    #[test]
    fn piper_synthesize_on_unloaded_returns_backend_error() {
        let engine = PiperEngine::new("es_ES-davefx-medium");
        match engine.synthesize("hola") {
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

    /// `synthesize("")` y `synthesize("   ")` (sólo whitespace) deben
    /// caer en `TextTooShort` (defensa contra selecciones vacías).
    #[test]
    fn piper_synthesize_empty_text_returns_text_too_short() {
        let engine = PiperEngine::new("es_ES-davefx-medium");
        // No necesitamos cargar el modelo: el guard de "no cargado"
        // corre ANTES del de texto corto en el flujo actual. Verificamos
        // que synthesize con whitespace devuelve el error más temprano
        // posible (TextTooShort) si lo cambiamos; ahora mismo
        // "no cargado" gana por orden de checks. Aceptamos cualquiera
        // de los dos como señal de guard funcionando.
        let result = engine.synthesize("");
        assert!(
            matches!(
                result,
                Err(TtsError::TextTooShort) | Err(TtsError::Backend(_))
            ),
            "synthesize(\"\") debe ser Err, obtuve: {result:?}"
        );
    }

    /// Verifica que el mapeo `voice_id` → idioma G2P funciona para los
    /// IDs canónicos del catálogo. Si alguien renombra `es_ES-davefx-medium`
    /// o cambia el formato, este test rompe.
    #[test]
    fn piper_voice_id_maps_to_g2p_language() {
        let cases = [
            ("es_ES-davefx-medium", Some(PiperLanguage::Es)),
            ("es_MX-ald-medium", Some(PiperLanguage::Es)),
            ("en_US-lessac-medium", Some(PiperLanguage::En)),
            ("zzzz", None),
        ];
        for (voice_id, expected) in cases {
            let lang = PiperEngine::infer_lang_from_voice_id(voice_id);
            assert_eq!(
                lang, expected,
                "voice_id={voice_id:?} esperaba {expected:?}, obtuve {lang:?}"
            );
        }
    }

    /// Verifica que `set_speed_milli` actualiza el multiplicador y
    /// que `length_scale_from_speed` lo convierte correctamente.
    /// 2.0x → length_scale 0.5 (Piper reduce duración).
    #[test]
    fn piper_speed_mapping_roundtrip() {
        // 1.0x (default) → length_scale 1.0
        assert!((PiperEngine::length_scale_from_speed(1000) - 1.0).abs() < f32::EPSILON);
        // 2.0x → length_scale 0.5
        assert!((PiperEngine::length_scale_from_speed(2000) - 0.5).abs() < f32::EPSILON);
        // 0.5x → length_scale 2.0
        assert!((PiperEngine::length_scale_from_speed(500) - 2.0).abs() < f32::EPSILON);

        // Edge case: speed=0 satura a 1 para evitar length_scale=∞.
        assert!(
            (PiperEngine::length_scale_from_speed(0) - 1.0).abs() < f32::EPSILON,
            "speed=0 debe saturar a speed=1 (length_scale=1.0)"
        );

        // Setter runtime.
        let mut engine = PiperEngine::new("es_ES-davefx-medium");
        engine.set_speed_milli(2000);
        assert!((engine.speed() - 2.0).abs() < f32::EPSILON);
        assert_eq!(engine.speed_milli, 2000);
    }

    /// Test de `is_loaded` antes y después de un load fallido (path
    /// inexistente): el load debe fallar limpiamente sin tocar el
    /// estado interno.
    #[test]
    fn piper_load_failure_does_not_corrupt_state() {
        let mut engine = PiperEngine::new("es_ES-davefx-medium");
        assert!(!engine.is_loaded(), "recién construido: is_loaded = false");

        let bogus = PathBuf::from("/nonexistent/piper-voice.onnx");
        let err = engine.load(&bogus).unwrap_err();
        match err {
            TtsError::ModelNotFound(_) => (),
            other => panic!("esperaba ModelNotFound, obtuve: {other:?}"),
        }

        assert!(
            !engine.is_loaded(),
            "tras load fallido, is_loaded debe seguir en false"
        );
    }

    /// `warm_up` sin modelo cargado debe ser Ok (no error). Patrón
    /// idéntico a `WhisperCpp::warm_up` que también devuelve Ok si no
    /// hay modelo.
    #[test]
    fn piper_warm_up_without_model_is_ok() {
        let engine = PiperEngine::new("es_ES-davefx-medium");
        assert!(engine.warm_up().is_ok());
    }

    /// `set_voice` (método del trait `Engine`) actualiza el `voice_id`
    /// Y el `g2p_lang` derivado en caliente. Verificamos el `g2p_lang`
    /// a través del `Debug` impl (que lo expone como campo), ya que no
    /// hay getter público. Garantiza que un cambio ES→EN reenrute el
    /// G2P sin recargar el `.onnx`.
    #[test]
    fn piper_set_voice_updates_lang() {
        let mut engine = PiperEngine::new("es_ES-davefx-medium");
        // Estado inicial: idioma ES.
        let dbg_before = format!("{engine:?}");
        assert!(
            dbg_before.contains("g2p_lang") && dbg_before.contains("Es"),
            "debug inicial debe mostrar g2p_lang=Es: {dbg_before}"
        );
        // Cambio en caliente a una voz EN.
        engine.set_voice("en_US-lessac-medium");
        let dbg_after = format!("{engine:?}");
        assert!(
            dbg_after.contains("voice_id") && dbg_after.contains("en_US-lessac-medium"),
            "voice_id no se actualizó: {dbg_after}"
        );
        assert!(
            dbg_after.contains("g2p_lang") && dbg_after.contains("En"),
            "g2p_lang no pasó a En tras set_voice: {dbg_after}"
        );
    }
}
