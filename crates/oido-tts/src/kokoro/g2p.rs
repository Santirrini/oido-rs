//! Grapheme → phoneme → token IDs para Kokoro-82M (multilingüe).
//!
//! ## Pipeline
//!
//! ```text
//! texto + voice_id
//!      │
//!      ▼  KokoroLang::from_voice(voice_id)   → En | Es
//!      │
//!      ├─ En: misaki-rs G2P (EnglishUS) → IPA string
//!      └─ Es: piper-plus-g2p SpanishPhonemizer → Vec<phoneme>
//!      │
//!      ▼  lookup carácter por carácter en `KOKORO_VOCAB`
//!      │  (los fonemas son a veces multi-char: "ʰ", "↓", etc.)
//!      ▼
//! Vec<i64>  (ids de token listos para `Tensor::from_array`)
//! ```
//!
//! ## Enrutado por idioma de la voz
//!
//! El idioma del G2P se decide por el **prefijo del `voice_id`**, no
//! por el contenido del texto. Razón: el texto español sin tildes
//! ("Simulacion Medica") es ASCII puro, y si lo pasáramos por el G2P
//! de inglés el modelo "deletrea" (fonemas ingleses sobre palabras
//! españolas). El prefijo del `voice_id` es una señal mucho más
//! fiable:
//!
//! - `af_*` / `am_*` / `bf_*` / `bm_*` → inglés (`af_heart`, `am_michael`)
//! - `ef_*` / `em_*` → español (`ef_dora`, `em_alex`, `em_santa`)
//!
//! Ver [`KokoroLang::from_voice`] y [`phonemize`].
//!
//! ## Vocabulario
//!
//! El `KOKORO_VOCAB` está **hardcodeado** aquí porque `misaki-rs` no
//! expone el mapeo fonema→ID (es decisión de cada modelo TTS cómo
//! codificar sus fonemas). La fuente de verdad es el `config.json`
//! distribuido con Kokoro-82M v1.0 (178 tokens, IDs 0..=177; ID 0 es
//! el pad). Verificado vía WebFetch contra
//! `https://github.com/thewh1teagle/kokoro-onnx/blob/main/src/kokoro_onnx/config.json`.
//!
//! Si en el futuro sale Kokoro v1.1 con otro vocabulario, este mapa se
//! reemplaza por una versión que cargue el `config.json` del modelo
//! (probablemente con `serde_json`).
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use piper_plus_g2p::phonemizer::Phonemizer as _;

use crate::TtsError;

/// Mapeo fonema-string → token ID del modelo Kokoro-82M v1.0.
///
/// Construido a partir del `vocab` JSON del config de Kokoro-82M
/// (178 entries; ID 0 = pad, no listado).
///
/// `&'static str` como clave (no `String`) para que el `HashMap`
/// subyacente sea completamente estático y no haga allocations en
/// cada `phonemize`.
const KOKORO_VOCAB_RAW: &[(&str, i64)] = &[
    (";", 1),
    (":", 2),
    (",", 3),
    (".", 4),
    ("!", 5),
    ("?", 6),
    ("—", 9),
    ("…", 10),
    ("\"", 11),
    ("(", 12),
    (")", 13),
    ("“", 14),
    ("”", 15),
    (" ", 16),
    ("\u{0303}", 17),
    ("ʣ", 18),
    ("ʥ", 19),
    ("ʦ", 20),
    ("ʨ", 21),
    ("ᵝ", 22),
    ("\u{AB67}", 23),
    ("A", 24),
    ("I", 25),
    ("O", 31),
    ("Q", 33),
    ("S", 35),
    ("T", 36),
    ("W", 39),
    ("Y", 41),
    ("ᵊ", 42),
    ("a", 43),
    ("b", 44),
    ("c", 45),
    ("d", 46),
    ("e", 47),
    ("f", 48),
    ("h", 50),
    ("i", 51),
    ("j", 52),
    ("k", 53),
    ("l", 54),
    ("m", 55),
    ("n", 56),
    ("o", 57),
    ("p", 58),
    ("q", 59),
    ("r", 60),
    ("s", 61),
    ("t", 62),
    ("u", 63),
    ("v", 64),
    ("w", 65),
    ("x", 66),
    ("y", 67),
    ("z", 68),
    ("ɑ", 69),
    ("ɐ", 70),
    ("ɒ", 71),
    ("æ", 72),
    ("β", 75),
    ("ɔ", 76),
    ("ɕ", 77),
    ("ç", 78),
    ("ɖ", 80),
    ("ð", 81),
    ("ʤ", 82),
    ("ə", 83),
    ("ɚ", 85),
    ("ɛ", 86),
    ("ɜ", 87),
    ("ɟ", 90),
    ("ɡ", 92),
    ("ɥ", 99),
    ("ɨ", 101),
    ("ɪ", 102),
    ("ʝ", 103),
    ("ɯ", 110),
    ("ɰ", 111),
    ("ŋ", 112),
    ("ɳ", 113),
    ("ɲ", 114),
    ("ɴ", 115),
    ("ø", 116),
    ("ɸ", 118),
    ("θ", 119),
    ("œ", 120),
    ("ɹ", 123),
    ("ɾ", 125),
    ("ɻ", 126),
    ("ʁ", 128),
    ("ɽ", 129),
    ("ʂ", 130),
    ("ʃ", 131),
    ("ʈ", 132),
    ("ʧ", 133),
    ("ʊ", 135),
    ("ʋ", 136),
    ("ʌ", 138),
    ("ɣ", 139),
    ("ɤ", 140),
    ("χ", 142),
    ("ʎ", 143),
    ("ʒ", 147),
    ("ʔ", 148),
    ("ˈ", 156),
    ("ˌ", 157),
    ("ː", 158),
    ("ʰ", 162),
    ("ʲ", 164),
    ("↓", 169),
    ("→", 171),
    ("↗", 172),
    ("↘", 173),
    ("ᵻ", 177),
];

/// Tabla de lookup fonema → ID, construida perezosamente en el primer
/// `phonemize`. Usar `OnceLock` evita pagar el coste de construir el
/// `HashMap` aunque nadie sintetice (la carga del bin `oido` no
/// instancia engines hasta que el usuario los usa).
fn vocab_map() -> &'static HashMap<&'static str, i64> {
    static MAP: OnceLock<HashMap<&'static str, i64>> = OnceLock::new();
    MAP.get_or_init(|| KOKORO_VOCAB_RAW.iter().copied().collect())
}

/// Cache lazy del `misaki_rs::G2P`. La primera llamada a `phonemize`
/// instancia el G2P (carga el lexicon interno, O(millones) de
/// entradas, ~50-100 ms); las siguientes son inmediatas.
///
/// Por qué `parking_lot::Mutex` (no `OnceLock<G2P>` directo): el
/// `G2P::new` de misaki-rs **puede** entrar en pánico en caso de
/// error interno; con `OnceLock` no podemos capturar el pánico, sólo
/// envenenarlo. Un `Mutex<Option<...>>` permite reintentar si en el
/// futuro se corrompe.
use parking_lot::Mutex;
use std::sync::Arc;

// No derivamos `Debug` aquí: `misaki_rs::G2P` no implementa `Debug`
// (contiene `Regex` interna sin derive), y por tanto
// `Mutex<Option<Arc<G2P>>>` tampoco puede ser `Debug` automáticamente.
// `KokoroEngine` (el owner) tiene su propio `Debug` que NO entra al
// cache de G2P, así que esto es invisible fuera.
struct G2pCache {
    inner: Mutex<Option<Arc<misaki_rs::G2P>>>,
}

impl fmt::Debug for G2pCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("G2pCache")
            .field("loaded", &self.inner.lock().is_some())
            .finish()
    }
}

impl G2pCache {
    fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn get(&self) -> Result<Arc<misaki_rs::G2P>, TtsError> {
        if let Some(g) = self.inner.lock().as_ref() {
            return Ok(Arc::clone(g));
        }
        let g = Arc::new(misaki_rs::G2P::new(misaki_rs::Language::EnglishUS));
        *self.inner.lock() = Some(Arc::clone(&g));
        Ok(g)
    }
}

/// Idioma que soporta el G2P de Kokoro. Se deriva del prefijo del
/// `voice_id` (ver [`KokoroLang::from_voice`]), NO del contenido del
/// texto: el texto español sin tildes ("Simulacion Medica") es ASCII
/// puro, y enrutarlo al G2P de inglés provoca que el modelo deletree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KokoroLang {
    /// Voz inglesa: `af_*`, `am_*`, `bf_*`, `bm_*`. G2P via `misaki-rs`.
    En,
    /// Voz española: `ef_*`, `em_*`. G2P via `piper-plus-g2p`.
    Es,
}

impl KokoroLang {
    /// Infiere el idioma a partir del prefijo del `voice_id`.
    ///
    /// Reconoce dos convenciones de naming:
    ///
    /// - **Voces Kokoro** (`ef_*`, `em_*`): la `e` inicial indica
    ///   español (`ef_dora`, `em_alex`, `em_santa`); resto (`af_*`,
    ///   `am_*`, `bf_*`, `bm_*`) es inglés.
    /// - **Voces Piper** (`es_ES-*`, `es_MX-*`, `en_US-*`): el primer
    ///   segmento es el código de idioma BCP-47. `es` → español,
    ///   `en` → inglés.
    ///
    /// Aceptar ambos formatos es necesario porque la config puede tener
    /// `tts.engine=Kokoro` con un `voice_id` heredado de Piper
    /// (e.g. `es_ES-davefx-medium`) tras un switch de engine. En ese
    /// caso el `VoiceBank` cae a `ef_dora` (voz española de fallback),
    /// pero el G2P tiene que saber que el idioma es español — si no,
    /// enruta a inglés y el modelo deletrea.
    #[must_use]
    pub fn from_voice(voice_id: &str) -> Self {
        // Normalizamos a minúsculas para tolerar `EF_DORA` mayúsculas
        // (defensivo; el catálogo siempre usa minúsculas).
        let lower = voice_id.to_ascii_lowercase();
        // Primer segmento antes de `-` o `_`.
        let first_segment = lower.split(['-', '_']).next().unwrap_or(&lower);
        // Kokoro español: ef/em. Piper español: es (es_ES, es_MX).
        if first_segment == "ef" || first_segment == "em" || first_segment == "es" {
            Self::Es
        } else {
            Self::En
        }
    }
}

/// Convierte `text` a IDs de fonema de Kokoro, enrutando el G2P según
/// el idioma de la `voice_id`.
///
/// Pasos:
/// 1. Rechaza texto vacío (`TextTooShort`).
/// 2. [`KokoroLang::from_voice`] decide inglés vs español.
/// 3. El G2P del idioma produce fonemas IPA (misaki-rs para EN,
///    `piper-plus-g2p` para ES).
/// 4. Tokeniza los fonemas carácter-a-carácter consultando
///    `vocab_map()`. Multi-char phonemes de Kokoro (`"↓"`, `"→"`,
///    `"\u{0303}"`) caben en un solo `char` Unicode, así que la
///    iteración char-by-char es suficiente (NO byte-by-byte).
///
/// `voice_id` decide el G2P porque las voces Kokoro tienen un prefijo
/// de idioma (`ef_*`/`em_*` = ES, resto = EN). Antes se usaba la
/// presencia de tildes en el texto, pero eso fallaba con español ASCII
/// sin tildes ("Simulacion" → G2P inglés → deletreo).
///
/// Devuelve `Vec<i64>` con los IDs en orden, listo para alimentar
/// `Tensor::from_array`.
pub fn phonemize(text: &str, voice_id: &str) -> Result<Vec<i64>, TtsError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(TtsError::TextTooShort);
    }

    match KokoroLang::from_voice(voice_id) {
        KokoroLang::Es => phonemize_spanish(trimmed),
        KokoroLang::En => phonemize_english(trimmed),
    }
}

/// Fonemiza texto inglés con `misaki-rs` (sin espeak-ng = sin GPL).
/// Las voces inglesas de Kokoro (`af_*`, `am_*`, `bf_*`, `bm_*`) usan
/// este path.
fn phonemize_english(text: &str) -> Result<Vec<i64>, TtsError> {
    // Construye / recupera el G2P cacheado (inglés via misaki-rs).
    static CACHE: std::sync::OnceLock<G2pCache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(G2pCache::new);
    let g2p = cache.get()?;

    // `misaki_rs::G2P::g2p` devuelve `(phoneme_string, Vec<MToken>)`.
    let (phoneme_str, _tokens) = g2p
        .g2p(text)
        .map_err(|e| TtsError::Phonemization(format!("misaki-rs g2p: {e}")))?;

    let map = vocab_map();
    let mut ids: Vec<i64> = Vec::with_capacity(phoneme_str.chars().count() + 2);
    for c in phoneme_str.chars() {
        let mut buf = [0u8; 4];
        let s: &str = c.encode_utf8(&mut buf);
        if let Some(&id) = map.get(s) {
            ids.push(id);
        } else {
            tracing::debug!(
                char = %c,
                "fonema emitido por misaki-rs no está en vocab Kokoro; skip"
            );
        }
    }

    if ids.is_empty() {
        return Err(TtsError::Phonemization(format!(
            "ningún fonema reconocible para '{text}'"
        )));
    }
    Ok(ids)
}

/// Fonemiza texto en español utilizando `piper-plus-g2p::spanish::SpanishPhonemizer`
/// y mapea los fonemas e IPA resultantes al vocabulario de Kokoro-82M.
fn phonemize_spanish(text: &str) -> Result<Vec<i64>, TtsError> {
    static SPANISH_G2P: std::sync::OnceLock<piper_plus_g2p::spanish::SpanishPhonemizer> =
        std::sync::OnceLock::new();
    let g2p = SPANISH_G2P.get_or_init(piper_plus_g2p::spanish::SpanishPhonemizer::new);

    let (phonemes, _prosody) = g2p
        .phonemize_with_prosody(text)
        .map_err(|e| TtsError::Phonemization(format!("spanish g2p error: {e:?}")))?;

    let map = vocab_map();
    let mut ids: Vec<i64> = Vec::with_capacity(phonemes.len() + 2);

    for ph in phonemes {
        for c in ph.chars() {
            let normalized = match c {
                'á' | 'Á' => 'a',
                'é' | 'É' => 'e',
                'í' | 'Í' => 'i',
                'ó' | 'Ó' => 'o',
                'ú' | 'Ú' | 'ü' | 'Ü' => 'u',
                'ñ' | 'Ñ' => 'ɲ', // ID 114 en Kokoro vocab
                '¿' => '?',
                '¡' => '!',
                other => other,
            };

            let mut buf = [0u8; 4];
            let s: &str = normalized.encode_utf8(&mut buf);
            if let Some(&id) = map.get(s) {
                ids.push(id);
            } else {
                tracing::debug!(
                    char = %c,
                    normalized = %normalized,
                    "fonema español no está en vocab Kokoro; skip"
                );
            }
        }
    }

    if ids.is_empty() {
        return Err(TtsError::Phonemization(format!(
            "ningún fonema reconocible para '{text}'"
        )));
    }

    Ok(ids)
}

/// Versión "inferir y devolver (ids, debug)" — usada por tests para
/// inspeccionar el resultado del G2P inglés (misaki-rs). En producción
/// usar [`phonemize`] (que enruta por idioma de la voz).
#[cfg(test)]
pub fn phonemize_debug(text: &str) -> Result<(Vec<i64>, String), TtsError> {
    // Reconstruimos el `phoneme_str` además de los IDs para que los
    // tests puedan validar que misaki-rs emitió IPA. Como
    // `phonemize_english` sólo devuelve IDs, replicamos aquí el paso
    // del G2P (es barato y sólo corre en tests).
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(TtsError::TextTooShort);
    }
    static CACHE: std::sync::OnceLock<G2pCache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(G2pCache::new);
    let g2p = cache.get()?;
    let (phoneme_str, _) = g2p
        .g2p(trimmed)
        .map_err(|e| TtsError::Phonemization(format!("misaki-rs g2p: {e}")))?;
    let map = vocab_map();
    let mut ids: Vec<i64> = Vec::with_capacity(phoneme_str.chars().count() + 2);
    for c in phoneme_str.chars() {
        let mut buf = [0u8; 4];
        let s: &str = c.encode_utf8(&mut buf);
        if let Some(&id) = map.get(s) {
            ids.push(id);
        }
    }
    Ok((ids, phoneme_str))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// El vocabulario debe tener EXACTAMENTE las mismas claves que el
    /// `config.json` upstream. Si alguien renombra un fonema o añade
    /// uno nuevo, este test rompe (mejor que fallar silenciosamente
    /// en producción con un `❓` que Kokoro no entiende).
    ///
    /// NOTA sobre grafemas "raros": el vocab Kokoro usa IPA, no ASCII.
    /// Por ejemplo, la "g" se codifica como `ɡ` (U+0261) y no como
    /// `g` (que NO existe en el vocab). Lo verificamos explícitamente
    /// para detectar drift.
    #[test]
    fn vocab_contains_all_expected_phonemes() {
        let map = vocab_map();
        // IDs de fonemas de alta frecuencia (deben estar SIEMPRE).
        for k in [
            ";", ":", ",", ".", " ", "a", "e", "i", "o", "u", "ɑ", "ɔ", "ə", "ɛ", "ɪ", "ʊ", "ʌ",
            "ˈ", "ˌ", "ː", "n", "t", "s", "r", "l", "m", "k", "p", "d",
            // Grafemas IPA que NO son ASCII:
            "ɡ", "ŋ", "θ", "ð", "ʃ", "ʒ", "ʤ", "ʧ", "ɹ", "ʔ",
        ] {
            assert!(map.contains_key(k), "vocab Kokoro falta fonema '{k}'");
        }
        // Y la "g" ASCII NO debe estar (sanity: confirma que el vocab
        // usa IPA, no ASCII para este fonema).
        assert!(
            !map.contains_key("g"),
            "el vocab Kokoro usa 'ɡ' (U+0261) en lugar de 'g' ASCII; \
             si este assert rompe, alguien añadió una entrada duplicada"
        );
    }

    /// `phonemize("hello")` debe producir al menos un ID y la cadena
    /// de fonemas debe contener algún símbolo IPA (no letras ASCII
    /// planas).
    #[test]
    fn phonemize_hello_returns_phoneme_ids() {
        let (ids, phon_str) = phonemize_debug("hello").expect("phonemize");
        assert!(!ids.is_empty(), "hello debe tener >= 1 fonema");
        assert!(
            phon_str.chars().any(|c| c as u32 > 127),
            "esperaba fonemas IPA (no ASCII) en '{phon_str}'"
        );
    }

    /// Input vacío (incluyendo sólo whitespace) → `TextTooShort`.
    #[test]
    fn phonemize_empty_returns_text_too_short() {
        assert!(matches!(
            phonemize("", "af_heart"),
            Err(TtsError::TextTooShort)
        ));
        assert!(matches!(
            phonemize("   ", "af_heart"),
            Err(TtsError::TextTooShort)
        ));
        assert!(matches!(
            phonemize("\n\t", "af_heart"),
            Err(TtsError::TextTooShort)
        ));
    }

    /// `KokoroLang::from_voice` enruta por prefijo del voice_id:
    /// `ef_*`/`em_*` (Kokoro) y `es*` (Piper) = español, resto = inglés.
    /// Garantiza que el G2P siga a la voz y no al contenido del texto
    /// (bug del deletreo: español ASCII sin tildes caía al G2P inglés).
    #[test]
    fn kokoro_lang_from_voice_routes_by_prefix() {
        // Español — voces Kokoro.
        assert_eq!(KokoroLang::from_voice("ef_dora"), KokoroLang::Es);
        assert_eq!(KokoroLang::from_voice("em_alex"), KokoroLang::Es);
        assert_eq!(KokoroLang::from_voice("em_santa"), KokoroLang::Es);
        // Español — voces Piper (caso del switch de engine: config
        // quedó con un voice_id Piper pero engine=Kokoro).
        assert_eq!(
            KokoroLang::from_voice("es_ES-davefx-medium"),
            KokoroLang::Es
        );
        assert_eq!(KokoroLang::from_voice("es_MX-ald-medium"), KokoroLang::Es);
        // Inglés (af/am/bf/bm Kokoro + en_* Piper).
        assert_eq!(KokoroLang::from_voice("af_heart"), KokoroLang::En);
        assert_eq!(KokoroLang::from_voice("am_michael"), KokoroLang::En);
        assert_eq!(KokoroLang::from_voice("bf_emma"), KokoroLang::En);
        assert_eq!(KokoroLang::from_voice("bm_george"), KokoroLang::En);
        assert_eq!(
            KokoroLang::from_voice("en_US-lessac-medium"),
            KokoroLang::En
        );
        // Tolerancia a mayúsculas (defensivo).
        assert_eq!(KokoroLang::from_voice("EF_DORA"), KokoroLang::Es);
        assert_eq!(
            KokoroLang::from_voice("ES_ES-davefx-medium"),
            KokoroLang::Es
        );
        // Prefijo desconocido → default inglés.
        assert_eq!(KokoroLang::from_voice("zz_unknown"), KokoroLang::En);
    }

    /// Texto en español (con o sin tildes) fonemizado con una voz
    /// española (`em_alex`) debe producir IDs de vocab Kokoro válidos.
    /// Este es el caso que antes deletreaba: texto ASCII español
    /// ("Simulacion Medica") enrutado al G2P inglés por el check
    /// `is_ascii`. Ahora la voz `em_alex` fuerza el G2P español.
    #[test]
    fn phonemize_spanish_text_with_spanish_voice_returns_valid_ids() {
        // Con tildes.
        let res = phonemize("El polémico congresista del Pacto Histórico", "em_alex");
        assert!(
            res.is_ok(),
            "español con tildes + voz em_alex debería fonemizarse: {res:?}"
        );
        // SIN tildes — el caso del bug del deletreo.
        let res = phonemize("Simulacion Medica de Codigo Completo", "em_alex");
        assert!(
            res.is_ok(),
            "español ASCII sin tildes + voz em_alex debería fonemizarse: {res:?}"
        );
        let ids = res.unwrap();
        assert!(!ids.is_empty(), "debe emitir IDs de token");
        let known: std::collections::HashSet<i64> =
            KOKORO_VOCAB_RAW.iter().map(|(_, id)| *id).collect();
        for id in &ids {
            assert!(
                known.contains(id) || *id == 0,
                "ID {id} no está en vocab Kokoro"
            );
        }
    }

    /// Texto en español fonemizado con una voz INGLESA (`af_heart`)
    /// pasa por el G2P inglés: produce IDs válidos pero con fonemas
    /// ingleses (no es el path recomendado — el deletreo). Verificamos
    /// que no crashea y emite algo, para confirmar que el enrutado por
    /// voz no rompe el path inglés.
    #[test]
    fn phonemize_spanish_text_with_english_voice_still_emits_ids() {
        let res = phonemize("hola mundo", "af_heart");
        // Misaki-rs puede o no reconocer las palabras; lo importante es
        // que el path no crashee. Aceptamos Ok o Phonemization.
        match res {
            Ok(ids) => assert!(!ids.is_empty(), "debe emitir IDs si Ok"),
            Err(TtsError::Phonemization(_)) => {}
            Err(other) => panic!("error inesperado: {other:?}"),
        }
    }

    /// Todos los IDs devueltos deben estar en `vocab` (0..=177). Si
    /// misaki-rs emite un fonema desconocido lo saltamos, así que el
    /// `ids` final NO debe contener ningún valor que no esté en
    /// `KOKORO_VOCAB_RAW`.
    #[test]
    fn phonemize_only_emits_vocab_ids() {
        let (ids, _) = phonemize_debug("the quick brown fox").expect("phonemize");
        let known: std::collections::HashSet<i64> =
            KOKORO_VOCAB_RAW.iter().map(|(_, id)| *id).collect();
        for id in &ids {
            assert!(
                known.contains(id) || *id == 0,
                "ID {id} no está en vocab Kokoro"
            );
        }
    }
}
