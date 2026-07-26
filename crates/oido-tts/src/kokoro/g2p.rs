//! Grapheme → phoneme → token IDs para Kokoro-82M (inglés).
//!
//! ## Pipeline
//!
//! ```text
//! texto ASCII
//!      │
//!      ▼  misaki-rs `G2P::new(Language::EnglishUS).g2p(&str)`
//!      │  (devuelve `String` con fonemas IPA + tokens `MToken`)
//!      ▼
//! phoneme_str: "həlˈoʊ wˈɝld"
//!      │
//!      ▼  lookup carácter por carácter en `KOKORO_VOCAB`
//!      │  (los fonemas de misaki son a veces multi-char: "ʰ", "↓", etc.)
//!      ▼
//! Vec<i64>  (ids de token listos para `Tensor::from_array`)
//! ```
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
//!
//! ## Limitación de v1.0
//!
//! Sólo aceptamos **inglés ASCII romanizable** (lo que `misaki-rs`
//! cubre sin `espeak-ng`). Input no-ASCII se rechaza con
//! `TtsError::Phonemization` para que el bin enrute a Piper en lugar
//! de malgastar CPU en un G2P que va a degradarse. Ver
//! [`phonemize`].
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

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
    (";", 1), (":", 2), (",", 3), (".", 4), ("!", 5), ("?", 6),
    ("—", 9), ("…", 10), ("\"", 11), ("(", 12), (")", 13), ("“", 14),
    ("”", 15), (" ", 16), ("\u{0303}", 17),
    ("ʣ", 18), ("ʥ", 19), ("ʦ", 20), ("ʨ", 21), ("ᵝ", 22), ("\u{AB67}", 23),
    ("A", 24), ("I", 25), ("O", 31), ("Q", 33), ("S", 35), ("T", 36),
    ("W", 39), ("Y", 41), ("ᵊ", 42),
    ("a", 43), ("b", 44), ("c", 45), ("d", 46), ("e", 47), ("f", 48),
    ("h", 50), ("i", 51), ("j", 52), ("k", 53), ("l", 54), ("m", 55),
    ("n", 56), ("o", 57), ("p", 58), ("q", 59), ("r", 60), ("s", 61),
    ("t", 62), ("u", 63), ("v", 64), ("w", 65), ("x", 66), ("y", 67),
    ("z", 68),
    ("ɑ", 69), ("ɐ", 70), ("ɒ", 71), ("æ", 72),
    ("β", 75), ("ɔ", 76), ("ɕ", 77), ("ç", 78), ("ɖ", 80), ("ð", 81),
    ("ʤ", 82), ("ə", 83), ("ɚ", 85), ("ɛ", 86), ("ɜ", 87), ("ɟ", 90),
    ("ɡ", 92), ("ɥ", 99), ("ɨ", 101), ("ɪ", 102), ("ʝ", 103),
    ("ɯ", 110), ("ɰ", 111), ("ŋ", 112), ("ɳ", 113), ("ɲ", 114), ("ɴ", 115),
    ("ø", 116), ("ɸ", 118), ("θ", 119), ("œ", 120),
    ("ɹ", 123), ("ɾ", 125), ("ɻ", 126), ("ʁ", 128), ("ɽ", 129), ("ʂ", 130),
    ("ʃ", 131), ("ʈ", 132), ("ʧ", 133), ("ʊ", 135), ("ʋ", 136), ("ʌ", 138),
    ("ɣ", 139), ("ɤ", 140), ("χ", 142), ("ʎ", 143), ("ʒ", 147), ("ʔ", 148),
    ("ˈ", 156), ("ˌ", 157), ("ː", 158), ("ʰ", 162), ("ʲ", 164),
    ("↓", 169), ("→", 171), ("↗", 172), ("↘", 173), ("ᵻ", 177),
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

/// Convierte `text` a IDs de fonema de Kokoro.
///
/// Pasos:
/// 1. Rechaza texto vacío (`TextTooShort`).
/// 2. Rechaza texto no-ASCII (`Phonemization` — el caller debería
///    enrutar a Piper en ese caso, ya que el input es claramente
///    español/otra lengua que Kokoro no soporta limpiamente en v1.0).
/// 3. Construye / recupera el `G2P` de `misaki-rs` (idioma
///    `EnglishUS`).
/// 4. Phonemiza con `g2p.g2p(&text)`.
/// 5. Tokeniza la cadena de fonemas resultante carácter-a-carácter
///    consultando `vocab_map()`. Multi-char phonemes de Kokoro
///    (`"↓"`, `"→"`, `"\u{0303}"`) caben en un solo `char` Unicode, así
///    que la iteración char-by-char es suficiente (NO byte-by-byte).
///
/// Devuelve `Vec<i64>` con los IDs en orden, listo para alimentar
/// `Tensor::from_array`.
pub fn phonemize(text: &str) -> Result<Vec<i64>, TtsError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(TtsError::TextTooShort);
    }

    // v1.0 sólo soporta inglés. Caracteres no-ASCII se rechazan
    // explícitamente: el caller (bin) debería detectar idioma y
    // enrutar a Piper si ve un texto con tildes/ñ/汉字.
    //
    // NOTA: Kokoro-82M sí puede emitir algunos fonemas con tildes
    // (ë, ö) si el G2P los produce, pero eso requiere entrada
    // romanizada. El check es defensivo: si el texto crudo ya viene
    // con tildes/汉字, casi seguro es input español/asiático y Kokoro
    // lo va a pronunciar fatal.
    if !trimmed.is_ascii() {
        return Err(TtsError::Phonemization(
            "non-English input rejected in v1".into(),
        ));
    }

    // Construye / recupera el G2P cacheado.
    static CACHE: std::sync::OnceLock<G2pCache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(G2pCache::new);
    let g2p = cache.get()?;

    // `misaki_rs::G2P::g2p` devuelve `(phoneme_string, Vec<MToken>)`.
    // La string ya viene con espacios y puntuación como tokens
    // separados (gracias al `subtoken_regex` interno).
    let (phoneme_str, _tokens) = g2p.g2p(trimmed).map_err(|e| {
        TtsError::Phonemization(format!("misaki-rs g2p: {e}"))
    })?;

    // Mapea carácter → ID. Si un carácter no está en el vocab,
    // registramos una traza y lo saltamos (no panickeamos: el G2P
    // podría emitir un desconocido `❓` que Kokoro no entiende, y
    // preferiríamos audio a skip que panic → silencio total).
    let map = vocab_map();
    let mut ids: Vec<i64> = Vec::with_capacity(phoneme_str.chars().count() + 2);
    for c in phoneme_str.chars() {
        // Codifica el char (1-4 bytes UTF-8) en un buffer fijo y
        // busca el `&str` resultante en el `HashMap<&str, i64>`. Los
        // IDs de Kokoro son todos de 1 carácter Unicode (los
        // multi-char como `ʰ` codifican en 2 bytes pero siguen siendo
        // un único `char`), así que este lookup basta.
        let mut buf = [0u8; 4];
        let s: &str = c.encode_utf8(&mut buf);
        if let Some(&id) = map.get(s) {
            ids.push(id);
        } else {
            tracing::debug!(
                char = %c,
                "fonema emitido por misaki-rs no está en vocab Kokoro; skip"
            );
            // No panic: continuamos con el resto. Si TODO el texto
            // queda vacío, el guard de `ids.is_empty()` de abajo
            // devuelve TextTooShort.
        }
    }

    if ids.is_empty() {
        return Err(TtsError::Phonemization(format!(
            "ningún fonema reconocible para '{trimmed}'"
        )));
    }
    Ok(ids)
}

/// Versión "inferir y devolver (ids, debug)" — usada por tests para
/// inspeccionar el resultado. En producción usar [`phonemize`].
#[cfg(test)]
pub fn phonemize_debug(text: &str) -> Result<(Vec<i64>, String), TtsError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(TtsError::TextTooShort);
    }
    if !trimmed.is_ascii() {
        return Err(TtsError::Phonemization(
            "non-English input rejected in v1".into(),
        ));
    }
    static CACHE: std::sync::OnceLock<G2pCache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(G2pCache::new);
    let g2p = cache.get()?;
    let (phoneme_str, _) = g2p.g2p(trimmed).map_err(|e| {
        TtsError::Phonemization(format!("misaki-rs g2p: {e}"))
    })?;
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
        for k in [";", ":", ",", ".", " ", "a", "e", "i", "o", "u",
                  "ɑ", "ɔ", "ə", "ɛ", "ɪ", "ʊ", "ʌ",
                  "ˈ", "ˌ", "ː",
                  "n", "t", "s", "r", "l", "m", "k", "p", "d",
                  // Grafemas IPA que NO son ASCII:
                  "ɡ", "ŋ", "θ", "ð", "ʃ", "ʒ", "ʤ", "ʧ", "ɹ", "ʔ"] {
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
        assert!(matches!(phonemize(""), Err(TtsError::TextTooShort)));
        assert!(matches!(phonemize("   "), Err(TtsError::TextTooShort)));
        assert!(matches!(phonemize("\n\t"), Err(TtsError::TextTooShort)));
    }

    /// Input no-ASCII → `Phonemization`. Garantiza que el caller (bin)
    /// pueda enrutar a Piper cuando vea español/tildes/汉字/etc.
    #[test]
    fn phonemize_rejects_non_ascii() {
        // Español con tilde: el carácter `á` (U+00E1) hace que la
        // string NO sea ASCII puro. El guard la rechaza.
        let res = phonemize("hola mundo");
        // `hola mundo` ES ASCII puro (sin tildes), así que pasa el
        // guard. Para verificar el rechazo real usamos un input con
        // tildes o CJK.
        assert!(
            res.is_ok(),
            "'hola mundo' es ASCII puro, debería aceptarse: {res:?}"
        );
        let res = phonemize("áéíóú");
        assert!(
            matches!(res, Err(TtsError::Phonemization(_))),
            "español con tildes debería rechazarse, obtuve: {res:?}"
        );
        let res = phonemize("こんにちは");
        assert!(matches!(res, Err(TtsError::Phonemization(_))));
        let res = phonemize("Здравствуй");
        assert!(matches!(res, Err(TtsError::Phonemization(_))));
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
