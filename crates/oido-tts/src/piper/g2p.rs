//! Wrapper sobre `piper-plus-g2p` para el subset de idiomas que v1.0
//! de oido soporta con Piper (ES/EN/PT/FR).
//!
//! ## Por qué un wrapper
//!
//! `piper-plus-g2p` no expone un único "G2p" polimórfico: cada idioma
//! tiene su propio phonemizer (`SpanishPhonemizer::new()`, `EnglishPhonemizer::new()`,
//! etc.) y cada uno con constructor distinto (`Self` vs `Result<Self, G2pError>`).
//!
//! Este módulo los homogeneiza detrás de un enum [`Language`] + una
//! función [`phonemize`] que devuelve `Vec<String>` (un string por
//! fonema, listo para [`super::phoneme::encode`]). El motor TTS del
//! bin no necesita saber qué idioma concreto está usando: le pasa el
//! texto y obtiene tokens.
//!
//! ## Idiomas soportados en F2
//!
//! | `Language` | Constructor upstream                  | Notas                       |
//! |------------|----------------------------------------|------------------------------|
//! | `Es`       | `SpanishPhonemizer::new() -> Self`     | Cero deps, pura regla       |
//! | `En`       | `EnglishPhonemizer::new() -> Result`   | Carga CMU dict en disco     |
//! | `Pt`       | `PortuguesePhonemizer::new() -> Self`  | Pura regla                  |
//! | `Fr`       | `FrenchPhonemizer::new() -> Self`      | Pura regla                  |
//!
//! Idiomas adicionales (ZH/JA/KO/SV) **no** se incluyen en F2: sus
//! constructores requieren archivos binarios de diccionarios externos
//! (CMUDict, NAIST-JDIC, etc.) que F7 se encargará de descargar.
//!
//! ## Cache
//!
//! Cada phonemizer se construye una vez por sesión y se cachea en un
//! `Mutex<Option<...>>`. Las llamadas `EnglishPhonemizer::new()` pueden
//! tardar ~50-200 ms en la primera llamada (carga del CMUDict de
//! ~1 MB); las siguientes son inmediatas. El Mutex es `parking_lot`
//! (workspace-wide) para mantener consistencia con `oido-stt`.

use std::sync::Arc;

use parking_lot::Mutex;
use piper_plus_g2p::phonemizer::Phonemizer as _;
use tracing::warn;

use crate::TtsError;

/// Idiomas soportados por el backend Piper en F2.
///
/// La representación canónica usa códigos BCP-47 cortos (`"es"`, `"en"`,
/// `"pt"`, `"fr"`). El constructor de cada variante vive en este módulo
/// para que `piper-plus-g2p` no se filtre al resto del crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Es,
    En,
    Pt,
    Fr,
}

impl Language {
    /// Código BCP-47 corto (`"es"`, `"en"`, …). Lo usa el caller para
    /// mostrar "Leyendo con Piper — es" en el tray.
    #[must_use]
    pub fn bcp47(self) -> &'static str {
        match self {
            Language::Es => "es",
            Language::En => "en",
            Language::Pt => "pt",
            Language::Fr => "fr",
        }
    }

    /// Parsea un código BCP-47 o `espeak.voice` (alias `"es-419"`,
    /// `"en-us"`, etc.) y devuelve el `Language` correspondiente.
    ///
    /// Acepta tanto la forma corta (`"es"`) como las extendidas
    /// (`"es-ES"`, `"es-MX"`, `"es-419"`) que aparecen en los
    /// `.onnx.json` bajo `espeak.voice`. Devuelve `None` si el código
    /// no es uno de los cuatro soportados — el caller debe rechazar
    /// la voz en lugar de caer a un idioma incorrecto.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        let prefix = code.split(['-', '_']).next().unwrap_or(code);
        match prefix.to_ascii_lowercase().as_str() {
            "es" => Some(Language::Es),
            "en" => Some(Language::En),
            "pt" => Some(Language::Pt),
            "fr" => Some(Language::Fr),
            _ => None,
        }
    }
}

/// Handle cacheado a un phonemizer concreto de `piper-plus-g2p`.
///
/// Cada variant es opaca al resto del crate: el caller sólo necesita
/// invocar [`phonemize`] sobre el enum [`G2p`]. Esto evita que
/// `piper-plus-g2p` aparezca en el API surface del backend Piper.
#[derive(Clone)]
pub enum G2p {
    Spanish(Arc<piper_plus_g2p::spanish::SpanishPhonemizer>),
    English(Arc<piper_plus_g2p::english::EnglishPhonemizer>),
    Portuguese(Arc<piper_plus_g2p::portuguese::PortuguesePhonemizer>),
    French(Arc<piper_plus_g2p::french::FrenchPhonemizer>),
}

impl std::fmt::Debug for G2p {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // No expongo el árbol interno de piper-plus-g2p en Debug — sólo
        // el idioma. Mantiene el `Debug` determinista y rápido.
        let lang = match self {
            G2p::Spanish(_) => "Spanish",
            G2p::English(_) => "English",
            G2p::Portuguese(_) => "Portuguese",
            G2p::French(_) => "French",
        };
        f.debug_struct("G2p").field("language", &lang).finish()
    }
}

impl G2p {
    /// Construye el phonemizer para `language`.
    ///
    /// - ES/PT/FR: constructores infallible (`Self`).
    /// - EN: `EnglishPhonemizer::new()` devuelve `Result` porque busca
    ///   el CMU dict en disco; si no lo encuentra, devolvemos
    ///   `TtsError::Phonemization("...")`.
    ///
    /// Los phonemizers son `Send + Sync` (internamente `Box<dyn Phonemizer>`
    /// con trait `Send + Sync`); los envolvemos en `Arc` para no pagar
    /// clones innecesarios al mover el handle entre threads.
    pub fn new(language: Language) -> Result<Self, TtsError> {
        match language {
            Language::Es => Ok(G2p::Spanish(Arc::new(
                piper_plus_g2p::spanish::SpanishPhonemizer::new(),
            ))),
            Language::Pt => Ok(G2p::Portuguese(Arc::new(
                piper_plus_g2p::portuguese::PortuguesePhonemizer::new(),
            ))),
            Language::Fr => Ok(G2p::French(Arc::new(
                piper_plus_g2p::french::FrenchPhonemizer::new(),
            ))),
            Language::En => piper_plus_g2p::english::EnglishPhonemizer::new()
                .map(|p| G2p::English(Arc::new(p)))
                .map_err(|e| TtsError::Phonemization(format!(
                    "no se pudo inicializar EnglishPhonemizer (¿falta CMUDict?): {e}"
                ))),
        }
    }

    /// Phonemiza `text` y devuelve una lista de strings, uno por fonema.
    ///
    /// Esta es la función pura del wrapper: la lógica de cacheo vive
    /// en [`phonemize_with_cache`].
    pub fn phonemize(&self, text: &str) -> Result<Vec<String>, TtsError> {
        let result = match self {
            G2p::Spanish(p) => p.phonemize_with_prosody(text),
            G2p::English(p) => p.phonemize_with_prosody(text),
            G2p::Portuguese(p) => p.phonemize_with_prosody(text),
            G2p::French(p) => p.phonemize_with_prosody(text),
        };
        match result {
            Ok((tokens, _prosody)) => Ok(tokens),
            Err(e) => Err(TtsError::Phonemization(format!("{e}"))),
        }
    }

    /// Idioma BCP-47 corto del phonemizer.
    #[must_use]
    pub fn language(&self) -> Language {
        match self {
            G2p::Spanish(_) => Language::Es,
            G2p::English(_) => Language::En,
            G2p::Portuguese(_) => Language::Pt,
            G2p::French(_) => Language::Fr,
        }
    }
}

/// Cache lazy de un phonemizer por idioma. El primer `get_or_init` para
/// un idioma construye el `G2p` y lo cachea; los siguientes son
/// instantáneos (sólo un Mutex lock).
///
/// ## Por qué cache
///
/// `EnglishPhonemizer::new()` parsea el CMU dict de ~1 MB en cada llamada
/// (mitigado parcialmente por un `OnceLock` interno del crate, pero la
/// construcción del struct sigue costando ~50 ms). Cachear a nuestro
/// nivel garantiza que sólo pagamos ese coste una vez por sesión de
/// lectura.
#[derive(Debug)]
pub struct PhonemizerCache {
    inner: Mutex<Option<(Language, Arc<G2p>)>>,
}

impl Default for PhonemizerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PhonemizerCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Devuelve el phonemizer cacheado para `language`, construyendo
    /// uno nuevo si el cache está vacío o si cambió el idioma.
    pub fn get_or_init(&self, language: Language) -> Result<Arc<G2p>, TtsError> {
        // Fast path: cache hit + mismo idioma → clonar Arc (cheap).
        {
            let guard = self.inner.lock();
            if let Some((cached_lang, cached_g2p)) = guard.as_ref() {
                if *cached_lang == language {
                    return Ok(Arc::clone(cached_g2p));
                }
            }
        }
        // Slow path: construir nuevo G2p y reemplazar el cache.
        // Re-toma el lock; si otro thread ganó la carrera, respetamos
        // su trabajo (sólo reconstruimos si el cache actual sigue
        // siendo de otro idioma).
        let new_g2p = Arc::new(G2p::new(language)?);
        let mut guard = self.inner.lock();
        if let Some((cached_lang, _)) = guard.as_ref() {
            if *cached_lang == language {
                // Otro thread ya cacheó el idioma que pedimos.
                // Liberamos nuestro recién-construido (se dropea al
                // salir del scope del Arc local) y devolvemos el
                // cacheado.
                if let Some((_, cached)) = guard.as_ref() {
                    return Ok(Arc::clone(cached));
                }
            } else {
                warn!(
                    anterior = ?cached_lang,
                    nuevo = ?language,
                    "PhonemizerCache: cambiando de idioma en caliente; \
                     el phonemizer anterior se descarta"
                );
            }
        }
        *guard = Some((language, Arc::clone(&new_g2p)));
        Ok(new_g2p)
    }
}

/// Atajo de conveniencia: `cache.get_or_init(lang)?.phonemize(text)`.
pub fn phonemize_with_cache(
    cache: &PhonemizerCache,
    language: Language,
    text: &str,
) -> Result<Vec<String>, TtsError> {
    let g2p = cache.get_or_init(language)?;
    g2p.phonemize(text)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_from_code_accepts_short_and_extended() {
        assert_eq!(Language::from_code("es"), Some(Language::Es));
        assert_eq!(Language::from_code("es-ES"), Some(Language::Es));
        assert_eq!(Language::from_code("es-MX"), Some(Language::Es));
        assert_eq!(Language::from_code("es-419"), Some(Language::Es));
        assert_eq!(Language::from_code("EN"), Some(Language::En));
        assert_eq!(Language::from_code("en_us"), Some(Language::En));
        assert_eq!(Language::from_code("pt"), Some(Language::Pt));
        assert_eq!(Language::from_code("pt-BR"), Some(Language::Pt));
        assert_eq!(Language::from_code("fr"), Some(Language::Fr));
        assert_eq!(Language::from_code("fr-FR"), Some(Language::Fr));
    }

    #[test]
    fn language_from_code_rejects_unsupported() {
        assert_eq!(Language::from_code("de"), None);
        assert_eq!(Language::from_code("it"), None);
        assert_eq!(Language::from_code("ja"), None);
        assert_eq!(Language::from_code("zh"), None);
        assert_eq!(Language::from_code(""), None);
        assert_eq!(Language::from_code("xxxx"), None);
    }

    #[test]
    fn language_bcp47_roundtrip() {
        for lang in [Language::Es, Language::En, Language::Pt, Language::Fr] {
            let code = lang.bcp47();
            assert_eq!(Language::from_code(code), Some(lang));
        }
    }

    #[test]
    fn phonemize_cache_returns_same_arc_on_second_call() {
        // ES no requiere assets en disco, así que es seguro para CI.
        let cache = PhonemizerCache::new();
        let a = cache.get_or_init(Language::Es).expect("init ES");
        let b = cache.get_or_init(Language::Es).expect("init ES 2");
        assert!(
            Arc::ptr_eq(&a, &b),
            "segunda llamada debe devolver el mismo Arc (cache hit)"
        );
    }

    #[test]
    fn phonemize_spanish_simple_text() {
        let cache = PhonemizerCache::new();
        let tokens = phonemize_with_cache(&cache, Language::Es, "hola")
            .expect("phonemize hola");
        // "hola" debe producir al menos un fonema (no vacío).
        assert!(
            !tokens.is_empty(),
            "phonemize('hola') no debe devolver lista vacía"
        );
    }
}