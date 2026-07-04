//! Filtro de frases alucinadas por whisper.cpp.
//!
//! Port + extensión ES del `text_processor.py` de Oido. Las frases
//! listadas son CtQ conocidas que el modelo emite cuando el usuario
//! deja de hablar (silencio) o al final del audio. La comparación es
//! exacta contra `trim().to_lowercase()`, no substring, para no matar
//! texto legítimo del usuario que casualmente contenga la palabra
//! "subscribe".
//!
//! YAGNI: si aparecen más alucinaciones en la práctica, se añaden al
//! set. No merece la pena un catálogo on-the-fly por ahora (Fase 1).

/// Lista cerrada de frases a descartar. Match exacto contra
/// `trim().to_lowercase()`. ES + EN mezcladas.
pub const FILTER_PHRASES: &[&str] = &[
    // EN
    "thank you for watching",
    "thanks for watching",
    "thank you for watching!",
    "thanks for watching!",
    "please subscribe",
    "like and subscribe",
    "subscribe to my channel",
    "see you in the next video",
    "see you next time",
    "bye bye",
    // ES
    "gracias por ver",
    "gracias por ver!",
    "\u{a1}gracias por ver!", // «¡gracias por ver!»
    "gracias por verlo",
    "suscr\u{ed}bete", // «suscríbete»
    "suscribete",
    "no olvides suscribirte",
    "hasta la pr\u{f3}xima", // «hasta la próxima»
    "hasta la proxima",
    "hasta el pr\u{f3}ximo video",
    "nos vemos en el pr\u{f3}ximo video",
    "nos vemos",
];

/// ¿El texto completo coincide con una frase de la blacklist?
/// Case-insensitive sobre el texto recortado.
#[must_use]
pub fn is_filtered(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    FILTER_PHRASES.iter().any(|p| *p == t)
}

/// Devuelve `Some(text)` si debe inyectarse, `None` si es alucinación.
#[must_use]
pub fn filter(text: &str) -> Option<&str> {
    if is_filtered(text) {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_discarded_en() {
        assert!(is_filtered("Thank you for watching"));
        assert!(is_filtered("thanks for watching!"));
        assert!(is_filtered("  please subscribe  "));
    }

    #[test]
    fn exact_match_discarded_es() {
        assert!(is_filtered("\u{a1}gracias por ver!"));
        assert!(is_filtered("suscr\u{ed}bete"));
    }

    #[test]
    fn substring_does_not_match() {
        // El usuario podría decir "podéis suscribiros al canal"
        // sin que sea un filtro.
        assert!(!is_filtered("pod\u{e9}is suscribiros al canal"));
        assert!(!is_filtered("no olvides suscribirte al canal"));
    }

    #[test]
    fn normal_text_kept() {
        assert!(!is_filtered("hola mundo"));
        assert!(!is_filtered("esto es un test"));
    }

    #[test]
    fn filter_helper_returns_none_on_match() {
        assert_eq!(filter("thank you for watching"), None);
        assert_eq!(filter("hola"), Some("hola"));
    }
}
