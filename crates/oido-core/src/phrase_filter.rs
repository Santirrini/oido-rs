//! Filtro de frases alucinadas por whisper.cpp.
//!
//! Dos capas complementarias:
//!
//! 1. **Blacklist** (`FILTER_PHRASES`): match exacto contra frases
//!    CtQ conocidas. ES + EN mezcladas. La comparación es exacta
//!    contra `trim().to_lowercase()`, no substring, para no matar
//!    texto legítimo del usuario que casualmente contenga la palabra
//!    "subscribe". Port + extensión del `text_processor.py` de Oido.
//!
//! 2. **Repetition Guard** (`is_repetition_loop`): detecta alucinaciones
//!    de "bucle" del tipo `X X X X ...` (ej. "me voy a decir"
//!    repetido 30 veces) que el blacklist no puede enumerar a priori.
//!    Detecta palabras, bigramas y trigramas que se repiten
//!    consecutivamente o casi-consecutivamente, y considera "loop"
//!    si la unidad repetida aparece al menos `REPETITION_MIN_COUNT`
//!    veces y cubre al menos la mitad del texto.
//!
//! `filter` aplica las dos capas (blacklist + repetition) y devuelve
//! `Some(text)` sólo si pasa ambas.

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

/// ¿El texto es un "bucle de repetición" típico de alucinación?
///
/// Heurística: tokeniza por whitespace y para cada tamaño `n` de
/// n-grama (1 a `MAX_NGRAM`) busca el n-grama más frecuente en el
/// texto (no requiere contigüidad — la alucinación del usuario
/// "me voy a decir que me voy a decir que ..." intercala un "que"
/// entre repeticiones, así que no son contiguas). El texto se
/// considera "loop" si la unidad repetida:
///
/// 1. Aparece al menos `REPETITION_MIN_COUNT` veces en total.
/// 2. Cubre al menos `REPETITION_COVERAGE_NUMERATOR /
///    REPETITION_COVERAGE_DENOMINATOR` del texto total (en palabras).
///
/// "Cubrir" significa: `unit.len() * unit_count` palabras del texto
/// están dentro de repeticiones de esa unidad. Equivale a "la
/// unidad repetida ocupa al menos la mitad del texto", pero se evalúa
/// por longitud ya cubierta en lugar de ratio sobre `words.len()` para
/// no desfavorecer n-gramas largos.
#[must_use]
pub fn is_repetition_loop(text: &str) -> bool {
    let normalized = text.trim().to_lowercase();
    let words: Vec<&str> = normalized.split_whitespace().collect();

    // Texto demasiado corto para ser un loop; dejarlo pasar.
    // Mínimo necesario: REPETITION_MIN_COUNT repeticiones de un unigrama.
    if words.len() < REPETITION_MIN_COUNT {
        return false;
    }

    let threshold = (words.len() * REPETITION_COVERAGE_NUMERATOR)
        / REPETITION_COVERAGE_DENOMINATOR;

    // Probar n-gramas de longitud 1, 2 y 3.
    for n in 1..=MAX_NGRAM {
        if words.len() < n * REPETITION_MIN_COUNT {
            continue;
        }
        let (unit, unit_count) = most_frequent_ngram(&words, n);
        let coverage = unit.len() * unit_count;
        if unit_count >= REPETITION_MIN_COUNT && coverage >= threshold {
            return true;
        }
    }

    false
}

/// Devuelve `Some(text)` si debe inyectarse, `None` si es alucinación.
#[must_use]
pub fn filter(text: &str) -> Option<&str> {
    if is_filtered(text) || is_repetition_loop(text) {
        None
    } else {
        Some(text)
    }
}

// ---------------- Repetition Guard internals ----------------

/// Longitud máxima del n-grama a probar (1=palabra, 2=bigrama, 3=trigrama).
const MAX_NGRAM: usize = 3;
/// Mínimo de repeticiones (de la misma unidad) para considerarlo loop.
const REPETITION_MIN_COUNT: usize = 5;
/// Cobertura mínima: la unidad repetida debe cubrir al menos esta
/// fracción del texto (numerador).
const REPETITION_COVERAGE_NUMERATOR: usize = 1;
/// Cobertura mínima: denominador (=> 1/2 = 50%).
const REPETITION_COVERAGE_DENOMINATOR: usize = 2;

/// Devuelve el n-grama de tamaño `n` más frecuente en `words`
/// (no requiere contigüidad) junto con su conteo. Si no hay ninguno
/// (texto vacío o `n == 0`), devuelve un slice vacío y 0.
///
/// Implementación: usa `windows(n)` y cuenta con un `HashMap<&[&str], usize>`
/// keyed por el slice. Como `[&str]` no implementa `Hash`/`Eq`
/// directamente, comparamos por longitud + igualdad de elementos.
fn most_frequent_ngram<'a>(words: &'a [&'a str], n: usize) -> (&'a [&'a str], usize) {
    static EMPTY: [&str; 0] = [];
    if n == 0 || words.len() < n {
        return (&EMPTY, 0);
    }

    let mut best: &'a [&'a str] = &EMPTY;
    let mut best_count: usize = 0;

    for window in words.windows(n) {
        // Contar cuántas veces aparece ESTE window en todo el array.
        // Comparación por igualdad de slices (los &str apuntan a la
        // misma cadena normalizada en minúsculas, así que '==' es O(n)
        // sobre el tamaño del n-grama, constante aquí).
        let count = words.windows(n).filter(|w| *w == window).count();
        if count > best_count {
            best_count = count;
            best = window;
        }
    }

    (best, best_count)
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

    // ---------------- Repetition Guard tests ----------------

    /// Caso del usuario: "Y yo, en español, me voy a decir que ..."
    /// repetido decenas de veces tras una frase introductoria. El bigrama
    /// "me voy" o "voy a" o el trigrama "me voy a" deberían disparar.
    #[test]
    fn real_world_user_alucination_caught() {
        let loop_text = "Y yo, en español, me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir que \
                         me voy a decir que me voy a decir que me voy a decir";
        assert!(is_repetition_loop(loop_text));
        assert_eq!(filter(loop_text), None);
    }

    #[test]
    fn bigram_loop_caught() {
        let s = "subscribe subscribe subscribe subscribe subscribe subscribe subscribe";
        assert!(is_repetition_loop(s));
    }

    #[test]
    fn unigram_loop_caught() {
        let s = "hola hola hola hola hola hola hola hola hola hola";
        assert!(is_repetition_loop(s));
    }

    #[test]
    fn normal_dialogue_passes() {
        assert!(!is_repetition_loop(
            "Hola, voy a dictar en español e inglés. Hello, I will dictate."
        ));
    }

    #[test]
    fn short_text_passes() {
        // Menos del mínimo de palabras para activar el detector.
        assert!(!is_repetition_loop("hola hola hola"));
        assert!(!is_repetition_loop("uno dos"));
    }

    #[test]
    fn partial_repetition_below_threshold_passes() {
        // Solo 3 repeticiones del bigrama (umbral = 5).
        let s = "subscribe to the channel, subscribe to the channel, \
                 subscribe to the channel, el resto del texto del usuario \
                 que es legítimo y no debe filtrarse";
        assert!(!is_repetition_loop(s));
    }

    #[test]
    fn filter_applies_both_layers() {
        // Blacklist
        assert_eq!(filter("thank you for watching"), None);
        // Repetición
        let long_loop = "x ".repeat(50);
        assert_eq!(filter(long_loop.trim()), None);
        // OK
        assert_eq!(filter("hola mundo"), Some("hola mundo"));
    }
}
