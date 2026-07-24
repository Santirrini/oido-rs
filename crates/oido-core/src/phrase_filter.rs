//! Filtro de frases alucinadas por whisper.cpp.
//!
//! Tres capas complementarias:
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
//! 3. **Artifact Guard** (`is_likely_artifact`): detecta resultados que
//!    empiezan con un token de cierre huérfano (`)`, `]`, `}`, `»`, `›`)
//!    seguido de contenido alfanumérico. Es síntoma de que el decoder
//!    no encontró un inicio coherente y emitió puntuación de cierre
//!    como primer token (ej. `), so as anything that information is
//!    about...`). Atrapa lo que el "best decoder" entrega cuando el
//!    decoder en bucle fue descartado pero ningún candidato era bueno.
//!
//! `filter` aplica las tres capas y devuelve `Some(text)` sólo si pasa
//! todas.
//!
//! ## Limitación conocida
//!
//! Ninguna de las tres capas detecta una **frase coherente insertada al
//! final** de una transcripción correcta (ej. `"...texto válido.
//! Considera el hecho de que eso es una caza de ropa."`). Ese patrón
//! requiere análisis semántico. El remedio no está aquí sino arriba en
//! el pipeline: anclar el initial_prompt del decoder (`no_context=false`
//! en `build_base_params`) para que el modelo tenga contexto léxico y no
//! divague al final de segmentos largos.

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

    let threshold = (words.len() * REPETITION_COVERAGE_NUMERATOR) / REPETITION_COVERAGE_DENOMINATOR;

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

/// ¿El texto empieza con un token de cierre huérfano?
///
/// Síntoma: cuando todos los decoders fallan o entran en bucle,
/// whisper.cpp a veces entrega como "best" un resultado que arranca con
/// puntuación de cierre (`)`, `]`, `}`, `»`, `›`) sin su apertura
/// correspondiente, seguida de más contenido. Eso no es dictado
/// legítimo — es un primer token alucinado.
///
/// La heurística exige:
/// 1. El texto (sin whitespace inicial) empieza con uno de los tokens
///    de cierre listados, opcionalmente seguido de otro signo de
///    puntuación (`,`, `.`, `;`, `:`, `!`, `?`).
/// 2. Tras esa secuencia inicial hay contenido alfanumérico. Así
///    evitamos falsos positivos con dictados válidos de un solo
///    signo (ej. "cierra paréntesis" → ")").
///
/// Falsos positivos evitados a propósito:
/// - `(`, `[`, `{`: son aperturas, legítimas como inicio de frase.
/// - `¿`, `¡`: signos de apertura del español, siempre legítimos.
/// - `,` `.` `;` solos al inicio: pueden ser continuación de edición.
#[must_use]
pub fn is_likely_artifact(text: &str) -> bool {
    let t = text.trim_start();
    if t.is_empty() {
        return false;
    }
    let first = t.chars().next().expect("no-empty after trim");
    if !CLOSING_PUNCT.contains(&first) {
        return false;
    }
    // Saltar una secuencia opcional de puntuación de continuación
    // (`),`, `].`, `» `) para llegar al contenido sustantivo.
    let rest: String = t
        .chars()
        .skip_while(|c| CLOSING_PUNCT.contains(c) || CONTINUATION_PUNCT.contains(c))
        .collect();
    rest.chars().any(|c| c.is_alphanumeric())
}

/// Devuelve `Some(text)` si debe inyectarse, `None` si es alucinación.
#[must_use]
pub fn filter(text: &str) -> Option<&str> {
    if is_filtered(text) || is_repetition_loop(text) || is_likely_artifact(text) {
        None
    } else {
        Some(text)
    }
}

// ---------------- Repetition Guard internals ----------------

/// Caracteres de puntuación de cierre cuyo uso como primer token del
/// resultado es síntoma de alucinación del decoder. NO incluye
/// aperturas (`(`, `[`, `{`, `¿`, `¡`) que son legítimas al inicio.
const CLOSING_PUNCT: &[char] = &[')', ']', '}', '»', '›'];
/// Signos que pueden seguir a un cierre huérfano en el token alucinado
/// (ej. `),`, `].`). Se saltan al buscar el contenido sustantivo.
const CONTINUATION_PUNCT: &[char] = &[',', '.', ';', ':', '!', '?'];

/// Longitud máxima del n-grama a probar (1=palabra ... 6=frase corta).
///
/// Antes era 3 (hasta trigrama), pero eso dejaba escapar bucles de
/// frases largas: el caso real "se han quedado en la cadera" (5
/// palabras) repetido 7 veces tras texto válido no se cazaba porque el
/// trigrama "se han quedado" solo cubre 21 palabras sobre 88 (< umbral).
/// El 5-grama "se han quedado en la" sí cubre 35 y dispara el filtro.
/// 6 es suficiente para frases alucinadas típicas de whisper sin
/// dispararse en prosa legítima (requiere `REPETITION_MIN_COUNT`
/// repeticiones exactas de la MISMA secuencia de 6 palabras).
const MAX_NGRAM: usize = 6;
/// Mínimo de repeticiones (de la misma unidad) para considerarlo loop.
const REPETITION_MIN_COUNT: usize = 5;
/// Cobertura mínima: la unidad repetida debe cubrir al menos esta
/// fracción del texto (numerador).
const REPETITION_COVERAGE_NUMERATOR: usize = 1;
/// Cobertura mínima: denominador (=> 1/3 ~= 33%).
///
/// Antes era 1/2 (50%): dejaba escapar repeticiones al final de una
/// transcripción correcta (caso real: "se han quedado en la cadera"
/// repetido 7 veces tras ~20 palabras válidas no llegaba al 50% de
/// cobertura total). 1/3 lo caza sin tocar el `REPETITION_MIN_COUNT=5`,
/// que sigue exigiendo ≥5 repeticiones de la misma unidad para acotar
/// los falsos positivos.
const REPETITION_COVERAGE_DENOMINATOR: usize = 3;

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

    /// Caso real del log: tras una transcripción correcta inicial, el
    /// decoder (sin anclaje de prompt) entró en bucle repitiendo "se
    /// han quedado en la cadera" 7 veces. Con el umbral anterior de
    /// cobertura 1/2 esto NO se cazaba porque la repetición no llegaba
    /// al 50% del texto total (había ~20 palabras válidas antes). Con
    /// 1/3 sí.
    #[test]
    fn trailing_repetition_after_valid_text_caught() {
        let text = "Y, ya que no hay lugar para hidrón en la naturaleza \
                    de las violaciones, los dos son los que les permiten \
                    que los preditores se pongan en la casa. A un mismo \
                    tiempo, los que se han hecho poner en la cadera, \
                    se han quitado y se han quedado en la cadera. \
                    Se han quedado en la cadera. Se han quedado en la \
                    cadera. Se han quedado en la cadera. Se han quedado \
                    en la cadera. Se han quedado en la cadera. Se han \
                    quedado en la cadera.";
        assert!(
            is_repetition_loop(text),
            "la repetición final debe cazarse con cobertura 1/3"
        );
        assert_eq!(filter(text), None);
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
    fn filter_applies_all_layers() {
        // Blacklist
        assert_eq!(filter("thank you for watching"), None);
        // Repetición
        let long_loop = "x ".repeat(50);
        assert_eq!(filter(long_loop.trim()), None);
        // Artifact (token de cierre huérfano)
        assert_eq!(filter("), so as anything that information is about"), None);
        // OK
        assert_eq!(filter("hola mundo"), Some("hola mundo"));
    }

    // ---------------- Artifact Guard tests ----------------

    /// Caso real del log: tras descartar un repetition-loop, el "best
    /// decoder" entregó `), so as anything that information is about,
    /// this might be the answer.` como primer token. Eso es un token
    /// de cierre huérfano y debe cazarse.
    #[test]
    fn orphaned_closing_token_caught() {
        assert!(is_likely_artifact(
            "), so as anything that information is about"
        ));
        assert!(is_likely_artifact("]. algo"));
        assert!(is_likely_artifact("} más texto"));
        assert!(is_likely_artifact("», dijo ella"));
        assert_eq!(filter("), so as anything that information is about"), None);
    }

    #[test]
    fn opening_punct_passes() {
        // Aperturas son legítimas al inicio de frase.
        assert!(!is_likely_artifact("(hola mundo)"));
        assert!(!is_likely_artifact("[nota al margen]"));
        assert!(!is_likely_artifact("{clave: valor}"));
    }

    #[test]
    fn spanish_inverted_marks_pass() {
        // ¿ ¡ son aperturas del español, nunca artifacts.
        assert!(!is_likely_artifact("\u{bf}cómo estás?")); // «¿cómo estás?»
        assert!(!is_likely_artifact("\u{a1}qué bueno!")); // «¡qué bueno!»
    }

    #[test]
    fn bare_closing_punct_passes() {
        // Un solo signo de cierre sin contenido alfanumérico posterior
        // puede ser dictado legítimo ("cierra paréntesis" → ")").
        assert!(!is_likely_artifact(")"));
        assert!(!is_likely_artifact("] "));
        assert!(!is_likely_artifact("},."));
    }

    #[test]
    fn normal_text_passes_artifact_check() {
        assert!(!is_likely_artifact("Hola, ¿cómo estás?"));
        assert!(!is_likely_artifact("The quick brown fox"));
        assert!(!is_likely_artifact(""));
        assert!(!is_likely_artifact("   "));
    }
}
