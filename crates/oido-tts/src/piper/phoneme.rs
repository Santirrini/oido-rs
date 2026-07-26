//! Codificador fonema → IDs de Piper (BOS/PAD/EOS + PUA lookup).
//!
//! ## Contrato con el grafo ONNX de Piper
//!
//! El modelo VITS de Piper espera un tensor `input` `i64` de shape
//! `[1, seq_len]` con la codificación BOS/PAD/EOS ya aplicada. Esta
//! convención viene de `rhasspy/piper/src/python_run/piper/voice.py`
//! (líneas 140-185, función `_phonemes_to_ids`) y del port Rust
//! `thewh1teagle/piper-rs` (líneas 70-100 de `model.rs`):
//!
//! ```text
//! ids = [BOS]
//! for phoneme in phonemes:
//!     ids += phoneme_id_map[phoneme]   # típicamente 1 ID
//!     ids += [PAD]
//! ids += [EOS]
//!
//! BOS = phoneme_id_map["^"][0]
//! EOS = phoneme_id_map["$"][0]
//! PAD = phoneme_id_map["_"][0]        # si falta, default 0
//! ```
//!
//! ## Sobre PUA
//!
//! Los `.onnx.json` de Piper mapean fonemas IPA "largos" (vocal larga,
//! africada) a **PUA codepoints** (U+E000–U+E0FF). `piper-plus-g2p`
//! produce esos PUA directamente en sus tokens (ver `token_map.rs`
//! upstream), así que nuestra búsqueda en el mapa puede ir con el
//! string completo del token (no hace falta descomponer carácter a
//! carácter — cada token ya es un único fonema).
//!
//! ## Estrategia "Skip" para tokens desconocidos
//!
//! El set de fonemas varía entre voces (un modelo entrenado con
//! espeak-ng trae otros PUA que otro entrenado con espeak clásico).
//! Si un token no aparece en el `phoneme_id_map` de ESTA voz, lo
//! descartamos con un `tracing::warn!` y seguimos. Esto evita que un
//! solo carácter raro aborte toda la síntesis.

use std::collections::HashMap;

use crate::TtsError;

/// Longitud máxima del tensor `input` que enviaremos a Piper.
///
/// Piper internamente usa longitudes variables, pero un cap fijo protege
/// contra entradas patológicas (un párrafo entero pegado a la selección
/// por error). 256 IDs equivalen a ~125 fonemas reales (cada uno emite
/// 1 ID + 1 PAD), suficiente para una oración larga de ~200-250 chars
/// sin truncar. Subir más allá de 511 empieza a requerir mucha memoria
/// BFCArena en CPU sin beneficio práctico para una lectura de selección.
pub const MAX_PHONEMES: usize = 256;

/// IDs especiales extraídos del `phoneme_id_map` de Piper.
///
/// Piper usa tres tokens de control que NO aparecen en el texto
/// fonemizado: `^` (BOS = inicio de secuencia), `_` (PAD = padding entre
/// fonemas) y `$` (EOS = fin de secuencia). Sus IDs numéricos varían
/// entre modelos (los determina el vocabulario entrenado, normalmente
/// 0/1/2 pero no garantizado).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialIds {
    pub bos: i64,
    pub pad: i64,
    pub eos: i64,
}

impl SpecialIds {
    /// Extrae BOS/EOS/PAD del `phoneme_id_map` con defaults robustos.
    ///
    /// - `^` es obligatorio (BOS). Si falta, devuelve
    ///   `TtsError::InvalidVoiceConfig`.
    /// - `$` es obligatorio (EOS). Si falta, igual.
    /// - `_` (PAD) es opcional: si falta usamos `0` (el default que usa
    ///   el port `piper-rs` cuando el modelo no lo trae).
    pub fn from_map(map: &HashMap<String, Vec<i64>>) -> Result<Self, TtsError> {
        let bos = first_id(map, "^")
            .ok_or_else(|| TtsError::InvalidVoiceConfig(
                "phoneme_id_map no contiene '^' (BOS)".into(),
            ))?;
        let eos = first_id(map, "$")
            .ok_or_else(|| TtsError::InvalidVoiceConfig(
                "phoneme_id_map no contiene '$' (EOS)".into(),
            ))?;
        let pad = first_id(map, "_").unwrap_or(0);
        Ok(Self { bos, pad, eos })
    }
}

fn first_id(map: &HashMap<String, Vec<i64>>, key: &str) -> Option<i64> {
    map.get(key).and_then(|v| v.first().copied())
}

/// Codifica una secuencia de fonemas (strings) en IDs listos para el
/// tensor `input` de Piper.
///
/// ## Comportamiento
///
/// 1. Inserta `BOS` al principio y `EOS` al final.
/// 2. Para cada token, busca `phoneme_id_map[token]` y añade TODOS los
///    IDs (típicamente 1, ocasionalmente 2 para PUA multi-ID).
/// 3. Tras cada fonema, inserta `PAD` (excepto tras EOS — el último
///    token es el cierre).
/// 4. **Trunca** a `MAX_PHONEMES` (128) si el texto produce más.
///    `truncated` se setea a `true` para que el caller pueda avisar al
///    usuario vía el tray.
/// 5. Los fonemas no encontrados se saltan con `tracing::warn!` y un
///    contador (`skipped`).
///
/// Devuelve `(ids, report)` donde `report` resume el truncate/skip
/// para observabilidad.
pub fn encode(
    tokens: &[String],
    map: &HashMap<String, Vec<i64>>,
    special: SpecialIds,
) -> (Vec<i64>, EncodeReport) {
    let mut report = EncodeReport::default();
    // Reserva aproximada: BOS + (token × 2) + EOS + padding de seguridad.
    let mut ids = Vec::with_capacity(tokens.len() * 3 + 4);
    ids.push(special.bos);
    // PAD inmediatamente tras BOS (convención del upstream; ver test
    // `test_piper_encoder_basic` en piper-plus-g2p).
    ids.push(special.pad);

    // Cap total: no superar MAX_PHONEMES contando IDs finales. Dejamos
    // 2 huecos para EOS + (potencial PAD de cierre si alguien lo quiere).
    let max_inner = MAX_PHONEMES.saturating_sub(3);

    for token in tokens {
        if ids.len() >= max_inner {
            report.truncated = true;
            break;
        }
        match map.get(token) {
            Some(id_list) if !id_list.is_empty() => {
                if ids.len() + id_list.len() >= max_inner {
                    report.truncated = true;
                    break;
                }
                for &id in id_list {
                    ids.push(id);
                }
                ids.push(special.pad);
            }
            _ => {
                report.skipped += 1;
                // Sólo loguear si el fonema tiene contenido visible.
                // Algunos G2P emiten tokens vacíos o puramente de control
                // que no tienen entrada en el mapa — son ruido esperado,
                // no algo que el usuario necesite ver en su log.
                if !token.trim().is_empty() {
                    tracing::debug!(
                        phoneme = %token,
                        "fonema ausente en phoneme_id_map; descartado"
                    );
                }
            }
        }
    }

    // EOS al final (sin PAD posterior — el tensor se queda en EOS).
    ids.push(special.eos);
    (ids, report)
}

/// Resumen de la operación de encoding. Útil para logging y para que el
/// caller muestre un aviso en el tray si hubo truncado.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EncodeReport {
    /// El tensor fue truncado por exceder `MAX_PHONEMES`.
    pub truncated: bool,
    /// Número de tokens descartados por no estar en el mapa.
    pub skipped: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_map() -> HashMap<String, Vec<i64>> {
        let mut m = HashMap::new();
        m.insert("^".into(), vec![1]); // BOS
        m.insert("_".into(), vec![0]); // PAD
        m.insert("$".into(), vec![2]); // EOS
        m.insert("a".into(), vec![10]); // fonema 'a' → 1 ID
        m.insert("k".into(), vec![30]);
        m.insert(" ".into(), vec![3]);
        m
    }

    #[test]
    fn special_ids_extracted_from_map() {
        let m = fixture_map();
        let s = SpecialIds::from_map(&m).expect("specials");
        assert_eq!(s.bos, 1);
        assert_eq!(s.pad, 0);
        assert_eq!(s.eos, 2);
    }

    #[test]
    fn special_ids_missing_bos_returns_error() {
        let mut m = fixture_map();
        m.remove("^");
        let err = SpecialIds::from_map(&m).unwrap_err();
        match err {
            TtsError::InvalidVoiceConfig(msg) => {
                assert!(
                    msg.contains("BOS") || msg.contains("^"),
                    "mensaje debería mencionar BOS/'^', obtuve: {msg}"
                );
            }
            other => panic!("esperaba InvalidVoiceConfig, obtuve: {other:?}"),
        }
    }

    #[test]
    fn special_ids_missing_pad_defaults_to_zero() {
        let mut m = fixture_map();
        m.remove("_");
        let s = SpecialIds::from_map(&m).expect("specials");
        assert_eq!(s.pad, 0, "PAD sin '_' en el mapa debe ser 0");
    }

    #[test]
    fn encode_produces_bos_at_start_and_eos_at_end() {
        let m = fixture_map();
        let specials = SpecialIds::from_map(&m).unwrap();
        let tokens = vec!["a".to_string()];
        let (ids, _) = encode(&tokens, &m, specials);
        assert_eq!(ids[0], specials.bos, "primer ID debe ser BOS");
        assert_eq!(*ids.last().unwrap(), specials.eos, "último ID debe ser EOS");
    }

    #[test]
    fn encode_inserts_pad_between_phonemes() {
        let m = fixture_map();
        let specials = SpecialIds::from_map(&m).unwrap();
        let tokens = vec!["a".to_string(), "k".to_string()];
        let (ids, _) = encode(&tokens, &m, specials);
        // Esperado: [BOS, PAD, a(10), PAD, k(30), PAD, EOS]
        // (PAD tras BOS es convención del upstream.)
        assert_eq!(ids, vec![specials.bos, specials.pad, 10, specials.pad, 30, specials.pad, specials.eos]);
    }

    #[test]
    fn encode_skips_unknown_phonemes_with_warning() {
        let m = fixture_map();
        let specials = SpecialIds::from_map(&m).unwrap();
        let tokens = vec!["a".to_string(), "Z".to_string(), "k".to_string()];
        let (ids, report) = encode(&tokens, &m, specials);
        assert_eq!(report.skipped, 1);
        assert!(!ids.contains(&99), "el token desconocido 'Z' no debe colarse como 99");
        assert!(ids.contains(&10));
        assert!(ids.contains(&30));
    }

    #[test]
    fn encode_handles_multi_id_phoneme() {
        let mut m = fixture_map();
        // 'ã' mapea a dos IDs (PUA multi-id en algunos modelos).
        m.insert("ã".into(), vec![40, 41]);
        let specials = SpecialIds::from_map(&m).unwrap();
        let tokens = vec!["a".to_string(), "ã".to_string()];
        let (ids, _) = encode(&tokens, &m, specials);
        // Los dos IDs del multi-mapping deben aparecer en orden.
        let pos40 = ids.iter().position(|&x| x == 40).unwrap();
        let pos41 = ids.iter().position(|&x| x == 41).unwrap();
        assert!(pos40 < pos41, "los IDs de un multi-id deben ir en orden");
    }

    #[test]
    fn encode_truncates_to_max_phonemes() {
        // Construimos 300 tokens de un solo ID cada uno. Con PADs entre
        // ellos, ids.len() >> MAX_PHONEMES.
        let m = fixture_map();
        let specials = SpecialIds::from_map(&m).unwrap();
        let tokens: Vec<String> = (0..300).map(|_| "a".into()).collect();
        let (ids, report) = encode(&tokens, &m, specials);
        assert!(report.truncated, "truncated flag debe estar activo");
        assert!(
            ids.len() <= MAX_PHONEMES,
            "ids.len()={} debe ser <= MAX_PHONEMES={}",
            ids.len(),
            MAX_PHONEMES
        );
        // Debe terminar con EOS incluso si fue truncado.
        assert_eq!(*ids.last().unwrap(), specials.eos);
    }

    #[test]
    fn encode_empty_tokens_still_emits_bos_and_eos() {
        let m = fixture_map();
        let specials = SpecialIds::from_map(&m).unwrap();
        let (ids, report) = encode(&[], &m, specials);
        assert_eq!(ids, vec![specials.bos, specials.pad, specials.eos]);
        assert_eq!(report, EncodeReport::default());
    }
}