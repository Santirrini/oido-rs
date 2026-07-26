//! Catálogo maestro de voces TTS para sanitización de la config.
//!
//! Este módulo existe **separado** de los `voices` module-local de
//! cada backend (Piper y Kokoro) porque:
//!
//! - La sanitización corre en el bin antes de que el engine esté
//!   instanciado (necesita saber qué voces son válidas sin cargar
//!   el modelo).
//! - El set tiene que ser estable entre versiones (la config persiste
//!   IDs en disco).
//!
//! Mantiene una sola lista canónica. Los backends tienen sus propios
//! catálogos internos para mostrar al submenú del tray (más
//! extensos), pero la config sólo guarda IDs de esta lista.

/// Catálogo maestro. Mantener el set **finito** y **estable**.
/// Lista toda voz Piper o Kokoro conocida; otras voces (descargadas
/// dinámicamente) **no** entran acá porque el submenú estático del tray
/// las ignora.
pub const PIPER_DEFAULT_VOICES: &[&str] = &[
    "es_ES-davefx-medium",
    "es_MX-ald-medium",
    "en_US-lessac-medium",
];

pub const KOKORO_DEFAULT_VOICES: &[&str] = &["af_heart", "am_michael", "bf_emma"];

/// Devuelve todos los IDs de voz canónicos. Usada por
/// `sanitize_config` en el bin para detectar IDs corruptos.
#[must_use]
pub fn known_voice_ids() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = PIPER_DEFAULT_VOICES.to_vec();
    out.extend_from_slice(KOKORO_DEFAULT_VOICES);
    out
}

/// Voz por defecto del motor Piper. Coincide con el default en
/// `TtsConfig::default` y en `oido-models::TTS_VOICES_KOKORO` no —
/// Piper es el motor default en v1.0.
#[must_use]
pub fn piper_default_voice() -> &'static str {
    "es_ES-davefx-medium"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_voice_ids_non_empty_and_unique() {
        let v = known_voice_ids();
        assert!(!v.is_empty());
        let mut seen = std::collections::HashSet::new();
        for id in v {
            assert!(seen.insert(id), "voice duplicado en catálogo maestro: {id}");
        }
    }

    #[test]
    fn known_ids_include_config_defaults() {
        let v = known_voice_ids();
        // Piper default = `es_ES-davefx-medium`.
        assert!(
            v.contains(&piper_default_voice()),
            "Piper default {} debe estar en catálogo maestro",
            piper_default_voice()
        );
    }
}
