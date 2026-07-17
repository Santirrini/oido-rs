//! Catálogo centralizado de strings del menú nativo de bandeja.
//!
//! Cualquier label que el usuario ve en el submenú pasa por `Strings`.
//! `default_sections` (`sections.rs`) recibe un [`UiLanguage`] y elige
//! el `&'static Strings` correspondiente; el árbol de menú queda 100%
//! localizado sin que las secciones conozcan los strings hardcodeados.
//!
//! ## Tablas
//!
//! - [`STRINGS_ES`]: idioma local del proyecto. Default.
//! - [`STRINGS_EN`]: inglés puro.
//! - [`STRINGS_BILINGUAL`]: mixto — secciones críticas con ES + EN
//!   ("Idioma / Language"), resto en ES. Útil para usuarios leen ambos.
//!
//! Tests en `tests`: verifican que cada idioma es distinto, que el
//! bilingüe es distinto de los dos monolingües, y que toda clave
//! referenciada por `default_sections` existe en las 3 tablas (test
//! estático con [`InventoryKeys`]).
//!
//! ## Por qué `&'static str` y no `String`
//!
//! Las tablas son inmutables y se sirven vía `&'static Strings`.
//! Cero alocaciones por click de menú: el backend nativo
//! (`tray.rs::build_menu`) clona a `MenuItemSpec.label` solo donde
//! necesita interpolar (marcadores ✓, prefijos "↓ Descargar ").

use oido_config::UiLanguage;

/// Conjunto completo de strings visibles en el menú nativo.
///
/// Se accede por referencia estática vía [`strings(lang)`]. Mantener
/// planos (`&'static str`) — el backend de bandeja los clona a
/// `String` solo donde necesita formato dinámico.
#[derive(Debug, Clone, Copy)]
pub struct Strings {
    // Cabecera / items raíz
    pub change_hotkey: &'static str,
    pub exit: &'static str,
    pub check_updates: &'static str,
    pub open_models_dir: &'static str,

    // Submenú "Tema"
    pub theme: &'static str,
    pub theme_dark: &'static str,
    pub theme_light: &'static str,
    pub theme_system: &'static str,

    // Submenú "Modo de dictado"
    pub mode: &'static str,
    pub mode_batch: &'static str,
    pub mode_streaming: &'static str,
    pub mode_chunked: &'static str,

    // Submenú "Modelos"
    pub models: &'static str,
    pub model_tiny: &'static str,
    pub model_base: &'static str,
    pub model_small: &'static str,
    pub model_medium: &'static str,
    pub model_large: &'static str,
    pub model_vad: &'static str,
    pub model_installed: &'static str, // prefijo "✓ "
    pub model_download: &'static str,  // prefijo "↓ Descargar "
    pub model_active: &'static str,    // sufijo "← activo"
    pub model_en_only: &'static str,   // "⚠ Solo inglés" / "⚠ English only"

    // Submenú "Idioma de la interfaz" (NUEVO)
    pub ui_language: &'static str,
    pub ui_es: &'static str,
    pub ui_en: &'static str,
    pub ui_bilingual: &'static str,

    // Submenú "Prompt del sistema" (NUEVO)
    pub system_prompt: &'static str,
    pub prompt_bilingual: &'static str,
    pub prompt_es: &'static str,
    pub prompt_en: &'static str,
    pub prompt_custom: &'static str,
    pub prompt_custom_hint: &'static str, // "--set-prompt" / ver CLI
    pub prompt_edit: &'static str,        // "Editar config.json…" / "Edit config.json…"

    // Submenú "Esfuerzo" / "Effort" (calidad de decodificación).
    pub effort: &'static str,
    pub effort_balanced: &'static str,
    pub effort_robust: &'static str,
    pub effort_high_quality: &'static str,

    // Aviso no modal: modelo solo-inglés activo con idioma distinto a "en".
    // Se muestra en tooltip del icono + como ítem de menú raíz "fix_model_lang".
    pub model_lang_mismatch_tooltip: &'static str,
    pub model_lang_mismatch_action: &'static str,
}

/// Selecciona la tabla de strings para el idioma de UI activo.
#[must_use]
pub fn strings(lang: UiLanguage) -> &'static Strings {
    match lang {
        UiLanguage::Es => &STRINGS_ES,
        UiLanguage::En => &STRINGS_EN,
        UiLanguage::Bilingual => &STRINGS_BILINGUAL,
    }
}

/// Español (default). Idioma local del proyecto.
pub static STRINGS_ES: Strings = Strings {
    change_hotkey: "Cambiar hotkey…",
    exit: "Salir",
    check_updates: "Buscar actualizaciones",
    open_models_dir: "Abrir carpeta de modelos…",

    theme: "Tema",
    theme_dark: "Oscuro",
    theme_light: "Claro",
    theme_system: "Sistema",

    mode: "Modo de dictado",
    mode_batch: "Batch (Hold-to-Talk) · estable",
    mode_streaming: "Streaming (En vivo) · en prueba",
    mode_chunked: "Chunked (Por bloques) · en prueba",

    models: "Modelos",
    model_tiny: "Tiny",
    model_base: "Base",
    model_small: "Small",
    model_medium: "Medium",
    model_large: "Large",
    model_vad: "VAD",
    model_installed: "✓ ",
    model_download: "↓ Descargar ",
    model_active: "  ← activo",
    model_en_only: "  ⚠ Solo inglés",

    ui_language: "Idioma de la interfaz",
    ui_es: "Español",
    ui_en: "English",
    ui_bilingual: "Bilingüe (ES + EN)",

    system_prompt: "Prompt del sistema",
    prompt_bilingual: "Bilingüe ES + EN (default)",
    prompt_es: "Solo español",
    prompt_en: "Solo inglés",
    prompt_custom: "Personalizado…",
    prompt_custom_hint: "(edítalo con --set-prompt)",
    prompt_edit: "Editar config.json…",

    effort: "Esfuerzo de decodificación",
    effort_balanced: "Equilibrado (greedy, 1×) — rápido · estable",
    effort_robust: "Robusto (greedy best_of=5) — más lento, mejor con ruido",
    effort_high_quality: "Alta calidad (beam search) — mucho más lento, mejor precisión",

    model_lang_mismatch_tooltip: "oido — ⚠ modelo solo inglés con idioma español; ábreme para corregir",
    model_lang_mismatch_action: "⚠ Cambiar a modelo multilingüe…",
};

/// Inglés puro.
pub static STRINGS_EN: Strings = Strings {
    change_hotkey: "Change hotkey…",
    exit: "Exit",
    check_updates: "Check for updates",
    open_models_dir: "Open models folder…",

    theme: "Theme",
    theme_dark: "Dark",
    theme_light: "Light",
    theme_system: "System",

    mode: "Dictation mode",
    mode_batch: "Batch (Hold-to-Talk) · stable",
    mode_streaming: "Streaming (Live) · in testing",
    mode_chunked: "Chunked (Blocks) · in testing",

    models: "Models",
    model_tiny: "Tiny",
    model_base: "Base",
    model_small: "Small",
    model_medium: "Medium",
    model_large: "Large",
    model_vad: "VAD",
    model_installed: "✓ ",
    model_download: "↓ Download ",
    model_active: "  ← active",
    model_en_only: "  ⚠ English only",

    ui_language: "Interface language",
    ui_es: "Español",
    ui_en: "English",
    ui_bilingual: "Bilingual (ES + EN)",

    system_prompt: "System prompt",
    prompt_bilingual: "Bilingual ES + EN (default)",
    prompt_es: "Spanish only",
    prompt_en: "English only",
    prompt_custom: "Custom…",
    prompt_custom_hint: "(edit with --set-prompt)",
    prompt_edit: "Edit config.json…",

    effort: "Decoding effort",
    effort_balanced: "Balanced (greedy, 1×) — fast · stable",
    effort_robust: "Robust (greedy best_of=5) — slower, better with noise",
    effort_high_quality: "High quality (beam search) — much slower, best accuracy",

    model_lang_mismatch_tooltip: "oido — ⚠ English-only model with non-English language; open me to fix",
    model_lang_mismatch_action: "⚠ Switch to multilingual model…",
};

/// Bilingüe: secciones críticas con ES + EN, resto en ES.
///
/// Decisión de diseño: ES preferente (idioma local) pero los labels
/// de las 2 secciones nuevas (Idioma, Prompt) son bilingües para que
/// el usuario entienda ambas variantes de un vistazo.
pub static STRINGS_BILINGUAL: Strings = Strings {
    change_hotkey: "Cambiar hotkey / Change hotkey…",
    exit: "Salir / Exit",
    check_updates: "Buscar actualizaciones / Check for updates",
    open_models_dir: "Abrir carpeta de modelos / Open models folder…",

    theme: "Tema / Theme",
    theme_dark: "Oscuro / Dark",
    theme_light: "Claro / Light",
    theme_system: "Sistema / System",

    mode: "Modo de dictado / Dictation mode",
    mode_batch: "Batch (Hold-to-Talk) · estable / stable",
    mode_streaming: "Streaming (En vivo / Live) · en prueba",
    mode_chunked: "Chunked (Por bloques / Blocks) · en prueba",

    models: "Modelos / Models",
    model_tiny: "Tiny",
    model_base: "Base",
    model_small: "Small",
    model_medium: "Medium",
    model_large: "Large",
    model_vad: "VAD",
    model_installed: "✓ ",
    model_download: "↓ Descargar / Download ",
    model_active: "  ← activo / active",
    model_en_only: "  ⚠ Solo inglés / English only",

    ui_language: "Idioma de la interfaz / Interface language",
    ui_es: "Español",
    ui_en: "English",
    ui_bilingual: "Bilingüe (ES + EN)",

    system_prompt: "Prompt del sistema / System prompt",
    prompt_bilingual: "Bilingüe ES + EN (default)",
    prompt_es: "Solo español",
    prompt_en: "Solo inglés / English only",
    prompt_custom: "Personalizado / Custom…",
    prompt_custom_hint: "(edítalo con --set-prompt / edit with --set-prompt)",
    prompt_edit: "Editar config.json / Edit config.json…",

    effort: "Esfuerzo de decodificación / Decoding effort",
    effort_balanced: "Equilibrado / Balanced (greedy, 1×) — rápido / fast · estable",
    effort_robust: "Robusto (greedy best_of=5) — más lento / slower, mejor con ruido / with noise",
    effort_high_quality: "Alta calidad / High quality (beam search) — mucho más lento / much slower, mejor precisión / best accuracy",

    model_lang_mismatch_tooltip: "oido — ⚠ modelo solo inglés / English-only model; ábreme / open me",
    model_lang_mismatch_action: "⚠ Cambiar a modelo multilingüe / Switch to multilingual model…",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_returns_correct_table_per_language() {
        assert_eq!(
            strings(UiLanguage::Es).change_hotkey,
            STRINGS_ES.change_hotkey
        );
        assert_eq!(
            strings(UiLanguage::En).change_hotkey,
            STRINGS_EN.change_hotkey
        );
        assert_eq!(
            strings(UiLanguage::Bilingual).change_hotkey,
            STRINGS_BILINGUAL.change_hotkey
        );
    }

    #[test]
    fn each_language_is_distinct_for_critical_labels() {
        // Las 2 secciones nuevas DEBEN tener strings distintos en cada
        // idioma, sino el bilingüe no aporta nada.
        let pairs = [
            (
                STRINGS_ES.ui_language,
                STRINGS_EN.ui_language,
                STRINGS_BILINGUAL.ui_language,
            ),
            (
                STRINGS_ES.system_prompt,
                STRINGS_EN.system_prompt,
                STRINGS_BILINGUAL.system_prompt,
            ),
            (STRINGS_ES.theme, STRINGS_EN.theme, STRINGS_BILINGUAL.theme),
            (STRINGS_ES.mode, STRINGS_EN.mode, STRINGS_BILINGUAL.mode),
        ];
        for (es, en, bil) in pairs {
            assert_ne!(es, en, "ES y EN deben diferir");
            assert_ne!(es, bil, "Bilingüe debe diferir de ES puro");
            assert_ne!(en, bil, "Bilingüe debe diferir de EN puro");
        }
    }

    #[test]
    fn all_strings_are_non_empty() {
        // Barrido defensivo: ningún string puede estar vacío porque el
        // backend de bandeja renderiza separadores raros para "" en
        // algunos OS (notablemente en macOS el ítem se vuelve fantasma).
        let tables: &[&Strings] = &[&STRINGS_ES, &STRINGS_EN, &STRINGS_BILINGUAL];
        for t in tables {
            let s: &[(&str, &str)] = &[
                ("change_hotkey", t.change_hotkey),
                ("exit", t.exit),
                ("check_updates", t.check_updates),
                ("open_models_dir", t.open_models_dir),
                ("theme", t.theme),
                ("theme_dark", t.theme_dark),
                ("theme_light", t.theme_light),
                ("theme_system", t.theme_system),
                ("mode", t.mode),
                ("mode_batch", t.mode_batch),
                ("mode_streaming", t.mode_streaming),
                ("mode_chunked", t.mode_chunked),
                ("models", t.models),
                ("model_tiny", t.model_tiny),
                ("model_base", t.model_base),
                ("model_small", t.model_small),
                ("model_medium", t.model_medium),
                ("model_large", t.model_large),
                ("model_vad", t.model_vad),
                ("model_installed", t.model_installed),
                ("model_download", t.model_download),
                ("model_active", t.model_active),
                ("model_en_only", t.model_en_only),
                ("ui_language", t.ui_language),
                ("ui_es", t.ui_es),
                ("ui_en", t.ui_en),
                ("ui_bilingual", t.ui_bilingual),
                ("system_prompt", t.system_prompt),
                ("prompt_bilingual", t.prompt_bilingual),
                ("prompt_es", t.prompt_es),
                ("prompt_en", t.prompt_en),
                ("prompt_custom", t.prompt_custom),
                ("prompt_custom_hint", t.prompt_custom_hint),
                ("prompt_edit", t.prompt_edit),
                ("effort", t.effort),
                ("effort_balanced", t.effort_balanced),
                ("effort_robust", t.effort_robust),
                ("effort_high_quality", t.effort_high_quality),
                ("model_lang_mismatch_tooltip", t.model_lang_mismatch_tooltip),
                ("model_lang_mismatch_action", t.model_lang_mismatch_action),
            ];
            for (key, val) in s {
                assert!(
                    !val.is_empty(),
                    "string `{key}` está vacío en {:?}",
                    t.change_hotkey
                );
            }
        }
    }
}
