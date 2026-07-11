//! Diagnóstico del sistema (`--check`).
//! Formatea la configuración activa en una tabla Unicode y la imprime.

use std::path::Path;

use anyhow::Result;
use oido_config::{Config, Theme};

use crate::resolve_models_dir;

/// Formatea la configuración activa en una tabla elegante usando caracteres Unicode.
pub(crate) fn format_config_table(config: &Config, models_dir: &Path) -> String {
    let mut s = String::new();
    s.push_str("┌──────────────────────────────────────────────────────────┐\n");
    s.push_str("│                  oido 2.0 Configuración                  │\n");
    s.push_str("├──────────────────────────┬───────────────────────────────┤\n");
    s.push_str(&format!(
        "│ Tecla de Activación      │ {:<29} │\n",
        config.hotkey
    ));
    s.push_str(&format!(
        "│ Modelo Whisper           │ {:<29} │\n",
        config.model
    ));
    s.push_str(&format!(
        "│ Idioma de UI             │ {:<29} │\n",
        config.language_ui
    ));
    s.push_str(&format!(
        "│ Usar GPU                 │ {:<29} │\n",
        if config.use_gpu { "Sí" } else { "No" }
    ));
    let threads_str = match config.n_threads {
        Some(t) => t.to_string(),
        None => "Auto".to_string(),
    };
    s.push_str(&format!(
        "│ Hilos Whisper            │ {:<29} │\n",
        threads_str
    ));
    let theme_str = match config.theme {
        Theme::Dark => "dark",
        Theme::Light => "light",
        Theme::System => "system",
    };
    s.push_str(&format!(
        "│ Tema                     │ {:<29} │\n",
        theme_str
    ));
    let ui_lang_str = match config.ui_language {
        oido_config::UiLanguage::Es => "es",
        oido_config::UiLanguage::En => "en",
        oido_config::UiLanguage::Bilingual => "bilingual",
    };
    s.push_str(&format!(
        "│ Idioma del Menú          │ {:<29} │\n",
        ui_lang_str
    ));
    let prompt_str = match config.prompt_preset {
        oido_config::PromptPreset::BilingualEsEn => "Bilingüe ES+EN",
        oido_config::PromptPreset::SpanishOnly => "Solo español",
        oido_config::PromptPreset::EnglishOnly => "Solo inglés",
        oido_config::PromptPreset::Custom => "Personalizado",
    };
    s.push_str(&format!(
        "│ System Prompt            │ {:<29} │\n",
        prompt_str
    ));
    s.push_str("├──────────────────────────┼───────────────────────────────┤\n");
    let models_dir_str = models_dir.to_string_lossy();
    let truncated_dir = if models_dir_str.len() > 29 {
        format!("...{}", &models_dir_str[models_dir_str.len() - 26..])
    } else {
        models_dir_str.to_string()
    };
    s.push_str(&format!(
        "│ Directorio de Modelos    │ {:<29} │\n",
        truncated_dir
    ));
    s.push_str("└──────────────────────────┴───────────────────────────────┘");
    s
}

/// Ejecuta el diagnóstico del sistema para el flag --check.
pub(crate) fn run_check(config: &Config) -> Result<()> {
    let models_dir = resolve_models_dir();
    let table = format_config_table(config, &models_dir);
    println!("{}", table);

    println!("\n--- Diagnóstico de Sistema ---");
    println!("Versión oido: {}", env!("CARGO_PKG_VERSION"));

    let cfg_file = oido_config::config_file();
    println!(
        "Archivo de Configuración: {} ({})",
        cfg_file.display(),
        if cfg_file.exists() {
            "Presente"
        } else {
            "No encontrado, usando defaults"
        }
    );

    println!("Directorio de Modelos: {}", models_dir.display());
    let model_path = models_dir.join(&config.model);
    println!(
        "Modelo Whisper: {} ({})",
        model_path.display(),
        if model_path.exists() {
            "Descargado"
        } else {
            "Falta / No encontrado"
        }
    );

    let has_gpu_support = oido_config::default_use_gpu();
    println!(
        "GPU Compilada: {}",
        if has_gpu_support { "Sí" } else { "No" }
    );

    Ok(())
}
