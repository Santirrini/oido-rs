//! Trait `MenuSection` — abstracción declarativa sobre el menú nativo
//! de bandeja. Cada sección del menú (hotkey, tema, modo, utilidades)
//! implementa este trait y se registra en `PlatformTray::with_sections`.
//!
//! ## Por qué este trait
//!
//! Antes: `tray.rs::build_menu` definía el árbol del menú en un solo
//! bloque de ~80 líneas, y `spawn_event_forwarder` mantenía un map
//! paralelo `MenuId → MenuAction`. Añadir un item requería tocar 3-4
//! lugares distintos.
//!
//! Ahora: cada sección describe su contenido de forma agnóstica al
//! backend (`MenuItemSpec`, sin `tray_icon::MenuId` directo). El
//! backend (Windows / macOS) traduce esa descripción al árbol nativo
//! en una sola pasada, y rellena automáticamente el
//! `HashMap<MenuId, MenuAction>` para el forwarder.
//!
//! ## Compatibilidad
//!
//! El árbol visible para el usuario es idéntico al anterior. El enum
//! `MenuAction` no se toca (regla del plan: no romper tests e2e).
//! `PlatformTray::new()` sigue existiendo (construye las secciones
//! por defecto). `PlatformTray::with_sections(s)` es la API nueva.

use std::path::PathBuf;

use oido_config::{SttMode, Theme};
use oido_models::{human_size, Language, ModelEntry, ModelFamily};

use crate::traits::MenuAction;

/// ID canónico de un item, estable entre OS. Sirve como clave del
/// `HashMap<MenuId, MenuAction>` que el forwarder necesita para mapear
/// clics a acciones.
///
/// Es `String` (no `&'static str`) para soportar ids dinámicos como
/// `"model:ggml-base.bin"` que se generan a partir del catálogo de
/// modelos. Los ids estáticos (cambiar hotkey, abrir carpeta, etc.) se
/// pasan como `String::from("…")`.
pub type ItemId = String;

/// Descripción agnóstica de un item de menú.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuItemSpec {
    pub id: ItemId,
    pub label: String,
    pub enabled: bool,
}

/// Descripción de un sub-árbol de menú.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Section {
    /// Item simple en el nivel actual.
    Item(MenuItemSpec),
    /// Submenú con título e items anidados.
    Submenu {
        label: String,
        items: Vec<MenuItemSpec>,
    },
    /// Separador visual. No tiene items.
    Separator,
}

/// Sección declarativa. Implementar este trait permite que el backend
/// de bandeja la convierta a items nativos (tray-icon/ksni) sin que
/// la sección conozca el backend.
pub trait MenuSection: Send + Sync + 'static {
    /// Identificador único (útil para diagnóstico/tests). No se usa
    /// para mapear clics; ese mapeo va por `MenuItemSpec::id`.
    fn id(&self) -> &'static str;

    /// Construye la lista de items / submenús que componen esta
    /// sección. El backend se encarga de añadir separadores antes y
    /// después según las convenciones del OS.
    fn build(&self) -> Vec<Section>;
}

/// Sección "Cambiar hotkey". Item simple en el nivel raíz.
#[derive(Debug)]
pub struct HotkeySection;

impl MenuSection for HotkeySection {
    fn id(&self) -> &'static str {
        "hotkey"
    }

    fn build(&self) -> Vec<Section> {
        vec![Section::Item(MenuItemSpec {
            id: "change_hotkey".into(),
            label: "Cambiar hotkey…".into(),
            enabled: true,
        })]
    }
}

/// Sección "Tema" — submenú con 3 opciones de tema.
#[derive(Debug)]
pub struct ThemeSection;

impl MenuSection for ThemeSection {
    fn id(&self) -> &'static str {
        "theme"
    }

    fn build(&self) -> Vec<Section> {
        vec![Section::Submenu {
            label: "Tema".into(),
            items: vec![
                MenuItemSpec {
                    id: "theme_dark".into(),
                    label: "Dark".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_light".into(),
                    label: "Light".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_system".into(),
                    label: "Sistema".into(),
                    enabled: true,
                },
            ],
        }]
    }
}

/// Sección "Modo de dictado" — submenú con Batch / Streaming.
#[derive(Debug)]
pub struct ModeSection;

impl MenuSection for ModeSection {
    fn id(&self) -> &'static str {
        "mode"
    }

    fn build(&self) -> Vec<Section> {
        vec![Section::Submenu {
            label: "Modo de dictado".into(),
            items: vec![
                MenuItemSpec {
                    id: "mode_batch".into(),
                    label: "Batch (Hold-to-Talk)".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "mode_streaming".into(),
                    label: "Streaming (En vivo)".into(),
                    enabled: true,
                },
            ],
        }]
    }
}

/// Sección "Modelos" — submenú dinámico con cada entry del catálogo de
/// `oido_models`. Cada item se marca con `✓` (instalado) o `↓ Descargar`
/// (no instalado) según el contenido de `models_dir` al construir la
/// sección. El item activo lleva el sufijo `← activo`.
///
/// Esta sección debe reconstruirse vía `Tray::rebuild_menu` después de
/// cada descarga o cambio de modelo activo para refrescar las marcas.
#[derive(Debug)]
pub struct ModelsSection {
    pub models_dir: PathBuf,
    pub active_model: String,
}

impl ModelsSection {
    pub fn new(models_dir: PathBuf, active_model: impl Into<String>) -> Self {
        Self {
            models_dir,
            active_model: active_model.into(),
        }
    }

    /// Renderiza un único item del submenú a partir de un entry del catálogo.
    fn render_item(entry: &ModelEntry, installed: bool, active: bool) -> MenuItemSpec {
        let state = if installed { "✓ " } else { "↓ Descargar " };
        let mut label = format!("{}{} ({})", state, entry.filename, human_size(entry.size_bytes));
        if active {
            label.push_str("  ← activo");
        }
        MenuItemSpec {
            id: item_id(&entry.filename),
            label,
            enabled: true,
        }
    }

    /// Construye el submenú de una familia (Tiny / Base / Small / Vad).
    #[allow(dead_code)]
    fn family_submenu(
        family: ModelFamily,
        installed: &[String],
        active_model: &str,
    ) -> Section {
        let family_label = match family {
            ModelFamily::Tiny => "Tiny",
            ModelFamily::Base => "Base",
            ModelFamily::Small => "Small",
            ModelFamily::Vad => "VAD",
        };
        let mut items: Vec<MenuItemSpec> = oido_models::catalog()
            .iter()
            .filter(|e| e.family == family)
            .map(|entry| {
                let installed = installed.iter().any(|f| f == &entry.filename);
                let active = entry.filename == active_model;
                Self::render_item(entry, installed, active)
            })
            .collect();
        items.sort_by(|a, b| {
            let a_is_en = a.label.contains(".en.bin");
            let b_is_en = b.label.contains(".en.bin");
            b_is_en.cmp(&a_is_en)
        });
        Section::Submenu {
            label: family_label.to_string(),
            items,
        }
    }
}

impl MenuSection for ModelsSection {
    fn id(&self) -> &'static str {
        "models"
    }

    fn build(&self) -> Vec<Section> {
        // Escaneo de disco (tolerante a errores: si el dir no existe, lista vacía).
        let installed =
            oido_models::list_installed(&self.models_dir).unwrap_or_default();

        // Submenú "Modelos" con 4 sub-submenús por familia.
        let mut model_items: Vec<MenuItemSpec> = Vec::new();
        // Renderizamos como Submenu recursivo de Submenu. Como
        // `Section::Submenu` solo soporta items planos (no submenús
        // anidados), hacemos un flatten: agrupamos por familia y emitimos
        // un Submenu por familia dentro del Submenu principal, pero el
        // tipo actual no lo permite.
        //
        // Solución: emitimos los items con el family como prefijo en el
        // label, agrupados visualmente por separadores.
        for family in [
            ModelFamily::Tiny,
            ModelFamily::Base,
            ModelFamily::Small,
            ModelFamily::Vad,
        ] {
            // Marca la familia como sub-encabezado usando un item
            // deshabilitado con el nombre + un separador visual después
            // (vía Section::Separator).
            model_items.push(MenuItemSpec {
                id: String::new(),
                label: format!("── {} ──", family_label(family)),
                enabled: false,
            });
            for entry in oido_models::catalog().iter().filter(|e| e.family == family) {
                let installed = installed.iter().any(|f| f == &entry.filename);
                let active = entry.filename == self.active_model;
                model_items.push(Self::render_item(entry, installed, active));
            }
        }
        let _ = (Language::En, Language::Multi); // silencio unused si cambia el catálogo

        vec![
            Section::Submenu {
                label: "Modelos".to_string(),
                items: model_items,
            },
            Section::Item(MenuItemSpec {
                id: "open_models_dir".into(),
                label: "Abrir carpeta de modelos…".into(),
                enabled: true,
            }),
            Section::Item(MenuItemSpec {
                id: "check_updates".into(),
                label: "Buscar actualizaciones".into(),
                enabled: true,
            }),
        ]
    }
}

fn family_label(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Tiny => "Tiny",
        ModelFamily::Base => "Base",
        ModelFamily::Small => "Small",
        ModelFamily::Vad => "VAD",
    }
}

/// Construye el `id` canónico del item a partir del filename del modelo.
/// Formato: `"model:{filename}"`. Lo parsea `id_to_action` y el bin lo
/// extrae del `MenuAction::ModelItem(String)`.
pub fn item_id(filename: &str) -> ItemId {
    format!("model:{filename}")
}

/// Sección "Salir" — siempre la última; item simple.
#[derive(Debug)]
pub struct ExitSection;

impl MenuSection for ExitSection {
    fn id(&self) -> &'static str {
        "exit"
    }

    fn build(&self) -> Vec<Section> {
        vec![Section::Item(MenuItemSpec {
            id: "exit".into(),
            label: "Salir".into(),
            enabled: true,
        })]
    }
}

/// Conjunto canónico de secciones. La sección "Modelos" es dinámica y
/// recibe `models_dir` + `active_model` para renderizar el estado de
/// cada modelo (instalado/no-instalado + activo).
///
/// El bin debe llamar a `Tray::rebuild_menu` después de cada descarga
/// o cambio de modelo activo para refrescar las marcas.
pub fn default_sections(
    models_dir: PathBuf,
    active_model: impl Into<String>,
) -> Vec<Box<dyn MenuSection>> {
    vec![
        Box::new(HotkeySection),
        Box::new(ThemeSection),
        Box::new(ModeSection),
        Box::new(ModelsSection::new(models_dir, active_model)),
        Box::new(ExitSection),
    ]
}

/// Mapea el `id` canónico de un `MenuItemSpec` a la acción que debe
/// disparar el menú. Centraliza la traducción para que el forwarder
/// no necesite un map gigante con un `if` por item.
pub fn id_to_action(id: &str) -> Option<MenuAction> {
    Some(match id {
        "change_hotkey" => MenuAction::ChangeHotkey,
        "theme_dark" => MenuAction::SetTheme(Theme::Dark),
        "theme_light" => MenuAction::SetTheme(Theme::Light),
        "theme_system" => MenuAction::SetTheme(Theme::System),
        "mode_batch" => MenuAction::SetSttMode(SttMode::Batch),
        "mode_streaming" => MenuAction::SetSttMode(SttMode::Streaming),
        "open_models_dir" => MenuAction::OpenModelsDir,
        "check_updates" => MenuAction::CheckUpdates,
        "exit" => MenuAction::Exit,
        id if id.starts_with("model:") => {
            MenuAction::ModelItem(id["model:".len()..].to_string())
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn id_to_action_covers_known_ids() {
        assert_eq!(id_to_action("change_hotkey"), Some(MenuAction::ChangeHotkey));
        assert_eq!(id_to_action("theme_dark"), Some(MenuAction::SetTheme(Theme::Dark)));
        assert_eq!(id_to_action("theme_light"), Some(MenuAction::SetTheme(Theme::Light)));
        assert_eq!(id_to_action("theme_system"), Some(MenuAction::SetTheme(Theme::System)));
        assert_eq!(
            id_to_action("mode_batch"),
            Some(MenuAction::SetSttMode(SttMode::Batch))
        );
        assert_eq!(
            id_to_action("mode_streaming"),
            Some(MenuAction::SetSttMode(SttMode::Streaming))
        );
        assert_eq!(id_to_action("open_models_dir"), Some(MenuAction::OpenModelsDir));
        assert_eq!(id_to_action("check_updates"), Some(MenuAction::CheckUpdates));
        assert_eq!(id_to_action("exit"), Some(MenuAction::Exit));
        assert_eq!(
            id_to_action("model:ggml-base.bin"),
            Some(MenuAction::ModelItem("ggml-base.bin".to_string()))
        );
    }

    #[test]
    fn id_to_action_returns_none_for_unknown() {
        assert_eq!(id_to_action("nonexistent_item"), None);
        assert_eq!(id_to_action(""), None);
    }

    fn empty_models_dir() -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        // Mantenemos el TempDir vivo en el retorno para que no se
        // elimine antes de que el test termine.
        (dir, path)
    }

    #[test]
    fn default_sections_produces_expected_tree() {
        let (_tmp, models_dir) = empty_models_dir();
        let sections = default_sections(models_dir, "ggml-base.bin");
        assert_eq!(
            sections.len(),
            5,
            "5 secciones: hotkey, theme, mode, models, exit"
        );
        for s in &sections {
            assert!(!s.build().is_empty(), "sección {} no debe estar vacía", s.id());
        }
    }

    #[test]
    fn default_sections_models_submenu_marks_installed_and_active() {
        let (_tmp, models_dir) = empty_models_dir();
        // Simulamos dos modelos ya descargados.
        std::fs::write(models_dir.join("ggml-tiny.bin"), b"x").unwrap();
        std::fs::write(models_dir.join("ggml-base.bin"), b"x").unwrap();

        let sections = default_sections(models_dir, "ggml-base.bin");
        let models_section = sections
            .iter()
            .find(|s| s.id() == "models")
            .expect("debe existir la sección models");

        let mut items: Vec<MenuItemSpec> = Vec::new();
        for sec in models_section.build() {
            match sec {
                Section::Item(spec) => items.push(spec),
                Section::Submenu { items: sub_items, .. } => items.extend(sub_items),
                Section::Separator => {}
            }
        }

        // ggml-tiny.bin está descargado y debe tener ✓.
        let tiny = items
            .iter()
            .find(|i| i.id == "model:ggml-tiny.bin")
            .expect("debe existir item ggml-tiny.bin");
        assert!(tiny.label.starts_with("✓ "), "tiny debe estar marcado como instalado: {}", tiny.label);

        // ggml-small.bin NO está descargado y debe tener ↓ Descargar.
        let small = items
            .iter()
            .find(|i| i.id == "model:ggml-small.bin")
            .expect("debe existir item ggml-small.bin");
        assert!(
            small.label.starts_with("↓ Descargar "),
            "small debe estar marcado como no instalado: {}",
            small.label
        );

        // ggml-base.bin está descargado Y es el activo → debe llevar ← activo.
        let base = items
            .iter()
            .find(|i| i.id == "model:ggml-base.bin")
            .expect("debe existir item ggml-base.bin");
        assert!(base.label.starts_with("✓ "));
        assert!(
            base.label.contains("← activo"),
            "base debe estar marcado como activo: {}",
            base.label
        );
    }

    #[test]
    fn default_sections_cover_all_menu_actions() {
        let (_tmp, models_dir) = empty_models_dir();
        let sections = default_sections(models_dir, "ggml-base.bin");
        let mut all_ids: Vec<String> = Vec::new();
        for s in &sections {
            for item in s.build() {
                match item {
                    Section::Item(spec) => all_ids.push(spec.id),
                    Section::Submenu { items, .. } => {
                        for spec in items {
                            if !spec.id.is_empty() {
                                all_ids.push(spec.id);
                            }
                        }
                    }
                    Section::Separator => {}
                }
            }
        }
        for expected in [
            "change_hotkey",
            "theme_dark",
            "theme_light",
            "theme_system",
            "mode_batch",
            "mode_streaming",
            "open_models_dir",
            "check_updates",
            "exit",
        ] {
            assert!(
                all_ids.iter().any(|id| id == expected),
                "default_sections debe incluir item id {expected}"
            );
            assert!(
                id_to_action(expected).is_some(),
                "id_to_action debe mapear {expected}"
            );
        }
    }
}
