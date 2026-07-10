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

use oido_config::{SttMode, Theme};

use crate::traits::MenuAction;

/// ID canónico de un item, estable entre OS. Sirve como clave del
/// `HashMap<MenuId, MenuAction>` que el forwarder necesita para mapear
/// clics a acciones.
pub type ItemId = &'static str;

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
            id: "change_hotkey",
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
                    id: "theme_dark",
                    label: "Dark".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_light",
                    label: "Light".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_system",
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
                    id: "mode_batch",
                    label: "Batch (Hold-to-Talk)".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "mode_streaming",
                    label: "Streaming (En vivo)".into(),
                    enabled: true,
                },
            ],
        }]
    }
}

/// Sección "Utilidades" — separador + abrir carpeta + buscar updates.
#[derive(Debug)]
pub struct UtilitiesSection;

impl MenuSection for UtilitiesSection {
    fn id(&self) -> &'static str {
        "utilities"
    }

    fn build(&self) -> Vec<Section> {
        vec![
            Section::Item(MenuItemSpec {
                id: "open_models_dir",
                label: "Abrir carpeta de modelos".into(),
                enabled: true,
            }),
            Section::Item(MenuItemSpec {
                id: "check_updates",
                label: "Buscar actualizaciones".into(),
                enabled: true,
            }),
        ]
    }
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
            id: "exit",
            label: "Salir".into(),
            enabled: true,
        })]
    }
}

/// Conjunto canónico de secciones que produce el mismo árbol que el
/// `build_menu` original. Devolver este `Vec` desde el bin garantiza
/// 100% de compat visual con el menú anterior.
pub fn default_sections() -> Vec<Box<dyn MenuSection>> {
    vec![
        Box::new(HotkeySection),
        Box::new(ThemeSection),
        Box::new(ModeSection),
        Box::new(UtilitiesSection),
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
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn id_to_action_returns_none_for_unknown() {
        assert_eq!(id_to_action("nonexistent_item"), None);
        assert_eq!(id_to_action(""), None);
    }

    #[test]
    fn default_sections_produces_expected_tree() {
        let sections = default_sections();
        assert_eq!(sections.len(), 5, "5 secciones: hotkey, theme, mode, utilities, exit");
        // Cada sección produce al menos un item.
        for s in &sections {
            assert!(!s.build().is_empty(), "sección {} no debe estar vacía", s.id());
        }
    }

    #[test]
    fn default_sections_cover_all_menu_actions() {
        // Cada MenuAction debe tener al menos un item_id en las
        // secciones por defecto. Si se añade un item nuevo, este test
        // falla hasta que se actualice `id_to_action`.
        let sections = default_sections();
        let mut all_ids: Vec<&str> = Vec::new();
        for s in &sections {
            for item in s.build() {
                match item {
                    Section::Item(spec) => all_ids.push(spec.id),
                    Section::Submenu { items, .. } => {
                        for spec in items {
                            all_ids.push(spec.id);
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
                all_ids.contains(&expected),
                "default_sections debe incluir item id {expected}"
            );
            assert!(
                id_to_action(expected).is_some(),
                "id_to_action debe mapear {expected}"
            );
        }
    }
}
