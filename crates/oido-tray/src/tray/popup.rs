//! Adapter safe-Rust entre el árbol declarativo `MenuSection` y el
//! renderer Win32 GDI (`popup_window`).
//!
//! Convierte cada `MenuSection::build()` (que devuelve `Vec<Section>`
//! con `Item`/`Submenu`/`Separator`) a un `Vec<PopupRow>` plano que
//! `popup_window::run_popup` puede pintar y devolver como índice
//! clickeado.
//!
//! Define también la `PopupPalette` (color scheme para el renderer)
//! parametrizable por `Theme` (Dark / Light / System).

use oido_config::Theme;

use crate::tray::sections::{ItemId, MenuItemSpec, Section};

/// Una fila del popup. La vida útil es exactamente la del message
/// loop modal: se construye al abrir y se destruye al cerrar.
///
/// `PopRow` es plano por sí mismo — los submenús viven como
/// `RowKind::Submenu { child: Vec<PopupRow> }`. El renderer los abre
/// en su propio popup al lado (un nivel de anidación, suficiente para
/// nuestro árbol actual).
#[derive(Debug, Clone)]
pub struct PopupRow {
    pub kind: RowKind,
}

#[derive(Debug, Clone)]
pub enum RowKind {
    /// Botón seleccionable. `id` mapea a `MenuAction` vía
    /// `crate::tray::sections::id_to_action`.
    Action {
        id: ItemId,
        label: String,
        active: bool,
    },
    /// Sub-popup al lado. `child` es el árbol interno (plano en su
    /// propio message loop modal).
    Submenu {
        id: ItemId,
        label: String,
        child: Vec<PopupRow>,
    },
    /// Línea horizontal de separación visual.
    Separator,
}

/// Paleta de colores del popup (sin alpha — GDI no la usa).
#[derive(Debug, Clone, Copy)]
pub struct PopupPalette {
    /// Fondo de toda la ventana.
    pub bg: (u8, u8, u8),
    /// Fondo de la fila hovereada.
    pub hover: (u8, u8, u8),
    /// Color de la línea de separación.
    pub separator: (u8, u8, u8),
    /// Texto principal.
    pub text: (u8, u8, u8),
    /// Acento (futuro: marca ✓, foco de teclado).
    pub accent: (u8, u8, u8),
}

impl PopupPalette {
    /// Construye la paleta a partir del `Theme`. `Theme::System` se
    /// resuelve con `dark_light::detect` (la crate ya está en el
    /// workspace para `icon.rs`).
    #[must_use]
    pub fn from_theme(theme: Theme) -> Self {
        let resolved = match theme {
            Theme::Dark => Theme::Dark,
            Theme::Light => Theme::Light,
            Theme::System => match dark_light::detect() {
                dark_light::Mode::Light => Theme::Light,
                _ => Theme::Dark,
            },
        };
        match resolved {
            Theme::Dark => Self {
                bg: (10, 10, 10),        // #0A0A0A
                hover: (31, 31, 31),     // #1F1F1F
                separator: (42, 42, 42), // #2A2A2A
                text: (236, 236, 236),   // #ECECEC
                accent: (59, 130, 246),  // #3B82F6
            },
            Theme::Light => Self {
                bg: (255, 255, 255),        // #FFFFFF
                hover: (234, 234, 234),     // #EAEAEA
                separator: (208, 208, 208), // #D0D0D0
                text: (26, 26, 26),         // #1A1A1A
                accent: (47, 111, 237),     // #2F6FED
            },
            Theme::System => unreachable!("resuelto arriba"),
        }
    }
}

/// Aplana un árbol de `Section` (proveniente de
/// `MenuSection::build()`) en un `Vec<PopupRow>` que el renderer
/// puede pintar.
///
/// ## Reglas
/// - `Section::Item` → `RowKind::Action` con la `id`, `label` y
///   el flag `enabled` (deshabilitados se renderizan con color
///   atenuado en una iteración futura; por ahora todos habilitados).
/// - `Section::Submenu` → `RowKind::Submenu` con sus `items`
///   convertidos recursivamente como `child` (1 nivel de anidación).
/// - `Section::Separator` → `RowKind::Separator`.
///
/// ## Submenús
/// El árbol actual permite **una** jerarquía: los items dentro de
/// un Submenu son siempre `Section::Item` o `Separator`. No hay
/// sub-submenús. Esto es coherente con el menú anterior (no usamos
/// submenús anidados en producción).
pub fn flatten_sections(sections: &[Section]) -> Vec<PopupRow> {
    let mut out = Vec::new();
    for s in sections {
        push_section(&mut out, s);
    }
    out
}

fn push_section(out: &mut Vec<PopupRow>, section: &Section) {
    match section {
        Section::Item(item) => {
            out.push(PopupRow {
                kind: RowKind::Action {
                    id: item.id.clone(),
                    label: item.label.clone(),
                    active: !item.enabled,
                },
            });
        }
        Section::Submenu { label, items } => {
            // El id del submenu en sí mismo (no es clickeable directo,
            // sino abre el sub-popup). Usamos el primer item id con
            // el prefijo `submenu:` para que `id_to_action` lo ignore.
            let id: ItemId = format!(
                "submenu:{}",
                items.first().map(|m| m.id.as_str()).unwrap_or("")
            );
            let child = items
                .iter()
                .map(|item| PopupRow {
                    kind: RowKind::Action {
                        id: item.id.clone(),
                        label: item.label.clone(),
                        active: !item.enabled,
                    },
                })
                .collect::<Vec<_>>();
            out.push(PopupRow {
                kind: RowKind::Submenu {
                    id,
                    label: label.clone(),
                    child,
                },
            });
        }
        Section::Separator => out.push(PopupRow {
            kind: RowKind::Separator,
        }),
    }
}

/// Helper de conversión desde `MenuItemSpec` directo (usado por tests
/// que no montan `Section`).
#[allow(dead_code)]
pub fn rows_from_items(items: &[MenuItemSpec]) -> Vec<PopupRow> {
    items
        .iter()
        .map(|item| PopupRow {
            kind: RowKind::Action {
                id: item.id.clone(),
                label: item.label.clone(),
                active: !item.enabled,
            },
        })
        .collect()
}

#[cfg(target_os = "windows")]
pub use crate::tray::popup_window::run_popup;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_dark_is_black_bg_white_text() {
        let p = PopupPalette::from_theme(Theme::Dark);
        assert_eq!(p.bg, (10, 10, 10));
        assert!(p.text.0 > 200 && p.text.1 > 200 && p.text.2 > 200);
    }

    #[test]
    fn palette_light_inverts() {
        let dark = PopupPalette::from_theme(Theme::Dark);
        let light = PopupPalette::from_theme(Theme::Light);
        // Dark_bg es muy oscuro; Light_bg es muy claro.
        assert!(dark.bg.0 < 50);
        assert!(light.bg.0 > 200);
        // Y los textos se invierten.
        assert!(dark.text.0 > light.text.0);
    }

    #[test]
    fn flatten_sections_drops_empty_minimal() {
        let out = flatten_sections(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn flatten_sections_converts_item() {
        let s = vec![Section::Item(MenuItemSpec {
            id: "x".into(),
            label: "Cambiar tecla".into(),
            enabled: true,
        })];
        let out = flatten_sections(&s);
        assert_eq!(out.len(), 1);
        match &out[0].kind {
            RowKind::Action { id, label, active } => {
                assert_eq!(id, "x");
                assert_eq!(label, "Cambiar tecla");
                assert!(!*active, "enabled=true → active=false");
            }
            other => panic!("esperaba Action, obtuve {:?}", other),
        }
    }

    #[test]
    fn flatten_sections_passes_separator() {
        let s = vec![Section::Separator];
        let out = flatten_sections(&s);
        assert!(matches!(out[0].kind, RowKind::Separator));
    }

    #[test]
    fn flatten_sections_nests_submenu() {
        let s = vec![Section::Submenu {
            label: "Tema".into(),
            items: vec![
                MenuItemSpec {
                    id: "theme_dark".into(),
                    label: "Oscuro".into(),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_light".into(),
                    label: "Claro".into(),
                    enabled: true,
                },
            ],
        }];
        let out = flatten_sections(&s);
        assert_eq!(out.len(), 1);
        match &out[0].kind {
            RowKind::Submenu { label, child, .. } => {
                assert_eq!(label, "Tema");
                assert_eq!(child.len(), 2);
                assert!(matches!(child[0].kind, RowKind::Action { .. }));
            }
            other => panic!("esperaba Submenu, obtuve {:?}", other),
        }
    }

    #[test]
    fn rows_from_items_works() {
        let items = vec![MenuItemSpec {
            id: "i".into(),
            label: "L".into(),
            enabled: true,
        }];
        let rows = rows_from_items(&items);
        assert_eq!(rows.len(), 1);
    }
}
