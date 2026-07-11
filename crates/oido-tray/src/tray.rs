//! Implementaciones nativas del icono de bandeja por plataforma.
//!
//! - Windows + macOS: `tray-icon` + `muda` (re-exportado por `tray_icon::menu`).
//! - Linux: `ksni` (StatusNotifierItem / D-Bus).
//!
//! La API pública (`PlatformTray`, `Tray` trait) permanece idéntica; solo
//! el interior cambia de stub a implementación real.
//!
//! ## Modularidad del menú (Fase 3)
//!
//! El árbol del menú se construye a partir de una lista de
//! `MenuSection` (ver `sections.rs`). Cada sección se describe con
//! `MenuItemSpec` agnóstico al backend, y el backend nativo
//! (Windows / macOS) lo traduce a items `tray_icon::menu::*`. El
//! forwarder de eventos usa un `HashMap<MenuId, MenuAction>` que se
//! rellena automáticamente con `id_to_action` desde el `id`
//! canónico del spec.
//!
//! `PlatformTray::new()` usa `default_sections()` (mismo árbol que
//! antes). `PlatformTray::with_sections(...)` permite al bin pasar
//! un set personalizado.

pub mod i18n;
#[cfg(target_os = "windows")]
pub mod popup;
#[cfg(target_os = "windows")]
pub mod popup_window;
pub mod sections;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use oido_config::Theme;
use parking_lot::Mutex;
use tray_icon::menu::MenuId;

use crate::icon;
use crate::traits::{MenuAction, Tray, TrayError, TrayState};

use self::sections::{default_sections, id_to_action, BuildContext, MenuSection, Section};

// ---------------------------------------------------------------------------
// PlatformTray — wrapper que selecciona la impl según el OS en compilación
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub struct PlatformTray(LinuxTray);
#[cfg(target_os = "macos")]
pub struct PlatformTray(MacTray);
#[cfg(target_os = "windows")]
pub struct PlatformTray(WindowsTray);

impl PlatformTray {
    /// Construye el tray con un set mínimo de defaults. El bin debe
    /// llamar a `rebuild_menu` con un `BuildContext` completo tras el
    /// startup, para que el árbol refleje el `Config` real (tema,
    /// modo STT, prompt selection, etc.).
    pub fn new(
        models_dir: PathBuf,
        active_model: String,
        ui_language: oido_config::UiLanguage,
        prompt_preset: oido_config::PromptPreset,
    ) -> Result<Self, TrayError> {
        let mut ctx = BuildContext::initial(models_dir, active_model);
        ctx.ui_language = ui_language;
        ctx.prompt_preset = prompt_preset;
        Ok(Self(Inner::new(default_sections(&ctx))?))
    }

    /// Construye el tray con un set de secciones declarativo. El
    /// árbol visible es el descrito por las secciones; los items
    /// siguen el mismo mapeo `id → MenuAction` que el set por
    /// defecto.
    pub fn with_sections(sections: Vec<Box<dyn MenuSection>>) -> Result<Self, TrayError> {
        Ok(Self(Inner::new(sections)?))
    }
}

#[cfg(target_os = "linux")]
type Inner = LinuxTray;
#[cfg(target_os = "macos")]
type Inner = MacTray;
#[cfg(target_os = "windows")]
type Inner = WindowsTray;

impl std::fmt::Debug for PlatformTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PlatformTray").finish()
    }
}

impl Tray for PlatformTray {
    fn show(&mut self) -> Result<(), TrayError> {
        self.0.show()
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), TrayError> {
        self.0.set_state(state, theme)
    }
    fn hide(&mut self) -> Result<(), TrayError> {
        self.0.hide()
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.0.take_menu_events()
    }
    fn rebuild_menu(&mut self, sections: Vec<Box<dyn MenuSection>>) -> Result<(), TrayError> {
        self.0.rebuild_menu(sections)
    }
}

// ---------------------------------------------------------------------------
// Linux — ksni (StatusNotifierItem / D-Bus)
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub struct LinuxTray {
    sender: crossbeam_channel::Sender<MenuAction>,
    receiver: Option<crossbeam_channel::Receiver<MenuAction>>,
    current_state: TrayState,
    current_theme: Theme,
    /// Secciones declarativas registradas (sin uso en la impl stub,
    /// pero mantenidas para que el shape del struct sea simétrico al
    /// de Windows/macOS y el API público sea uniforme entre OS).
    #[allow(dead_code)]
    sections: Vec<Box<dyn MenuSection>>,
}

#[cfg(target_os = "linux")]
impl std::fmt::Debug for LinuxTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinuxTray")
            .field("state", &self.current_state)
            .finish()
    }
}

#[cfg(target_os = "linux")]
impl LinuxTray {
    pub fn new(sections: Vec<Box<dyn MenuSection>>) -> Result<Self, TrayError> {
        let (sender, receiver) = crossbeam_channel::bounded(16);
        tracing::info!(
            count = sections.len(),
            "tray Linux (stub): secciones registradas (no se renderizan aún)"
        );
        Ok(Self {
            sender,
            receiver: Some(receiver),
            current_state: TrayState::Idle,
            current_theme: Theme::System,
            sections,
        })
    }
}

#[cfg(target_os = "linux")]
impl Tray for LinuxTray {
    fn show(&mut self) -> Result<(), TrayError> {
        tracing::info!("tray Linux (ksni): show");
        Ok(())
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), TrayError> {
        self.current_state = state;
        self.current_theme = theme;
        tracing::info!(?state, "tray Linux state");
        Ok(())
    }
    fn hide(&mut self) -> Result<(), TrayError> {
        Ok(())
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.receiver.take()
    }
}

// ---------------------------------------------------------------------------
// macOS — tray-icon + menu
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub struct MacTray {
    #[allow(dead_code)]
    icon: tray_icon::TrayIcon,
    receiver: Option<crossbeam_channel::Receiver<MenuAction>>,
    /// Compartido con el forwarder de eventos. Se reemplaza atómicamente
    /// en cada `rebuild_menu` para que los nuevos items del menú sean
    /// visibles inmediatamente (mismo patrón que `WindowsTray`).
    id_map: SharedIdMap,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for MacTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacTray").finish()
    }
}

#[cfg(target_os = "macos")]
impl MacTray {
    pub fn new(sections: Vec<Box<dyn MenuSection>>) -> Result<Self, TrayError> {
        let (sender, receiver) = crossbeam_channel::bounded::<MenuAction>(16);
        let rgba = icon::render_state(TrayState::Idle, Theme::System);
        let tray_icon_img = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        let (menu, id_map) = build_menu_from_sections(&sections);
        let icon = tray_icon::TrayIconBuilder::new()
            .with_icon(tray_icon_img)
            .with_tooltip("oido — idle")
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| TrayError::Tray(e.to_string()))?;

        // El forwarder se queda con su clon; guardamos el nuestro para
        // poder rotarlo en `rebuild_menu`. Si no lo retuviéramos, el
        // primer rebuild dejaría al forwarder leyendo un mapa viejo
        // (y los items reconstruidos emitirían MenuAction desconocidas).
        let id_map_for_self = Arc::clone(&id_map);
        spawn_event_forwarder(id_map, sender);

        Ok(Self {
            icon,
            receiver: Some(receiver),
            id_map: id_map_for_self,
        })
    }
}

#[cfg(target_os = "macos")]
impl Tray for MacTray {
    fn show(&mut self) -> Result<(), TrayError> {
        Ok(())
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), TrayError> {
        let rgba = icon::render_state(state, theme);
        let new_icon = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        self.icon
            .set_icon(Some(new_icon))
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        let tooltip = state_tooltip(state);
        self.icon
            .set_tooltip(Some(tooltip))
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        Ok(())
    }
    fn hide(&mut self) -> Result<(), TrayError> {
        Ok(())
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.receiver.take()
    }
    fn rebuild_menu(&mut self, sections: Vec<Box<dyn MenuSection>>) -> Result<(), TrayError> {
        let (menu, new_id_map) = build_menu_from_sections(&sections);
        // En tray-icon 0.24, `set_menu` retorna `Result` solo en macOS;
        // en Windows no. Como aquí estamos en la rama macOS, propagamos
        // el error tal cual. (Windows tiene su propia rama cfg dentro de
        // su impl homólogo para absorber la diferencia.)
        if let Err(e) = self.icon.set_menu(Some(Box::new(menu))) {
            return Err(TrayError::Tray(format!("set_menu: {e}")));
        }
        // Reemplaza el id_map atómicamente; el forwarder leerá la
        // versión nueva a partir del próximo evento.
        let replacement = Arc::try_unwrap(new_id_map)
            .map(|m| m.into_inner())
            .unwrap_or_else(|arc| MenuIdMap {
                id_to_spec_id: arc.lock().id_to_spec_id.clone(),
            });
        *self.id_map.lock() = replacement;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Windows — tray-icon + menu
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
pub struct WindowsTray {
    icon: tray_icon::TrayIcon,
    receiver: Option<crossbeam_channel::Receiver<MenuAction>>,
    /// Compartido con el forwarder de eventos. Se reemplaza atómicamente
    /// en cada `rebuild_menu` para que los nuevos items del menú sean
    /// visibles inmediatamente.
    id_map: SharedIdMap,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for WindowsTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowsTray").finish()
    }
}

#[cfg(target_os = "windows")]
impl WindowsTray {
    pub fn new(sections: Vec<Box<dyn MenuSection>>) -> Result<Self, TrayError> {
        let (sender, receiver) = crossbeam_channel::bounded::<MenuAction>(16);
        let rgba = icon::render_state(TrayState::Idle, Theme::System);
        let tray_icon_img = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        let (menu, id_map) = build_menu_from_sections(&sections);
        let icon = tray_icon::TrayIconBuilder::new()
            .with_icon(tray_icon_img)
            .with_tooltip("oido — idle")
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| TrayError::Tray(e.to_string()))?;

        spawn_event_forwarder(Arc::clone(&id_map), sender);

        Ok(Self {
            icon,
            receiver: Some(receiver),
            id_map,
        })
    }
}

#[cfg(target_os = "windows")]
impl Tray for WindowsTray {
    fn show(&mut self) -> Result<(), TrayError> {
        Ok(())
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), TrayError> {
        let rgba = icon::render_state(state, theme);
        let new_icon = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        self.icon
            .set_icon(Some(new_icon))
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        let tooltip = state_tooltip(state);
        self.icon
            .set_tooltip(Some(tooltip))
            .map_err(|e| TrayError::Tray(e.to_string()))?;
        Ok(())
    }
    fn hide(&mut self) -> Result<(), TrayError> {
        Ok(())
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.receiver.take()
    }
    fn rebuild_menu(&mut self, sections: Vec<Box<dyn MenuSection>>) -> Result<(), TrayError> {
        let (menu, new_id_map) = build_menu_from_sections(&sections);
        // Sustituye el menú nativo adjunto al icono. En tray-icon 0.24
        // `set_menu` no retorna Result en Windows; en macOS sí. Manejamos
        // ambas variantes vía una sola rama condicional.
        #[cfg(target_os = "macos")]
        {
            if let Err(e) = self.icon.set_menu(Some(Box::new(menu))) {
                return Err(TrayError::Tray(format!("set_menu: {e}")));
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.icon.set_menu(Some(Box::new(menu)));
        }
        // Reemplaza el id_map atómicamente; el forwarder leerá la
        // versión nueva a partir del próximo evento.
        let replacement = Arc::try_unwrap(new_id_map)
            .map(|m| m.into_inner())
            .unwrap_or_else(|arc| MenuIdMap {
                id_to_spec_id: arc.lock().id_to_spec_id.clone(),
            });
        *self.id_map.lock() = replacement;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers compartidos (Win + macOS)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn state_tooltip(state: TrayState) -> String {
    match state {
        TrayState::Idle => "oido — idle (F8 para dictar)".into(),
        TrayState::Listening => "oido — escuchando…".into(),
        TrayState::Processing => "oido — procesando…".into(),
        TrayState::Paused => "oido — pausado".into(),
        TrayState::Error => "oido — error".into(),
        TrayState::Loading => "oido — cargando modelo…".into(),
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
struct MenuIdMap {
    /// Map del `MenuId` nativo al id canónico del spec. El forwarder
    /// consulta `id_to_action` para producir el `MenuAction`.
    id_to_spec_id: HashMap<MenuId, String>,
}

/// Tipo del id_map compartido entre el thread del tray y el forwarder
/// de eventos. Se envuelve en `Arc<Mutex<…>>` para que `rebuild_menu`
/// pueda reemplazarlo atómicamente sin cerrar el channel de eventos.
#[cfg(any(target_os = "windows", target_os = "macos"))]
type SharedIdMap = Arc<Mutex<MenuIdMap>>;

/// Construye el menú nativo de bandeja iterando sobre las secciones
/// declarativas. Produce el MISMO árbol que el `build_menu` original
/// (título, separador, Cambiar hotkey, Tema, Modo, separador,
/// Utilidades, separador, Salir) y rellena un `MenuIdMap` que el
/// forwarder usa para traducir clics a `MenuAction`s.
///
/// Compilado para Windows + macOS: usamos el menú nativo de `tray-icon`
/// mientras el popup GDI custom (en `popup_window.rs`) está en
/// preparación. Cuando esté listo, será el backend principal en Windows
/// y `build_native_menu_for_windows` quedará como fallback.
#[cfg(any(target_os = "windows", target_os = "macos"))]
fn build_menu_from_sections(
    sections: &[Box<dyn MenuSection>],
) -> (tray_icon::menu::Menu, SharedIdMap) {
    use tray_icon::menu::{Menu, MenuItem, PredefinedMenuItem};

    let menu = Menu::new();
    let mut id_to_spec_id: HashMap<MenuId, String> = HashMap::new();

    // Título + separador inicial (idéntico al original).
    let title = MenuItem::new("oido 2.0", false, None);
    let _ = menu.append(&title);
    let _ = menu.append(&PredefinedMenuItem::separator());

    // Insertamos un separador entre cada sección para que la
    // apariencia sea equivalente a la del menú original (que usaba
    // separadores explícitos).
    for (i, section) in sections.iter().enumerate() {
        if i > 0 {
            // Antes de cada sección posterior a la primera, agregamos
            // separador (replica el patrón "Cambiar hotkey | Tema |
            // Modo | sep | Utilidades | sep | Salir").
            let _ = menu.append(&PredefinedMenuItem::separator());
        }
        for item in section.build() {
            append_section(&menu, item, &mut id_to_spec_id);
        }
    }

    let id_map = Arc::new(Mutex::new(MenuIdMap { id_to_spec_id }));
    (menu, id_map)
}

/// Helper recursivo: añade un `Section` al menú nativo y registra su
/// mapeo `MenuId → spec_id`. Mantiene la traducción centralizada
/// para que añadir un nuevo tipo de `Section` solo requiera un match
/// aquí.
#[cfg(any(target_os = "windows", target_os = "macos"))]
fn append_section(
    menu: &tray_icon::menu::Menu,
    section: Section,
    id_to_spec_id: &mut HashMap<MenuId, String>,
) {
    use tray_icon::menu::{MenuItem, PredefinedMenuItem, Submenu};

    match section {
        Section::Item(spec) => {
            let item = MenuItem::new(&spec.label, spec.enabled, None);
            id_to_spec_id.insert(item.id().clone(), spec.id);
            let _ = menu.append(&item);
        }
        Section::Submenu { label, items } => {
            let sub = Submenu::new(&label, true);
            for spec in items {
                let item = MenuItem::new(&spec.label, spec.enabled, None);
                id_to_spec_id.insert(item.id().clone(), spec.id);
                let _ = sub.append(&item);
            }
            let _ = menu.append(&sub);
        }
        Section::Separator => {
            let _ = menu.append(&PredefinedMenuItem::separator());
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn spawn_event_forwarder(id_map: SharedIdMap, sender: crossbeam_channel::Sender<MenuAction>) {
    use tray_icon::menu::MenuEvent;

    std::thread::spawn(move || {
        let menu_channel = MenuEvent::receiver();
        while let Ok(event) = menu_channel.recv() {
            // Resolver MenuId → spec_id (estático) → MenuAction.
            // Clonamos la vista bajo lock para minimizar hold time.
            let spec_id_opt = {
                let guard = id_map.lock();
                guard.id_to_spec_id.get(&event.id).cloned()
            };
            if let Some(spec_id) = spec_id_opt {
                if let Some(action) = id_to_action(&spec_id) {
                    let _ = sender.send(action);
                } else {
                    tracing::warn!(spec_id, "MenuEvent con spec_id sin MenuAction mapeada");
                }
            } else {
                tracing::debug!(
                    ?event.id,
                    "MenuEvent con MenuId desconocido (ignorado)"
                );
            }
        }
    });
}
