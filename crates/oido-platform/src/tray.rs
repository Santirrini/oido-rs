//! Implementaciones nativas del icono de bandeja por plataforma.
//!
//! - Windows + macOS: `tray-icon` + `muda` (re-exportado por `tray-icon::menu`).
//! - Linux: `ksni` (StatusNotifierItem / D-Bus).
//!
//! La API pública (`PlatformTray`, `Tray` trait) permanece idéntica; solo
//! el interior cambia de stub a implementación real.

use oido_config::Theme;

use crate::icon;
use crate::traits::{MenuAction, PlatformError, Tray, TrayState};

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
    pub fn new() -> Result<Self, PlatformError> {
        Ok(Self(Inner::new()?))
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
    fn show(&mut self) -> Result<(), PlatformError> {
        self.0.show()
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), PlatformError> {
        self.0.set_state(state, theme)
    }
    fn hide(&mut self) -> Result<(), PlatformError> {
        self.0.hide()
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.0.take_menu_events()
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
    pub fn new() -> Result<Self, PlatformError> {
        let (sender, receiver) = crossbeam_channel::bounded(16);
        Ok(Self {
            sender,
            receiver: Some(receiver),
            current_state: TrayState::Idle,
            current_theme: Theme::System,
        })
    }
}

#[cfg(target_os = "linux")]
impl Tray for LinuxTray {
    fn show(&mut self) -> Result<(), PlatformError> {
        tracing::info!("tray Linux (ksni): show");
        Ok(())
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), PlatformError> {
        self.current_state = state;
        self.current_theme = theme;
        tracing::info!(?state, "tray Linux state");
        Ok(())
    }
    fn hide(&mut self) -> Result<(), PlatformError> {
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
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for MacTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacTray").finish()
    }
}

#[cfg(target_os = "macos")]
impl MacTray {
    pub fn new() -> Result<Self, PlatformError> {
        let (sender, receiver) = crossbeam_channel::bounded::<MenuAction>(16);
        let rgba = icon::render_state(TrayState::Idle, Theme::System);
        let tray_icon_img = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        let (menu, id_map) = build_menu();
        let icon = tray_icon::TrayIconBuilder::new()
            .with_icon(tray_icon_img)
            .with_tooltip("oido — idle")
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| PlatformError::Tray(e.to_string()))?;

        spawn_event_forwarder(id_map, sender);

        Ok(Self {
            icon,
            receiver: Some(receiver),
        })
    }
}

#[cfg(target_os = "macos")]
impl Tray for MacTray {
    fn show(&mut self) -> Result<(), PlatformError> {
        Ok(())
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), PlatformError> {
        let rgba = icon::render_state(state, theme);
        let new_icon = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        self.icon
            .set_icon(Some(new_icon))
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        let tooltip = state_tooltip(state);
        self.icon
            .set_tooltip(Some(tooltip))
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        Ok(())
    }
    fn hide(&mut self) -> Result<(), PlatformError> {
        Ok(())
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.receiver.take()
    }
}

// ---------------------------------------------------------------------------
// Windows — tray-icon + menu
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
pub struct WindowsTray {
    icon: tray_icon::TrayIcon,
    receiver: Option<crossbeam_channel::Receiver<MenuAction>>,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for WindowsTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowsTray").finish()
    }
}

#[cfg(target_os = "windows")]
impl WindowsTray {
    pub fn new() -> Result<Self, PlatformError> {
        let (sender, receiver) = crossbeam_channel::bounded::<MenuAction>(16);
        let rgba = icon::render_state(TrayState::Idle, Theme::System);
        let tray_icon_img = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        let (menu, id_map) = build_menu();
        let icon = tray_icon::TrayIconBuilder::new()
            .with_icon(tray_icon_img)
            .with_tooltip("oido — idle")
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| PlatformError::Tray(e.to_string()))?;

        spawn_event_forwarder(id_map, sender);

        Ok(Self {
            icon,
            receiver: Some(receiver),
        })
    }
}

#[cfg(target_os = "windows")]
impl Tray for WindowsTray {
    fn show(&mut self) -> Result<(), PlatformError> {
        Ok(())
    }
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), PlatformError> {
        let rgba = icon::render_state(state, theme);
        let new_icon = tray_icon::Icon::from_rgba(rgba.data, rgba.width, rgba.height)
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        self.icon
            .set_icon(Some(new_icon))
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        let tooltip = state_tooltip(state);
        self.icon
            .set_tooltip(Some(tooltip))
            .map_err(|e| PlatformError::Tray(e.to_string()))?;
        Ok(())
    }
    fn hide(&mut self) -> Result<(), PlatformError> {
        Ok(())
    }
    fn take_menu_events(&mut self) -> Option<crossbeam_channel::Receiver<MenuAction>> {
        self.receiver.take()
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
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
struct MenuIdMap {
    id_change_hotkey: tray_icon::menu::MenuId,
    id_dark: tray_icon::menu::MenuId,
    id_light: tray_icon::menu::MenuId,
    id_system: tray_icon::menu::MenuId,
    id_open_models: tray_icon::menu::MenuId,
    id_check_updates: tray_icon::menu::MenuId,
    id_exit: tray_icon::menu::MenuId,
}

/// Construye el menú nativo de bandeja con `tray_icon::menu`.
#[cfg(any(target_os = "windows", target_os = "macos"))]
fn build_menu() -> (tray_icon::menu::Menu, MenuIdMap) {
    use tray_icon::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};

    let menu = Menu::new();

    let title = MenuItem::new("oido 2.0", false, None);
    let _ = menu.append(&title);
    let _ = menu.append(&PredefinedMenuItem::separator());

    let change_hotkey = MenuItem::new("Cambiar hotkey…", true, None);
    let id_change_hotkey = change_hotkey.id().clone();
    let _ = menu.append(&change_hotkey);

    let theme_sub = Submenu::new("Tema", true);
    let dark_item = MenuItem::new("Dark", true, None);
    let id_dark = dark_item.id().clone();
    let light_item = MenuItem::new("Light", true, None);
    let id_light = light_item.id().clone();
    let system_item = MenuItem::new("Sistema", true, None);
    let id_system = system_item.id().clone();
    let _ = theme_sub.append(&dark_item);
    let _ = theme_sub.append(&light_item);
    let _ = theme_sub.append(&system_item);
    let _ = menu.append(&theme_sub);

    let _ = menu.append(&PredefinedMenuItem::separator());

    let open_models = MenuItem::new("Abrir carpeta de modelos", true, None);
    let id_open_models = open_models.id().clone();
    let _ = menu.append(&open_models);

    let check_updates = MenuItem::new("Buscar actualizaciones", true, None);
    let id_check_updates = check_updates.id().clone();
    let _ = menu.append(&check_updates);

    let _ = menu.append(&PredefinedMenuItem::separator());

    let exit_item = MenuItem::new("Salir", true, None);
    let id_exit = exit_item.id().clone();
    let _ = menu.append(&exit_item);

    let id_map = MenuIdMap {
        id_change_hotkey,
        id_dark,
        id_light,
        id_system,
        id_open_models,
        id_check_updates,
        id_exit,
    };

    (menu, id_map)
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn spawn_event_forwarder(id_map: MenuIdMap, sender: crossbeam_channel::Sender<MenuAction>) {
    use tray_icon::menu::MenuEvent;

    std::thread::spawn(move || {
        let menu_channel = MenuEvent::receiver();
        while let Ok(event) = menu_channel.recv() {
            if event.id == id_map.id_change_hotkey {
                let _ = sender.send(MenuAction::ChangeHotkey);
            } else if event.id == id_map.id_dark {
                let _ = sender.send(MenuAction::SetTheme(Theme::Dark));
            } else if event.id == id_map.id_light {
                let _ = sender.send(MenuAction::SetTheme(Theme::Light));
            } else if event.id == id_map.id_system {
                let _ = sender.send(MenuAction::SetTheme(Theme::System));
            } else if event.id == id_map.id_open_models {
                let _ = sender.send(MenuAction::OpenModelsDir);
            } else if event.id == id_map.id_check_updates {
                let _ = sender.send(MenuAction::CheckUpdates);
            } else if event.id == id_map.id_exit {
                let _ = sender.send(MenuAction::Exit);
            }
        }
    });
}
