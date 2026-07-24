//! Crate de UI nativa: bandeja, popup, icono y diálogos. Responsabilidad
//! única: **toda la UI del sistema operativo vive aquí**.
//!
//! - `Tray` / `PlatformTray` — icono de bandeja + menú contextual.
//! - `tray/popup_window.rs` — ventana Win32 borderless para el menú
//!   popup custom (con `unsafe` Win32 aislado).
//! - `dialog::show_model_prompt_windows` — MessageBox nativo Win32.
//! - `icon::generate_*` — generación procedural RGBA del icono.
//! - `win_helper::*` — helpers UI Win32 (`pump_windows_message_loop`,
//!   `set_windows_menu_theme`) que antes vivían en `oido-stt`.
//!
//! **Regla R2**: este crate **sí** contiene `unsafe` (Win32 FFI para
//! popup GDI, MessageBox y DPI awareness). Está habilitado
//! explícitamente vía `[lints.rust] unsafe_code = "allow"` en
//! `Cargo.toml`, y los bloques `unsafe` se localizan en archivos
//! marcados con `#![allow(unsafe_code)]` para auditoría trivial con
//! `grep`. Fuera de este crate, **únicamente** `oido-stt/src/whisper_cpp.rs`
//! contiene `unsafe` (regla R2 del workspace).

pub mod dialog;
pub mod dpi;
pub mod icon;
pub mod traits;
pub mod tray;
pub mod win_helper;

#[cfg(target_os = "windows")]
pub use dialog::show_model_prompt_windows;
pub use dialog::show_prompt_editor_windows;
pub use traits::{MenuAction, Tray, TrayError, TrayState};
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub use tray::mismatch_tooltip;
pub use tray::sections::{default_sections, BuildContext, MenuSection, MicDevice};
pub use tray::PlatformTray;

#[cfg(target_os = "windows")]
pub use win_helper::{pump_windows_message_loop, set_windows_menu_theme};

/// Habilita DPI awareness por monitor v2 antes de crear ventanas.
/// Llamar al inicio de `main`. Safe de invocar múltiples veces (la
/// API subyacente es idempotente a nivel de proceso).
pub fn enable_dpi_awareness() {
    dpi::enable_per_monitor_dpi_v2();
}

/// Re-export de `Theme` por compatibilidad con código que esperaba
/// `oido_platform::Theme`.
pub use oido_config::Theme;
