//! Helpers Win32 para el menú nativo y la pump de mensajes.
//!
//! Estas funciones viven aquí (no en `oido-stt`) por responsabilidad
//! única: son UI pura, no tienen nada que ver con el motor de STT.
//! Antes del refactor vivían en `oido-stt/src/whisper_cpp.rs` como
//! "helpers al lado del FFI", pero su razón de cambio es el tray, no
//! la transcripción.
//!
//! Todo el `unsafe` Win32 vive tras un `#![allow(unsafe_code)]` a
//! nivel de módulo. El resto del crate sigue siendo 100% safe Rust.

#![allow(unsafe_code)]

/// Procesa de forma no bloqueante la cola de mensajes de Win32 para
/// bombear los eventos del tray icon (clics y menú contextual).
#[cfg(target_os = "windows")]
pub fn pump_windows_message_loop() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
    };
    unsafe {
        let mut msg: MSG = std::mem::zeroed();
        while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// Modifica el tema preferido para el proceso en Windows.
///
/// Aplica `SetPreferredAppMode` (ordinal 135 de `uxtheme.dll`), que
/// afecta al non-client area del icono y a cualquier control nativo
/// residual que la app pueda crear (tooltips, system tray tooltip).
/// Para el menú popup ya no es relevante — `tray/popup_window.rs`
/// renderiza la ventana completa con GDI sin tocar APIs oscuras.
///
/// Mapeo:
///   `Dark`   → `ForceDark` (2)
///   `Light`  → `ForceLight` (3)
///   `System` → `Default` (0 = no tocamos; que decida el SO)
#[cfg(target_os = "windows")]
pub fn set_windows_menu_theme(theme: oido_config::Theme) {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

    let mode: i32 = match theme {
        oido_config::Theme::Dark => 2,
        oido_config::Theme::Light => 3,
        oido_config::Theme::System => 0,
    };
    if mode == 0 {
        return;
    }

    unsafe {
        let uxtheme = LoadLibraryA(c"uxtheme.dll".as_ptr() as *const u8);
        if uxtheme.is_null() {
            tracing::debug!("uxtheme.dll no disponible; SetPreferredAppMode omitido");
            return;
        }
        let func = GetProcAddress(uxtheme, 135 as _);
        if let Some(func) = func {
            let func: unsafe extern "system" fn(i32) -> i32 = std::mem::transmute(func);
            let ok = func(mode);
            tracing::debug!(
                mode,
                ok,
                "SetPreferredAppMode aplicado (no afecta al popup custom)"
            );
        }
    }
}
