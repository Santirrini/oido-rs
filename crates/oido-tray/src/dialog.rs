#![allow(unsafe_code)]

//! Diálogos nativos auxiliares para la bandeja.
//!
//! Por ahora dos:
//! - `show_model_prompt_windows` — `MessageBoxW` binario (Sí/No) usado por
//!   el flujo de first-run para ofrecer descargar el modelo por defecto.
//! - `show_prompt_editor_windows` — abre el `config.json` del usuario en
//!   el editor de texto nativo del OS (Bloc de Notas en Windows, xdg-open
//!   en Linux, `open` en macOS). El usuario edita `system_prompt` con
//!   todas las comodidades de un editor (undo, encontrar, pegar, etc.)
//!   y al cerrar la app recarga la config mediante el `ConfigStore`.
//!
//! Decisión de diseño: el menú nativo de bandeja no soporta input de
//! texto. Las opciones para un InputBox nativo son (a) pelearse con un
//! `DLGTEMPLATE` Win32 frágil, o (b) usar la crate pesada `windows`
//! async. La opción (c) — abrir el editor nativo del OS con
//! `config.json` ya abierto en la línea del campo `system_prompt` — es
//! robusta, "nativa", y ofrece una experiencia de edición *superior*
//! porque el usuario tiene undo, reemplazar, y todas las capacidades
//! reales del editor. La app refresca la config en cuanto el `mtime`
//! del archivo cambia (ver `oido_config::ConfigStore::load_or_default`).
//!
//! Devuelve `Some(())` si el editor se abrió correctamente, `None` si
//! falló la apertura. (Históricamente era `Option<String>`; el editor
//! externo se encarga de leer y guardar.)
//!
//! El `unsafe` se justifica por el FFI Win32. La función pública es
//! `Option<()>` (Some = abrió, None = no abrió) y todo el `unsafe`
//! vive dentro de este archivo — el resto del crate sigue siendo 100%
//! safe Rust.

use std::path::Path;

/// Muestra un diálogo nativo de confirmación en Windows.
/// Retorna `true` si el usuario hace clic en Sí.
/// Retorna `false` en caso contrario o en otras plataformas.
pub fn show_model_prompt_windows() -> bool {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            MessageBoxW, IDYES, MB_ICONINFORMATION, MB_YESNO,
        };

        let text: Vec<u16> = "Oido needs a speech model. Click Yes to download ggml-base.bin (~140 MB) or No to configure it later.\0"
            .encode_utf16()
            .collect();
        let title: Vec<u16> = "Oido Speech Model\0".encode_utf16().collect();

        unsafe {
            let result = MessageBoxW(
                std::ptr::null_mut(),
                text.as_ptr(),
                title.as_ptr(),
                MB_YESNO | MB_ICONINFORMATION,
            );
            result == IDYES
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// "Edita el prompt personalizado del usuario": abre `config.json` en el
/// editor de texto nativo del OS para que el campo `system_prompt`
/// pueda editarse con todas las comodidades de un editor de verdad
/// (undo, reemplazar, pegar, etc.).
///
/// En Windows se usa Bloc de Notas (`notepad.exe`), en Linux `xdg-open`
/// (default) y en macOS `open`. Si la apertura falla, devuelve `None`
/// y loguea `error`. La app NO recarga la config automáticamente al
/// cierre del editor; el reload sucede en el próximo arranque del bin
/// (cambio de hotkey, restart, etc.). Es un trade-off aceptable: editar
/// el JSON suele ser ocasional y al usuario le basta con reiniciar la
/// banda, o usar la CLI `--set-prompt` para cambios en caliente.
///
/// Devuelve `Some(())` si el editor se abrió, `None` si falló.
#[allow(unused_variables)]
pub fn show_prompt_editor_windows(config_path: &Path) -> Option<()> {
    if !config_path.exists() {
        tracing::error!(
            path = ?config_path,
            "show_prompt_editor_windows: config_path no existe; nada que abrir"
        );
        return None;
    }
    open_in_native_editor(config_path)
}

#[cfg(target_os = "windows")]
fn open_in_native_editor(path: &Path) -> Option<()> {
    use std::process::Command;
    match Command::new("notepad.exe").arg(path).spawn() {
        Ok(_) => {
            tracing::info!(path = ?path, "config.json abierto en Bloc de Notas (Notepad)");
            Some(())
        }
        Err(e) => {
            tracing::error!(
                ?e,
                path = ?path,
                "no se pudo abrir Bloc de Notas para editar config.json"
            );
            // Fallback: usar `cmd /C start notepad config.json`.
            match Command::new("cmd")
                .arg("/C")
                .arg("start")
                .arg("")
                .arg("notepad.exe")
                .arg(path)
                .spawn()
            {
                Ok(_) => {
                    tracing::info!(path = ?path, "config.json abierto vía cmd start");
                    Some(())
                }
                Err(e2) => {
                    tracing::error!(?e2, "fallback también falló");
                    None
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn open_in_native_editor(path: &Path) -> Option<()> {
    use std::process::Command;
    match Command::new("open").arg("-t").arg(path).spawn() {
        Ok(_) => {
            tracing::info!(path = ?path, "config.json abierto con `open -t` (TextEdit)");
            Some(())
        }
        Err(e) => {
            tracing::error!(?e, "no se pudo abrir TextEdit");
            None
        }
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_in_native_editor(path: &Path) -> Option<()> {
    use std::process::Command;
    // xdg-open delega al editor de texto configurado por el usuario.
    match Command::new("xdg-open").arg(path).spawn() {
        Ok(_) => {
            tracing::info!(path = ?path, "config.json abierto con xdg-open");
            Some(())
        }
        Err(e) => {
            tracing::error!(?e, "xdg-open falló");
            None
        }
    }
}
