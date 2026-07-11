#![allow(unsafe_code)]

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
