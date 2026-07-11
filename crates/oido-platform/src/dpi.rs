//! DPI awareness del proceso.
//!
//! Por defecto, Windows corre las apps sin DPI awareness explícita y
//! **estira los bitmaps** al renderizarlos en pantallas HiDPI. El
//! resultado visual es texto/iconos borrosos en pantallas 4K o
//! escaladas al 125 % / 150 %.
//!
//! Llamamos a `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`
//! al inicio del proceso. El flag "V2" (`-4`) es la versión que cubre
//! los nuevos `DPI_AWARENESS_CONTEXT_*` introducidos en Win10 1703+:
//! permite que cada monitor tenga su propio DPI y que cambiemos de
//! monitor arrastrando la ventana sin recargar.
//!
//! ## Por qué V2 y no V1
//! V1 (`DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE`, `-3`) ya cubre el
//! caso "1 DPI por monitor"; V2 añade:
//! - `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2` = escalado non-client
//!   area (título, scrollbars) consistente con el DPI.
//! - Mejor comportamiento con `GetSystemMetricsForDpi` / `GetDpiForWindow`.
//!
//! ## Idempotencia
//! `SetProcessDpiAwarenessContext` solo puede llamarse una vez por
//! proceso. Si ya fue seteada por la CRT o un manifest embebido,
//! devuelve `FALSE` con last-error `ERROR_INVALID_PARAMETER` y no
//! pasa nada — seguimos como si nada.
//!
//! ## Falla silenciosa
//! Si la fn no existe (Win8.1 / Server 2012 sin parches), GetProcAddress
//! devuelve `null` y el proceso seguirá con DPI awareness "system",
//! que ya es mejor que "unaware" en builds modernos.
//!
//! Esta función se llama **antes** de crear cualquier ventana. Es
//! process-global.

#![allow(unsafe_code)]

/// Activa DPI awareness **por monitor v2** para todo el proceso.
/// Llamar al inicio de `main`, idealmente antes de cualquier
/// `CreateWindow*` o `tray-icon::TrayIconBuilder`.
///
/// No retorna nada: si la llamada falla, el proceso sigue con el
/// DPI awareness por defecto. Se loguea el error via `tracing` para
/// diagnóstico.
pub fn enable_per_monitor_dpi_v2() {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

        // Cache del puntero a la función. `user32.dll` siempre está
        // cargado en procesos GUI; los símbolos deberían resolverse
        // directamente, pero usamos GetProcAddress por defensa contra
        // un futuro user32 minimal (builds nanoserver).
        unsafe {
            let user32 = LoadLibraryA(c"user32.dll".as_ptr() as *const u8);
            if user32.is_null() {
                tracing::warn!("no se pudo cargar user32.dll; DPI awareness no aplicada");
                return;
            }
            let proc = GetProcAddress(user32, c"SetProcessDpiAwarenessContext".as_ptr() as _);
            if proc.is_none() {
                tracing::debug!(
                    "SetProcessDpiAwarenessContext no disponible (Win8.1?); \
                     usando DPI awareness por defecto del sistema"
                );
                return;
            }
            // Cast: SetProcessDpiAwarenessContext tiene firma
            // BOOL(HANDLE /* DPI_AWARENESS_CONTEXT */) -> BOOL. Ambos
            // son handles (LPVOID-equivalentes) y BOOL.
            //
            // El valor de `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`
            // no está expuesto por la feature `Win32_UI_HiDpi` 0.59
            // (es un handle opaco al runtime); el literal `-4` es
            // estable y está documentado en MSDN.
            type Fn = unsafe extern "system" fn(isize) -> i32;
            let set_awareness: Fn = std::mem::transmute(proc.unwrap());
            let ok = set_awareness(-4_isize); // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            if ok == 0 {
                // ERROR_INVALID_PARAMETER (87) si ya estaba seteada
                // por un manifest o llamada previa. No es un error
                // fatal.
                let err = windows_sys::Win32::Foundation::GetLastError();
                tracing::debug!(
                    last_error = err,
                    "SetProcessDpiAwarenessContext no aplicada (¿ya seteada?)"
                );
            } else {
                tracing::info!("DPI awareness per-monitor V2 activada");
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        // No-op fuera de Windows; en macOS/Linux el framework ya
        // gestiona HiDPI por defecto.
    }
}

/// Wrapper sobre `GetDpiForWindow`. Devuelve el DPI efectivo de la
/// ventana. Fallback a 96 (1.0× logical→physical) si no está
/// disponible o si `hwnd` es inválido.
///
/// **`unsafe`** porque `hwnd` es un raw pointer a un HWND Win32 — el
/// caller debe asegurar que el HWND sigue siendo válido (o usar la
/// versión `_default` que devuelve 96 si falla la llamada).
///
/// # Safety
///
/// `hwnd` debe ser un HWND válido o un pointer conocido a inválido
/// (tolerante: GetDpiForWindow retorna 0 y se devuelve 96 como default
/// en ese caso).
pub unsafe fn dpi_for_window_or_default(hwnd: *mut core::ffi::c_void) -> u32 {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

        unsafe {
            let user32 = LoadLibraryA(c"user32.dll".as_ptr() as *const u8);
            if user32.is_null() {
                return 96;
            }
            let proc = GetProcAddress(user32, c"GetDpiForWindow".as_ptr() as _);
            if proc.is_none() {
                return 96;
            };
            type Fn = unsafe extern "system" fn(*mut core::ffi::c_void) -> u32;
            let get: Fn = std::mem::transmute(proc.unwrap());
            let dpi = get(hwnd);
            if dpi == 0 {
                96
            } else {
                dpi
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = hwnd;
        96
    }
}

/// Conversión "logical units" (píxeles a 100% / 96 DPI) → píxeles
/// físicos al DPI actual. Útil para escalar fonts y paddings del
/// popup GDI a HiDPI.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn scale_for_dpi(dip: i32, dpi: u32) -> i32 {
    // dpi / 96 es el factor; usamos rounding para no perder píxeles
    // en escalados comunes (100%, 125%, 150%, 175%, 200%). El rango
    // final cabe en i32 para todos los valores razonables (un valor
    // de 4096 dip @ 4800 dpi aún está dentro de i32).
    let r = i64::from(dip) * i64::from(dpi);
    ((r + 48) / 96) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_for_dpi_at_100_percent_is_identity() {
        assert_eq!(scale_for_dpi(20, 96), 20);
        assert_eq!(scale_for_dpi(0, 96), 0);
        assert_eq!(scale_for_dpi(100, 96), 100);
    }

    #[test]
    fn scale_for_dpi_at_150_percent_rounds_correctly() {
        // 20 dip * 144 dpi / 96 = 30.0
        assert_eq!(scale_for_dpi(20, 144), 30);
        // 16 dip * 144 dpi / 96 = 24.0
        assert_eq!(scale_for_dpi(16, 144), 24);
    }

    #[test]
    fn scale_for_dpi_at_200_percent_doubles() {
        assert_eq!(scale_for_dpi(20, 192), 40);
    }

    #[test]
    fn scale_for_dpi_handles_fractional_rounding() {
        // 7 * 144 / 96 = 10.5 → con +48 rounding → (7*144+48)/96 = 1056/96 = 11
        // Doc del redondeo: usamos "+48" (half de 96) para redondeo
        // hacia el píxel más cercano, ceil para .5 exacto.
        assert_eq!(scale_for_dpi(7, 144), 11);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn enable_per_monitor_dpi_v2_does_not_panic() {
        // Como SetProcessDpiAwarenessContext es process-global y el
        // test runner ejecuta múltiples tests en el mismo proceso,
        // esta función puede ya estar aplicada. Solo verificamos
        // que la llamada no panica.
        enable_per_monitor_dpi_v2();
    }

    #[test]
    fn dpi_for_window_or_default_returns_96_when_invalid() {
        // hwnd inválido → 0 → devolvemos 96.
        // SAFETY: el puntero 0xdead no apunta a nada; GetDpiForWindow
        // retorna 0 en ese caso y caemos al default.
        let dpi = unsafe { dpi_for_window_or_default(0xdead as *mut core::ffi::c_void) };
        assert_eq!(dpi, 96);
    }
}
