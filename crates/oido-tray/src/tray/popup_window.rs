//! Ventana Win32 borderless para renderizar el menú popup de la
//! bandeja.
//!
//! ## Por qué existe (en vez de usar el menú nativo de `tray-icon`)
//!
//! `tray-icon` en Windows usa `muda`, que produce un menú Win32
//! nativo (clase `#32768`). Los intentos previos de teñirlo
//! dark/light (ordinal 135 + hook CBT) son frágiles y/o no funcionan
//! en builds modernos.
//!
//! En su lugar, esta ventana se **dibuja** ella misma con GDI en
//! respuesta a `WM_PAINT`. Controlamos cada píxel: en dark mode
//! pintamos fondo negro y texto blanco; en light, fondo blanco y
//! texto casi-negro. El DPI awareness aplicado al inicio del
//! proceso escala los textos a 4K.
//!
//! ## Scope actual
//!
//! MVP: ventana rectangular sin borde Win32, fondo del color del
//! tema, una fila por línea, símbolo ✓/▸ a la derecha. Sin
//! animaciones, sin transparencia, sin escalado dinámico de fuente
//! (usa `DEFAULT_GUI_FONT` del sistema).
//!
//! ## Seguridad
//!
//! Todo el `unsafe` Win32 vive aquí. El resto del crate (incluido
//! `popup.rs`, el adapter) sigue siendo 100% safe Rust.

#![allow(unsafe_code)]

use std::sync::OnceLock;

use crate::dpi::scale_for_dpi;
use crate::tray::popup::{PopupPalette, PopupRow, RowKind};

// Local type alias para Hwnd (HANDLE = *mut c_void en windows-sys 0.59).
// Mantener como `isize` aquí complica la FFI; usar el alias canónico
// reduce casts accidentales.
//
// `Hwnd` con notación capitalized first letter para evitar el clippy
// `upper_case_acronyms` mientras se mantiene el significado semántico.
type Hwnd = *mut core::ffi::c_void;

/// Padding horizontal interno (lógico).
const H_PAD: i32 = 12;
/// Padding vertical superior/inferior (lógico).
const V_PAD: i32 = 6;
/// Alto de cada fila (lógico).
const ROW_HEIGHT: i32 = 26;
/// Alto del separador (lógico).
const SEPARATOR_HEIGHT: i32 = 9;
/// Gutter derecho para marca/símbolo (lógico).
const RIGHT_GUTTER: i32 = 22;
const MIN_WIDTH: i32 = 200;
const MAX_WIDTH: i32 = 420;

/// WM_/VK_ explicit constants para evitar dependencia en nombres
/// que cambian entre versiones de windows-sys 0.59.
const WM_CREATE_VAL: u32 = 0x0001;
const WM_PAINT_VAL: u32 = 0x000F;
const WM_ERASEBKGND_VAL: u32 = 0x0014;
const WM_MOUSEMOVE_VAL: u32 = 0x0200;
const WM_LBUTTONUP_VAL: u32 = 0x0202;
const WM_KEYDOWN_VAL: u32 = 0x0100;
const WM_NCDESTROY_VAL: u32 = 0x0082;

const VK_UP: i32 = 0x26;
const VK_DOWN: i32 = 0x28;
const VK_RETURN: i32 = 0x0D;
const VK_ESCAPE: i32 = 0x1B;
const VK_LEFT: i32 = 0x25;
const VK_RIGHT: i32 = 0x27;

const DT_SINGLELINE: u32 = 0x0000_0020;
const DT_VCENTER: u32 = 0x0000_0004;
const TRANSPARENT_BK: i32 = 1;
const SRCCOPY: u32 = 0x00CC_0020;

// =========================================================================
// State compartido por instancia de popup.
// =========================================================================

struct PopupState {
    rows: Vec<PopupRow>,
    palette: PopupPalette,
    hwnd: Hwnd,
    dpi: u32,
    width_px: i32,
    height_px: i32,
    hover: Option<usize>,
    selected: Option<usize>,
}

// =========================================================================
// Class registration.
// =========================================================================

const CLASS_NAME: &str = "oidoPopupMenu";

fn ensure_class_registered() -> bool {
    static REGISTERED: OnceLock<bool> = OnceLock::new();
    *REGISTERED.get_or_init(|| unsafe { do_register_class() })
}

unsafe fn do_register_class() -> bool {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        LoadCursorW, RegisterClassExW, CS_HREDRAW, IDC_ARROW, WNDCLASSEXW,
    };
    let class_name_w: Vec<u16> = encode_wide(CLASS_NAME);
    let wc = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW,
        lpfnWndProc: Some(wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: GetModuleHandleA(std::ptr::null()),
        hIcon: std::ptr::null_mut(),
        hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
        hbrBackground: std::ptr::null_mut(),
        lpszMenuName: std::ptr::null(),
        lpszClassName: class_name_w.as_ptr(),
        hIconSm: std::ptr::null_mut(),
    };
    RegisterClassExW(&wc) != 0
}

// =========================================================================
// WndProc
// =========================================================================

unsafe extern "system" fn wnd_proc(hwnd: Hwnd, msg: u32, wparam: usize, lparam: isize) -> isize {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DefWindowProcW, GetWindowLongPtrW, SetWindowLongPtrW, GWLP_USERDATA,
    };

    match msg {
        m if m == WM_CREATE_VAL => {
            if lparam != 0 {
                let cs =
                    lparam as *const windows_sys::Win32::UI::WindowsAndMessaging::CREATESTRUCTW;
                if !cs.is_null() {
                    let create_params = (*cs).lpCreateParams;
                    if !create_params.is_null() {
                        SetWindowLongPtrW(hwnd, GWLP_USERDATA, create_params as isize);
                        let state = &mut *(create_params as *mut PopupState);
                        state.hwnd = hwnd;
                        state.dpi = crate::dpi::dpi_for_window_or_default(hwnd as _);
                    }
                }
            }
            0
        }
        m if m == WM_MOUSEMOVE_VAL => {
            let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut PopupState;
            if !state_ptr.is_null() {
                let state = &mut *state_ptr;
                let x = (lparam as i32) & 0xFFFF_i32;
                let y = ((lparam as i32) >> 16) & 0xFFFF_i32;
                let row = hit_test(state, x, y);
                if state.hover != row {
                    state.hover = row;
                    invalidate(state.hwnd);
                }
            }
            0
        }
        m if m == WM_LBUTTONUP_VAL => {
            let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut PopupState;
            if !state_ptr.is_null() {
                let state = &mut *state_ptr;
                let x = (lparam as i32) & 0xFFFF_i32;
                let y = ((lparam as i32) >> 16) & 0xFFFF_i32;
                if let Some(idx) = hit_test(state, x, y) {
                    if matches!(state.rows[idx].kind, RowKind::Action { .. }) {
                        state.selected = Some(idx);
                    } else {
                        state.selected = None;
                    }
                } else {
                    state.selected = None;
                }
                close_popup();
            }
            0
        }
        m if m == WM_KEYDOWN_VAL => {
            let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut PopupState;
            if state_ptr.is_null() {
                return DefWindowProcW(hwnd, msg, wparam, lparam);
            }
            let state = &mut *state_ptr;
            let vk = wparam as i32;
            match vk {
                v if v == VK_UP => move_hover(state, -1),
                v if v == VK_DOWN => move_hover(state, 1),
                v if v == VK_RETURN => {
                    if let Some(idx) = state.hover {
                        if matches!(state.rows[idx].kind, RowKind::Action { .. }) {
                            state.selected = Some(idx);
                            close_popup();
                        }
                    }
                }
                v if v == VK_ESCAPE => {
                    state.selected = None;
                    close_popup();
                }
                v if v == VK_RIGHT => {
                    if let Some(idx) = state.hover {
                        if matches!(state.rows[idx].kind, RowKind::Submenu { .. }) {
                            state.selected = Some(idx);
                            close_popup();
                        }
                    }
                }
                v if v == VK_LEFT => {
                    state.selected = None;
                    close_popup();
                }
                _ => return DefWindowProcW(hwnd, msg, wparam, lparam),
            }
            0
        }
        m if m == WM_PAINT_VAL => {
            let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut PopupState;
            if !state_ptr.is_null() {
                paint(&*state_ptr);
            }
            0
        }
        m if m == WM_ERASEBKGND_VAL => 1,
        m if m == WM_NCDESTROY_VAL => {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

// =========================================================================
// Hit-testing + helpers.
// =========================================================================

fn hit_test(state: &PopupState, _x: i32, y: i32) -> Option<usize> {
    let mut cy = scale_for_dpi(V_PAD, state.dpi);
    for (i, row) in state.rows.iter().enumerate() {
        let h = match row.kind {
            RowKind::Action { .. } | RowKind::Submenu { .. } => {
                scale_for_dpi(ROW_HEIGHT, state.dpi)
            }
            RowKind::Separator => scale_for_dpi(SEPARATOR_HEIGHT, state.dpi),
        };
        if y >= cy && y < cy + h {
            return Some(i);
        }
        cy += h;
    }
    None
}

fn move_hover(state: &mut PopupState, delta: i32) {
    if state.rows.is_empty() {
        return;
    }
    let n = state.rows.len() as i32;
    let mut idx = state
        .hover
        .map(|h| h as i32)
        .unwrap_or(if delta > 0 { -1 } else { n });
    for _ in 0..n {
        idx = (idx + delta).rem_euclid(n);
        if matches!(
            state.rows[idx as usize].kind,
            RowKind::Action { .. } | RowKind::Submenu { .. }
        ) {
            break;
        }
    }
    state.hover = Some(idx as usize);
    invalidate(state.hwnd);
}

fn invalidate(hwnd: Hwnd) {
    unsafe {
        let user32 = windows_sys::Win32::System::LibraryLoader::LoadLibraryA(
            c"user32.dll".as_ptr() as *const u8,
        );
        if user32.is_null() {
            return;
        }
        let proc = windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            user32,
            c"InvalidateRect".as_ptr() as _,
        );
        if let Some(proc) = proc {
            type Fn = unsafe extern "system" fn(Hwnd, *const core::ffi::c_void, i32) -> i32;
            let invalidate_rect: Fn = std::mem::transmute(proc);
            invalidate_rect(hwnd, std::ptr::null(), 1);
        }
    }
}

fn close_popup() {
    unsafe {
        let user32 = windows_sys::Win32::System::LibraryLoader::LoadLibraryA(
            c"user32.dll".as_ptr() as *const u8,
        );
        if user32.is_null() {
            return;
        }
        let proc = windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            user32,
            c"PostQuitMessage".as_ptr() as _,
        );
        if let Some(proc) = proc {
            type Fn = unsafe extern "system" fn(i32) -> i32;
            let post_quit: Fn = std::mem::transmute(proc);
            post_quit(0);
        }
    }
}

// =========================================================================
// Pintado.
// =========================================================================

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn paint(state: &PopupState) {
    unsafe {
        use windows_sys::Win32::Graphics::Gdi::{
            BeginPaint, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, CreateSolidBrush,
            DeleteDC, DeleteObject, DrawTextW, EndPaint, FillRect, SetBkMode, SetTextColor,
            PAINTSTRUCT,
        };
        let mut ps: PAINTSTRUCT = std::mem::zeroed();
        let hdc = BeginPaint(state.hwnd, &mut ps);

        // Doble buffer.
        let mem_dc = CreateCompatibleDC(hdc);
        let bmp = CreateCompatibleBitmap(hdc, state.width_px, state.height_px);
        let _old_bmp = windows_sys::Win32::Graphics::Gdi::SelectObject(mem_dc, bmp);

        // Fondo
        let bg_brush = CreateSolidBrush(rgb_to_colorref(state.palette.bg));
        let fill = make_rect(0, 0, state.width_px, state.height_px);
        FillRect(mem_dc, &fill, bg_brush);
        DeleteObject(bg_brush);

        // Fuente: usamos el DEFAULT_GUI_FONT del sistema (no
        // recreamos para mantener el popup simple).
        let default_font = get_stock_default_gui_font();
        let _old_font = windows_sys::Win32::Graphics::Gdi::SelectObject(mem_dc, default_font);

        SetTextColor(mem_dc, rgb_to_colorref(state.palette.text));
        SetBkMode(mem_dc, TRANSPARENT_BK);

        // Filas
        let mut cy = scale_for_dpi(V_PAD, state.dpi);
        for (i, row) in state.rows.iter().enumerate() {
            let (h, hover, is_sep) = match row.kind {
                RowKind::Action { .. } | RowKind::Submenu { .. } => (
                    scale_for_dpi(ROW_HEIGHT, state.dpi),
                    state.hover == Some(i),
                    false,
                ),
                RowKind::Separator => (scale_for_dpi(SEPARATOR_HEIGHT, state.dpi), false, true),
            };
            if hover && !is_sep {
                let hover_brush = CreateSolidBrush(rgb_to_colorref(state.palette.hover));
                let r = make_rect(0, cy, state.width_px, cy + h);
                FillRect(mem_dc, &r, hover_brush);
                DeleteObject(hover_brush);
            }
            if is_sep {
                let sep_brush = CreateSolidBrush(rgb_to_colorref(state.palette.separator));
                let mid = cy + h / 2;
                let r = make_rect(
                    scale_for_dpi(H_PAD, state.dpi),
                    mid - 1,
                    state.width_px - scale_for_dpi(H_PAD, state.dpi),
                    mid + 1,
                );
                FillRect(mem_dc, &r, sep_brush);
                DeleteObject(sep_brush);
            } else {
                let label = match &row.kind {
                    RowKind::Action { label, .. } | RowKind::Submenu { label, .. } => label.clone(),
                    RowKind::Separator => String::new(),
                };
                let right_text = match &row.kind {
                    RowKind::Action { active, .. } => {
                        if *active {
                            "\u{2713}".to_string()
                        } else {
                            String::new()
                        }
                    }
                    RowKind::Submenu { .. } => "\u{25B8}".to_string(),
                    RowKind::Separator => String::new(),
                };

                let text_left = scale_for_dpi(H_PAD, state.dpi);
                let right_gutter_px = scale_for_dpi(RIGHT_GUTTER, state.dpi);
                let h_pad_px = scale_for_dpi(H_PAD, state.dpi);

                let label_w: Vec<u16> = encode_wide(&label);
                let mut label_rect = make_rect(
                    text_left,
                    cy,
                    state.width_px - text_left - h_pad_px - right_gutter_px,
                    cy + h,
                );
                DrawTextW(
                    mem_dc,
                    label_w.as_ptr() as *mut _,
                    (label_w.len() - 1) as i32,
                    &mut label_rect as *mut _,
                    DT_SINGLELINE | DT_VCENTER,
                );

                if !right_text.is_empty() {
                    let sym_w: Vec<u16> = encode_wide(&right_text);
                    let mut sym_rect = make_rect(
                        state.width_px - h_pad_px - right_gutter_px,
                        cy,
                        state.width_px - h_pad_px,
                        cy + h,
                    );
                    DrawTextW(
                        mem_dc,
                        sym_w.as_ptr() as *mut _,
                        (sym_w.len() - 1) as i32,
                        &mut sym_rect as *mut _,
                        DT_SINGLELINE | DT_VCENTER,
                    );
                }
            }
            cy += h;
        }

        BitBlt(
            hdc,
            0,
            0,
            state.width_px,
            state.height_px,
            mem_dc,
            0,
            0,
            SRCCOPY,
        );

        DeleteObject(bmp);
        DeleteDC(mem_dc);
        EndPaint(state.hwnd, &ps);
    }
}

// =========================================================================
// Helpers
// =========================================================================

fn encode_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[allow(clippy::cast_possible_truncation)]
fn rgb_to_colorref(rgb: (u8, u8, u8)) -> u32 {
    let (r, g, b) = rgb;
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
}

fn make_rect(left: i32, top: i32, right: i32, bottom: i32) -> windows_sys::Win32::Foundation::RECT {
    windows_sys::Win32::Foundation::RECT {
        left,
        top,
        right,
        bottom,
    }
}

fn get_stock_default_gui_font() -> Hwnd {
    unsafe {
        let gdi32 = windows_sys::Win32::System::LibraryLoader::LoadLibraryA(
            c"gdi32.dll".as_ptr() as *const u8
        );
        if gdi32.is_null() {
            return std::ptr::null_mut();
        }
        let proc = windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            gdi32,
            c"GetStockObject".as_ptr() as _,
        );
        if let Some(proc) = proc {
            type Fn = unsafe extern "system" fn(i32) -> Hwnd;
            let get_stock: Fn = std::mem::transmute(proc);
            get_stock(17) // DEFAULT_GUI_FONT
        } else {
            std::ptr::null_mut()
        }
    }
}

// =========================================================================
// Loop modal público.
// =========================================================================

pub fn run_popup(
    rows: Vec<PopupRow>,
    palette: PopupPalette,
    anchor_x: i32,
    anchor_y: i32,
) -> Option<(usize, Option<usize>)> {
    if rows.is_empty() {
        return None;
    }
    if !ensure_class_registered() {
        return None;
    }

    let (width_px, height_px) = compute_size(&rows);
    let dpi = current_dpi();

    let mut state = PopupState {
        rows,
        palette,
        hwnd: std::ptr::null_mut(),
        dpi,
        width_px,
        height_px,
        hover: None,
        selected: None,
    };

    let hwnd = unsafe {
        use windows_sys::Win32::Graphics::Dwm::DwmSetWindowAttribute;
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, SetWindowPos, ShowWindow, HWND_TOPMOST, SW_SHOWNOACTIVATE,
            WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP,
        };
        let class_name_w: Vec<u16> = encode_wide(CLASS_NAME);
        let title_w: Vec<u16> = "oido\0".encode_utf16().collect();
        let hwnd_raw = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            class_name_w.as_ptr(),
            title_w.as_ptr(),
            WS_POPUP,
            anchor_x,
            anchor_y,
            width_px,
            height_px,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            GetModuleHandleA(std::ptr::null()),
            &mut state as *mut _ as *const _,
        );
        if hwnd_raw.is_null() {
            return None;
        }
        // DWMWA_WINDOW_CORNER_PREFERENCE = 33; DWMWCP_ROUND = 2.
        let rounded: u32 = 2;
        let _ = DwmSetWindowAttribute(
            hwnd_raw,
            33,
            &rounded as *const _ as *const _,
            std::mem::size_of_val(&rounded) as u32,
        );
        SetWindowPos(hwnd_raw, HWND_TOPMOST, 0, 0, 0, 0, 0x0017);
        ShowWindow(hwnd_raw, SW_SHOWNOACTIVATE);
        hwnd_raw
    };
    state.hwnd = hwnd;

    unsafe {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, GetMessageW, TranslateMessage, MSG,
        };
        // SetCapture por ordinal — `SetCapture` no siempre está en el
        // feature `Win32_UI_WindowsAndMessaging` 0.59. Resolvemos por
        // nombre en runtime. Como beneficio: aunque la captura falle,
        // el popup sigue funcionando (lose click-outside handling).
        let user32 = windows_sys::Win32::System::LibraryLoader::LoadLibraryA(
            c"user32.dll".as_ptr() as *const u8,
        );
        if !user32.is_null() {
            let proc = windows_sys::Win32::System::LibraryLoader::GetProcAddress(
                user32,
                c"SetCapture".as_ptr() as _,
            );
            if let Some(proc) = proc {
                type Fn = unsafe extern "system" fn(Hwnd) -> i32;
                let set_capture: Fn = std::mem::transmute(proc);
                let _ = set_capture(hwnd);
            }
        }

        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    let selected = state.selected;
    let rows = state.rows;
    let palette = state.palette;
    if let Some(idx) = selected {
        match &rows[idx].kind {
            RowKind::Action { .. } => Some((idx, None)),
            RowKind::Submenu { child, .. } => {
                if child.is_empty() {
                    return None;
                }
                let sub_anchor_x = anchor_x + width_px;
                let sub_anchor_y = anchor_y
                    + scale_for_dpi(V_PAD, dpi)
                    + (idx as i32) * scale_for_dpi(ROW_HEIGHT, dpi);
                let sub = run_popup(child.clone(), palette, sub_anchor_x, sub_anchor_y);
                if let Some((sub_idx, _)) = sub {
                    Some((idx, Some(sub_idx)))
                } else {
                    Some((idx, None))
                }
            }
            RowKind::Separator => None,
        }
    } else {
        None
    }
}

fn compute_size(rows: &[PopupRow]) -> (i32, i32) {
    let dpi = current_dpi();
    let max_label_len = rows
        .iter()
        .map(|r| match &r.kind {
            RowKind::Action { label, .. } | RowKind::Submenu { label, .. } => label.chars().count(),
            RowKind::Separator => 0,
        })
        .max()
        .unwrap_or(20);
    let ideal_width = ((max_label_len as i32) * 8) + 2 * H_PAD + RIGHT_GUTTER;
    let width = ideal_width.clamp(MIN_WIDTH, MAX_WIDTH);
    let width_px = scale_for_dpi(width, dpi);

    let mut height = V_PAD * 2;
    for row in rows {
        height += match row.kind {
            RowKind::Action { .. } | RowKind::Submenu { .. } => ROW_HEIGHT,
            RowKind::Separator => SEPARATOR_HEIGHT,
        };
    }
    let height_px = scale_for_dpi(height, dpi);
    (width_px, height_px)
}

fn current_dpi() -> u32 {
    unsafe {
        use windows_sys::Win32::UI::WindowsAndMessaging::GetDesktopWindow;
        let h = GetDesktopWindow();
        crate::dpi::dpi_for_window_or_default(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_size_one_short_row() {
        let rows = vec![PopupRow {
            kind: RowKind::Action {
                id: "x".into(),
                label: "Peque\u{00f1}o".into(),
                active: false,
            },
        }];
        let (_w, h) = compute_size(&rows);
        assert!(h > 0, "alto debe ser positivo");
    }

    #[test]
    fn compute_size_grows_with_more_rows() {
        let make_action = |i: usize| PopupRow {
            kind: RowKind::Action {
                id: format!("a{i}"),
                label: format!("Acci\u{00f3}n {i}"),
                active: false,
            },
        };
        let one = vec![make_action(0)];
        let many = (0..10).map(make_action).collect::<Vec<_>>();
        let (_, h1) = compute_size(&one);
        let (_, h10) = compute_size(&many);
        assert!(h10 > h1, "altura crece con m\u{00e1}s filas");
    }
}
