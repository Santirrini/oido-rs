//! Generación procedural del icono de bandeja (64×64 RGBA, safe Rust puro).
//!
//! No depende de ningún crate de rendering externo. Dibuja el micrófono a mano
//! como formas geométricas básicas (rectángulos + círculos rasterizados).
//!
//! # Convención de color
//! - Dark mode: fondo transparente, formas con color saturado. Visible sobre taskbar oscura.
//! - Light mode: fondo transparente, formas con overlay negro para contrastar con taskbar blanca.

use oido_config::Theme;

use crate::traits::TrayState;

pub const WIDTH: u32 = 64;
pub const HEIGHT: u32 = 64;

/// Buffer RGBA devuelto por el renderer.
#[derive(Debug)]
pub struct RgbaIcon {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Genera el icono para el `state` dado con la paleta según `theme`.
pub fn render_state(state: TrayState, theme: Theme) -> RgbaIcon {
    let mut buf = vec![0u8; (WIDTH * HEIGHT * 4) as usize];

    let palette = choose_palette(state, theme);

    // --- Cuerpo del micrófono (cápsula): rectángulo central + semicírculos ---
    let mic_x = 22u32;
    let mic_w = 20u32;
    let mic_top = 8u32;
    let mic_body_h = 28u32;
    let mic_cx = WIDTH / 2;
    let mic_cy_top = mic_top + 10; // centro del semicírculo superior
    let mic_cy_bot = mic_top + mic_body_h - 10; // centro del semicírculo inferior

    // Rectángulo del cuerpo
    fill_rect(
        &mut buf,
        mic_x,
        mic_top + 10,
        mic_w,
        mic_body_h - 20,
        palette.mic,
    );
    // Semicírculo superior
    fill_circle(&mut buf, mic_cx, mic_cy_top, 10, palette.mic);
    // Semicírculo inferior
    fill_circle(&mut buf, mic_cx, mic_cy_bot, 10, palette.mic);

    // --- Arco exterior (cuello del micrófono) ---
    let arc_cy = mic_top + mic_body_h; // base del mic
    let arc_outer = 14u32;
    let arc_inner = 10u32;
    draw_arc_bottom(&mut buf, mic_cx, arc_cy, arc_outer, arc_inner, palette.mic);

    // --- Barra vertical + base ---
    fill_rect(
        &mut buf,
        mic_cx - 1,
        arc_cy + arc_outer - 4,
        2,
        8,
        palette.mic,
    );
    fill_rect(
        &mut buf,
        mic_cx - 8,
        arc_cy + arc_outer + 4,
        16,
        2,
        palette.mic,
    );

    // --- Overlays por estado ---
    match state {
        TrayState::Idle => {}
        TrayState::Listening => {
            // Anillo pulsante alrededor del micrófono
            draw_ring(&mut buf, mic_cx, mic_cy_top + 9, 16, 2, palette.overlay);
        }
        TrayState::Processing => {
            // 3 puntos en arco (spinner visual estático)
            for i in 0u32..3 {
                let angle = std::f32::consts::PI * (0.25 + i as f32 * 0.25);
                let dx = (angle.cos() * 10.0) as i32;
                let dy = (angle.sin() * 10.0) as i32;
                let cx = (mic_cx as i32 + dx) as u32;
                let cy = (mic_cy_top as i32 + 9 + dy) as u32;
                fill_circle(&mut buf, cx, cy, 2, palette.overlay);
            }
        }
        TrayState::Paused => {
            // Dos barras verticales (pausa)
            fill_rect(&mut buf, mic_cx - 5, mic_cy_top + 4, 3, 12, palette.overlay);
            fill_rect(&mut buf, mic_cx + 2, mic_cy_top + 4, 3, 12, palette.overlay);
        }
        TrayState::Error => {
            // X sobre el cuerpo del micrófono
            draw_x(&mut buf, mic_cx, mic_cy_top + 9, 7, palette.overlay);
        }
    }

    RgbaIcon {
        data: buf,
        width: WIDTH,
        height: HEIGHT,
    }
}

// ---------------------------------------------------------------------------
// Paleta de colores
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Palette {
    mic: [u8; 4],
    overlay: [u8; 4],
}

fn choose_palette(state: TrayState, theme: Theme) -> Palette {
    let (mic_rgb, overlay_rgb) = match state {
        TrayState::Idle => ([0x00u8, 0x7A, 0xCC], [0xFF, 0xFF, 0xFF]),
        TrayState::Listening => ([0xF4, 0x43, 0x36], [0xFF, 0xFF, 0xFF]),
        TrayState::Processing => ([0xFF, 0x98, 0x00], [0xFF, 0xFF, 0xFF]),
        TrayState::Paused => ([0x80, 0x80, 0x80], [0xFF, 0xFF, 0xFF]),
        TrayState::Error => ([0xB7, 0x1C, 0x1C], [0xFF, 0xFF, 0xFF]),
    };

    // Light mode: overlay negro para contrastar sobre taskbar blanca.
    let resolved_theme = match theme {
        Theme::Dark => Theme::Dark,
        Theme::Light => Theme::Light,
        Theme::System => match dark_light::detect() {
            dark_light::Mode::Light => Theme::Light,
            _ => Theme::Dark,
        },
    };

    let overlay_rgba = match resolved_theme {
        Theme::Light => [0x00u8, 0x00, 0x00, 0xFF],
        _ => [overlay_rgb[0], overlay_rgb[1], overlay_rgb[2], 0xFF],
    };

    Palette {
        mic: [mic_rgb[0], mic_rgb[1], mic_rgb[2], 0xFF],
        overlay: overlay_rgba,
    }
}

// ---------------------------------------------------------------------------
// Primitivas de dibujo
// ---------------------------------------------------------------------------

#[inline]
fn set_pixel(buf: &mut [u8], x: u32, y: u32, color: [u8; 4]) {
    if x >= WIDTH || y >= HEIGHT {
        return;
    }
    let idx = ((y * WIDTH + x) * 4) as usize;
    buf[idx] = color[0];
    buf[idx + 1] = color[1];
    buf[idx + 2] = color[2];
    buf[idx + 3] = color[3];
}

fn fill_rect(buf: &mut [u8], x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    for dy in 0..h {
        for dx in 0..w {
            set_pixel(buf, x + dx, y + dy, color);
        }
    }
}

fn fill_circle(buf: &mut [u8], cx: u32, cy: u32, r: u32, color: [u8; 4]) {
    let r2 = (r * r) as i64;
    let r = r as i32;
    let cx = cx as i32;
    let cy = cy as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as i64 <= r2 {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 {
                    set_pixel(buf, px as u32, py as u32, color);
                }
            }
        }
    }
}

fn draw_ring(buf: &mut [u8], cx: u32, cy: u32, r_outer: u32, thickness: u32, color: [u8; 4]) {
    let r_inner = r_outer.saturating_sub(thickness);
    let ro2 = (r_outer * r_outer) as i64;
    let ri2 = (r_inner * r_inner) as i64;
    let r = r_outer as i32;
    let cx = cx as i32;
    let cy = cy as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as i64;
            if d2 <= ro2 && d2 >= ri2 {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 {
                    set_pixel(buf, px as u32, py as u32, color);
                }
            }
        }
    }
}

/// Semiarco inferior del cuello del micrófono (180° inferior).
fn draw_arc_bottom(buf: &mut [u8], cx: u32, cy: u32, r_outer: u32, r_inner: u32, color: [u8; 4]) {
    let ro2 = (r_outer * r_outer) as i64;
    let ri2 = (r_inner * r_inner) as i64;
    let r = r_outer as i32;
    let cx = cx as i32;
    let cy = cy as i32;
    for dy in 0..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as i64;
            if d2 <= ro2 && d2 >= ri2 {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 {
                    set_pixel(buf, px as u32, py as u32, color);
                }
            }
        }
    }
}

/// X centrada en (cx, cy) con semiancho `half`.
fn draw_x(buf: &mut [u8], cx: u32, cy: u32, half: u32, color: [u8; 4]) {
    let half = half as i32;
    let cx = cx as i32;
    let cy = cy as i32;
    for i in -half..=half {
        // diagonal \
        let (px, py) = (cx + i, cy + i);
        if px >= 0 && py >= 0 {
            set_pixel(buf, px as u32, py as u32, color);
            if px + 1 < WIDTH as i32 {
                set_pixel(buf, (px + 1) as u32, py as u32, color);
            }
        }
        // diagonal /
        let (px2, py2) = (cx - i, cy + i);
        if px2 >= 0 && py2 >= 0 {
            set_pixel(buf, px2 as u32, py2 as u32, color);
            if px2 + 1 < WIDTH as i32 {
                set_pixel(buf, (px2 + 1) as u32, py2 as u32, color);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::TrayState;
    use oido_config::Theme;

    #[test]
    fn buffer_size_is_exactly_width_times_height_times_4() {
        for state in [
            TrayState::Idle,
            TrayState::Listening,
            TrayState::Processing,
            TrayState::Paused,
            TrayState::Error,
        ] {
            let icon = render_state(state, Theme::Dark);
            assert_eq!(
                icon.data.len(),
                (WIDTH * HEIGHT * 4) as usize,
                "estado {:?} tiene tamaño de buffer incorrecto",
                state
            );
        }
    }

    #[test]
    fn idle_icon_dark_has_blue_pixels() {
        let icon = render_state(TrayState::Idle, Theme::Dark);
        let has_blue = icon
            .data
            .chunks(4)
            .any(|p| p[0] == 0x00 && p[1] == 0x7A && p[2] == 0xCC && p[3] == 0xFF);
        assert!(has_blue, "Idle dark debe contener píxeles azules");
    }

    #[test]
    fn listening_icon_has_red_pixels() {
        let icon = render_state(TrayState::Listening, Theme::Dark);
        let has_red = icon
            .data
            .chunks(4)
            .any(|p| p[0] == 0xF4 && p[1] == 0x43 && p[2] == 0x36 && p[3] == 0xFF);
        assert!(has_red, "Listening debe contener píxeles rojos");
    }

    #[test]
    fn dark_and_light_overlay_differ() {
        let dark = render_state(TrayState::Listening, Theme::Dark);
        let light = render_state(TrayState::Listening, Theme::Light);
        assert_ne!(
            dark.data, light.data,
            "Dark y Light deben generar buffers distintos"
        );
    }

    #[test]
    fn error_icon_has_dark_red_pixels() {
        let icon = render_state(TrayState::Error, Theme::Dark);
        let has_dark_red = icon
            .data
            .chunks(4)
            .any(|p| p[0] == 0xB7 && p[1] == 0x1C && p[2] == 0x1C && p[3] == 0xFF);
        assert!(has_dark_red, "Error debe contener píxeles rojo oscuro");
    }
}
