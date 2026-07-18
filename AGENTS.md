# Code Review Rules

> Estado: 2026-07-17. Refleja la división del antiguo `oido-platform` en
> `oido-audio`/`oido-hotkey`/`oido-input`/`oido-tray` y la adición del backend
> de accesibilidad Windows (`uiautomation`) en `oido-input`. No hay crate
> `oido-platform` ni bin `oido-tauri` en este workspace.

## Rust Rules
- **R1 — Channels for thread communication:** Use `crossbeam::channel` for synchronous pathways (audio → stt → filter) and `tokio::sync::mpsc` for asynchronous communication. Avoid `Arc<Mutex<T>>` or `Arc<RwLock<T>>` outside of `oido-config`.
- **R2 — FFI isolation:**
  - **Written `unsafe`**: solo se permite en `oido-stt/src/whisper_cpp.rs` y en los archivos Win32 de `oido-tray` (`dialog.rs`, `dpi.rs`, `tray/popup_window.rs`, `win_helper.rs`). El resto del workspace es **100% Safe Rust**.
  - **Internal `unsafe` en dependencias**: permitido. Si usamos un crate que internamente hace FFI (`uiautomation`, `accessibility`, `atspi`, `whisper-rs`, `arboard`, `enigo`, `windows-sys`, etc.), no propagamos `unsafe` a nuestro código: lo encapsulamos en wrappers safe y exponemos solo tipos/traits safe al resto del workspace. Los backends de accesibilidad (`oido-input::direct::{windows,macos,linux}`) son el patrón canónico: el estado FFI vive en un thread dedicado y se cruza al resto por canal `crossbeam`.
  - **Traits públicos con FFI subyacente**: `Transcriber` (`oido-stt`) y `DirectInjector` (`oido-input`) deben permanecer 100% safe Rust.
  - **Lints asociados**: `unsafe_code = "deny"` en `[workspace.lints.rust]`; crate `oido-tray/Cargo.toml:11` declara `unsafe_code = "allow"` con comentario justificando la excepción Win32.
- **R3 — Single workspace Mutex:** Use `parking_lot::Mutex` for the workspace mutex in `oido-config::ConfigStore`. Never use `std::sync::Mutex`. Ensure lock acquisitions do not panic; return errors instead of using `.lock().unwrap()`.

## Error Handling
- For crates (`-core`, `-stt`, `-audio`, `-hotkey`, `-input`, `-config`), use `thiserror` with domain-specific enums (e.g., `SttError`, `HotkeyError`, `InjectError`, `ConfigError`).
- For binaries (`oido`), use `anyhow` with `.context()` on any IO/FFI operations that can fail.
- Never use `Result<T, String>` or `panic!` for normal control flow.

## Concurrency and Observability
- Structure tracing events using `tracing` spans instead of printing plain error strings.
- Pass messages via bounded channels with appropriate sizing (e.g., bounded 32 for audio, bounded 8 for text, bounded 8 para jobs de inyección UIA).
- Tipos de terceros que no son `Send + Sync` (e.g. `UIAutomation`/`UIElement` del crate `uiautomation` por atadura a COM apartment): aislarlos en un thread dedicado y exponer solo un handle `Send + Sync` por canal. Ver `oido-input::direct::windows` como referencia.
