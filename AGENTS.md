# Code Review Rules

## Rust Rules
- **R1 — Channels for thread communication:** Use `crossbeam::channel` for synchronous pathways (audio → stt → filter) and `tokio::sync::mpsc` for asynchronous communication. Avoid `Arc<Mutex<T>>` or `Arc<RwLock<T>>` outside of `oido-config`.
- **R2 — FFI isolation:** Restrict `unsafe` and raw pointer operations strictly to `oido-stt/src/whisper_cpp.rs`. The public trait `Transcriber` in `oido-stt/src/lib.rs` must remain 100% safe Rust.
- **R3 — Single workspace Mutex:** Use `parking_lot::Mutex` for the workspace mutex in `oido-config::ConfigStore`. Never use `std::sync::Mutex`. Ensure lock acquisitions do not panic; return errors instead of using `.lock().unwrap()`.

## Error Handling
- For crates (`-core`, `-stt`, `-platform`, `-config`), use `thiserror` with domain-specific enums (e.g., `SttError`, `HotkeyError`, `InjectError`, `ConfigError`).
- For binaries (`oido`, `oido-tauri`), use `anyhow` with `.context()` on any IO/FFI operations that can fail.
- Never use `Result<T, String>` or `panic!` for normal control flow.

## Concurrency and Observability
- Structure tracing events using `tracing` spans instead of printing plain error strings.
- Pass messages via bounded channels with appropriate sizing (e.g., bounded 32 for audio, bounded 8 for text).
