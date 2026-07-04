# oido-rs

Rewrite of [Oido](https://github.com/Santirrini/Oido) in Rust. Local-first,
cross-platform voice dictation: hold a hotkey, speak, release, text appears at
the cursor.

**Status:** Fase 0 (bootstrap). The workspace compiles on Windows, macOS and
Linux but no feature is implemented yet. See `plans/PLAN.md` for the full
roadmap.

## Goals

- **100% local** STT via whisper.cpp (quantized ggml models).
- **Cross-platform** desktop: Windows, macOS, Linux (X11 + Wayland in Fase 8).
- **One-direction pipeline** (see `ARCHITECTURE.md`): no shared mutable state
  between stages.
- **Accessibility:** WCAG 2.1 AA in Fase 7.
- **i18n:** ES + EN day 1.

## Layout

```
crates/
  oido-core/       pipeline + dedup + phrase filter
  oido-stt/        trait Transcriber + WhisperCpp impl (FFI aislado)
  oido-platform/   OS traits (CaptureSource, Hotkey, Tray, Injector)
  oido-config/     atomic JSON config
  oido/            bin
plans/             planning docs (PLAN.md)
assets/            iconos tray, .strings ES/EN  (entradas en Fase 3)
models/            ggml *.bin (gitignored, descarga manual con scripts/)
```

Architecture rules: `ARCHITECTURE.md`.

## Prereqs (host)

- **Rust 1.85+** (`rustup install 1.96` o pin del `rust-toolchain.toml`).
- **C toolchain** (whisper.cpp + cpal lo exigen):
  - **Windows (MSVC):**
    1. [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-studio-build-tools/),
       workload **"Desarrollo para escritorio con C++"**.
       Incluye MSVC `cl.exe` + `link.exe` + Windows SDK.
    2. CMake (`winget install Kitware.CMake`).
    3. *Test:* `cargo build` debe encontrar `kernel32.lib`.

       Si ves `LNK1181 no se puede abrir el archivo de entrada 'kernel32.lib'`
       es que no se instaló el Windows SDK: abre el instalador de VS Build
       Tools y añade el componente **Windows 11 SDK** o **Windows 10 SDK**.
  - **macOS:** `xcode-select --install`, `brew install cmake`.
  - **Linux (Ubuntu/Debian):** `sudo apt install build-essential cmake
    libasound2-dev libgtk-3-dev libxdo-dev libdbus-1-dev libxkbcommon-dev
    libayatana-appindicator3-dev pkg-config`.

- **(Opcional Fase 0+)** `cargo nextest` para tests: `cargo install
  cargo-nextest --locked`.
- **(Opcional Fase 0+)** `cargo deny` para advisories + licenses: `cargo
  install cargo-deny --locked`.

## Build

```sh
cargo build           # debug
cargo build --release # optimizado
```

Cross-compile a las 4 targets de la CI matrix (ver
`.github/workflows/ci.yml`).

## Verifications

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
cargo deny check
cargo doc --workspace --no-deps
```

## Status per fase

| Fase | Estado            | Entregable                                       |
|------|-------------------|--------------------------------------------------|
| 0    | ✅ scaffold       | workspace compila 3 OS, CI verde                 |
| 1    | ⏳ no iniciado    | MVP dicta+pega + tray 3 estados                  |
| 2    | ⏳                | hotkey configurable, reload config               |
| 3    | ⏳                | Tauri panel + wizard + i18n ES/EN                |
| 4    | ⏳                | Model downloader                                  |
| 5    | ⏳                | Auto-update + signing                            |
| 6    | ⏳ (condicional)  | Translation popup                                |
| 7    | ⏳                | Accesibilidad WCAG 2.1 AA                        |
| 8    | ⏳                | Wayland + paquetes nativos                       |

## License

MIT OR Apache-2.0.
