# oido-rs

Rewrite of [Oido](https://github.com/Santirrini/Oido) in Rust. Local-first,
cross-platform voice dictation: hold a hotkey, speak, release, text appears at
the cursor.

**Status:** Fase 1 (MVP dicta+pega, en progreso). El pipeline core funciona
sobre traits mocks (`crates/oido-core/tests/pipeline_e2e.rs`); el bin real
compila y queda a la espera de un modelo ggml para STT real. Ver
`plans/PLAN.md` para el roadmap completo.

## Quick start (4 pasos desde clonar)

```sh
# 1. Toolchain y deps de sistema (ver sección Prereqs más abajo).
rustup show                                       # >= 1.85

# 2. Compilar (tarda la primera vez por whisper.cpp).
cargo build --release

# 3. Bajar el modelo STT (~140 MB para ggml-base.bin).
#    Windows (PowerShell):
.\scripts\download_model.ps1
#    macOS / Linux:
./scripts/download_model.sh

# 4. Correr. Mantené F8, dictá, soltá. Ctrl+C para salir.
cargo run --release -p oido
```

El modelo se guarda bajo `%APPDATA%\oido\models\` (Win) o
`~/.local/share/oido/models/` (Linux) o
`~/Library/Application Support/oido/models/` (macOS). Override con la env
var `OIDO_MODELS_DIR` o pasale `-TargetDir` / primer argumento al script.

## Windows Installation

Oido is distributed on Windows as a native MSI installer.

### Downloading & Installing
1. Download the latest `.msi` file from the [GitHub Releases](https://github.com/Santirrini/oido-rs/releases).
2. Double-click the installer. It will install Oido locally under `%LOCALAPPDATA%\Programs\Oido` without requiring administrator privileges (no UAC elevation).
3. The installer automatically adds a **Start Menu shortcut** named `Oido` so you can launch the app easily.

### Microsoft Defender SmartScreen
> [!NOTE]
> Since the MSI installer is not signed with a commercial EV Code Signing certificate, you may see a Microsoft Defender SmartScreen warning ("Windows protected your PC") on first run.
> 
> To bypass this:
> 1. Click on **More info** inside the SmartScreen dialog.
> 2. Click the **Run anyway** button to proceed with the installation.

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

Las 5 puertas que la CI corre en cada push. Las primeras 4 andan sin
`nextest` instalado; la quinta necesita `cargo install cargo-nextest --locked`.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                 # ó cargo nextest run --workspace
cargo deny check                       # ó cargo install cargo-deny --locked
cargo doc --workspace --no-deps
```

Cobertura actual de tests:

- Unit: `oido-core` (dedup, phrase_filter), `oido-stt` (whisper_cpp
  ramas de error), `oido-config` (atomic_write + Config roundtrip
  con proptest).
- Integration: `crates/oido-core/tests/pipeline_e2e.rs` ejercita el
  pipeline completo con mocks para `CaptureSource`/`Hotkey`/
  `Transcriber`/`Injector` — no necesita audio ni modelo.
- FFI whisper: smoke test marcado `#[ignore]`; activalo con
  `cargo test -p oido-stt -- --ignored` cuando tengas
  `models/ggml-base.bin`.

## Status per fase

| Fase | Estado            | Entregable                                       |
|------|-------------------|--------------------------------------------------|
| 0    | ✅ scaffold       | workspace compila 3 OS, CI verde                 |
| 1    | ✅ core           | MVP dicta+pega + tray stub; pipeline con mocks   |
| 2    | ⏳                | hotkey configurable, reload config               |
| 3    | ⏳                | Tauri panel + wizard + i18n ES/EN                |
| 4    | ⏳                | Model downloader (scripts de Fase 1 como base)   |
| 5    | ⏳                | Auto-update + signing                            |
| 6    | ⏳ (condicional)  | Translation popup                                |
| 7    | ⏳                | Accesibilidad WCAG 2.1 AA                        |
| 8    | ⏳                | Wayland + paquetes nativos                       |

### Modos de dictado

Solo **Batch** es estable y se recomienda para uso diario. **Streaming** y
**Chunked** están marcados como `en prueba` en el menú de bandeja y emiten
un `tracing::warn!` al cargarse; siguen siendo seleccionables para
experimentación. Detalle en `ARCHITECTURE.md` → "Estado de los modos de
dictado".

## License

MIT OR Apache-2.0.
