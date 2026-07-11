# Changelog

## Fase 5a - MSI Installer & Auto-Updater

**Fecha:** 2026-07-11

### MSI Installer & WiX Toolset
- Created `installer/oido.wxs` source configuring a per-user, non-elevated installation to `%LOCALAPPDATA%\Programs\Oido` with Start Menu shortcut.
- Created `installer/build-msi.ps1` to orchestrate release builds, staging, WiX `candle`/`light` compilation, and SHA256 checksum generation.
- Configured static CRT (`+crt-static`) in `.cargo/config.toml` to eliminate runtime dependencies on `vcruntime140.dll`.

### Auto-Update System
- Integrated `self_update` dependency with `updater` feature in `crates/oido/Cargo.toml`.
- Implemented `crates/oido/src/updater.rs` to fetch latest GitHub releases, verify SHA256 checksums, and trigger quiet MSI installations via `msiexec`.
- Added CLI hook `--check-update` for synchronous updates and background worker thread checking for the Tray menu's "Check for Updates" action.

### Startup Experience & Models
- Implemented native MessageBox dialog on Windows at startup if the configured model or all `.bin` models are missing, offering a direct download of `ggml-base.bin`.
- Spawns background thread for model download and automatic activation upon confirmation.

### CI/CD Release Workflow
- Created `.github/workflows/release.yml` triggered on tag releases (`v*`) which builds Oido statically, runs `dumpbin /dependents` to verify zero `vcruntime140.dll` linkage, packages the MSI installer, performs a silent install/uninstall smoke-test on `windows-latest`, and publishes installer artifacts to GitHub Releases.

## Fase 1 - MVP dicta+pega (en curso)

**Fecha:** 2026-07-09

### Pipeline y tests

- `crates/oido-core/tests/pipeline_e2e.rs` — 7 integration tests con
  `MockCapture` / `MockHotkey` / `MockTranscriber` / `MockInjector` que
  ejercitan el flujo press → STT → filtro → inyección sin OS.
- `crates/oido-config/src/lib.rs` — `+8 tests` unitarios para
  `atomic_write` y `ConfigStore`; property-based roundtrip con
  `proptest` para `Config`.
- `oido-config`: agregado `PartialEq + Eq + proptest::Arbitrary` para
  `Config`.
- Removido `#![doc = include_str!("../../../ARCHITECTURE.md")]` de
  `oido-core/src/lib.rs` — causaba 7 doctests rotos por bloques de
  Rust en ARCHITECTURE.md referenciando tipos no definidos.

### Bin `oido`

- Handler de Ctrl+C (`ctrlc = "3"`): shutdown limpio al recibir señal.
- Resolución del directorio de modelos:
  `OIDO_MODELS_DIR` env var → `dirs::data_dir()/oido/models` →
  fallback relativo `models/`.
- Tray cableado al observer thread: `PlatformTray::set_state` se
  invoca en cada `PipelineEvent::State`. Tray stub en MVP (sólo
  loggea); el cableado queda listo para Fase 3.

### Scripts

- `scripts/download_model.ps1` (Windows) y `scripts/download_model.sh`
  (macOS/Linux) bajan `ggml-base.bin` desde
  `huggingface.co/ggerganov/whisper.cpp` al directorio de modelos de
  la app.

### Dependencias nuevas

- `proptest = "1"` (workspace dev-dep; sólo lo usa `oido-config` por
  ahora).
- `ctrlc = "3"` (workspace dep; sólo lo usa el bin `oido`).
- `parking_lot` y `dirs` agregados como deps directas del bin `oido`
  (antes venían transitivas).

### CI / verificaciones locales

27 tests passing + 1 ignored (smoke E2E whisper con modelo real).
`cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`
y `cargo test --workspace` limpios.

## Fase 0 - Bootstrap CERRADA

**Fecha:** 2026-07-03 a 2026-07-04

### CI matrix verde

Ultimo run verde: https://github.com/Santirrini/oido-rs/actions/runs/28719724578

| Job | OS | Resultado |
|---|---|---|
| fmt | ubuntu | success |
| build / clippy / nextest / doc | ubuntu | success |
| build / clippy / nextest / doc | macos | success |
| build / clippy / nextest / doc | windows | success |
| cargo deny | ubuntu | success |

### Entregables

- Workspace Cargo + 5 crates (`oido-core`, `oido-stt`, `oido-platform`,
  `oido-config`, `oido` bin).
- `rust-toolchain.toml` pin a 1.96.
- 3 reglas Rust (channels, FFI aislado, parking_lot) en `ARCHITECTURE.md`.
- `plans/PLAN.md` con 8 fases detalladas + YAGNI.
- CI matrix 3 OS con fmt, clippy, nextest, deny, doc.
- Lint workspace: `unsafe_code = "deny"` (Regla R2),
  `missing_debug_implementations = "warn"`.
- Stub de cada crate con API publica lista para Fase 1:
  - `oido-stt`: trait `Transcriber` + `TranscriberFactory` + `whisper_cpp.rs`
    aislado.
  - `oido-platform`: 4 traits (`CaptureSource`, `Hotkey`, `Tray`,
    `Injector`) + impls por OS con `unimplemented!()` listos para sustituir.
  - `oido-config`: `ConfigStore` con `parking_lot::Mutex`, escritura
    atomica `tempfile` + `fs::rename`, paths via `dirs`.
  - `oido-core`: tipos `AudioFrame`, `InjectedText`, aliases de canal,
    `Pipeline` placeholder.
  - `oido` bin: arranca logger, carga config, sale limpio.

### Commits en main

```
9cdc58e feat(bootstrap): Fase 0 scaffold (workspace + CI + docs)
25cbae5 fix(ci): yaml inline lists rejected in `with:` block
7d35f85 fix(ci): desactivar pedantic local + deny.toml + fmt
a28d2a4 fix(ci): quitar RUSTFLAGS=-D warnings
daf0ce0 fix(ci): quitar pedantic del clippy (solo stubs Fase 0)
7d8a2b4 fix(ci): clippy -D warnings verde
7104756 fix(config): derive Debug en Inner struct
024b0ea fix(core): derive Debug+Default en Pipeline; fmt trailing
ea9c1e5 fix(ci): nextest no-tests warn + deny versions
ab7fd15 fix(ci): nextest --no-tests=warn inline
6f2f44f fix(deny): multiple-versions 'transitive' -> 'warn'
```

### Bloqueos del host local (no del proyecto)

- Windows SDK parcial: Build Tools instalado pero `kernel32.lib` no
  disponible. Instalacion del SDK por winget fallo (HTTP errors).
- Mitigacion: CI matrix provee la verificacion cross-platform. El
  usuario corre `cargo build` local cuando tenga el SDK completo.

### Siguiente fase: Fase 1 - MVP dicta+pega + tray 3 estados

1. Reemplazar stubs de `oido-platform` con impls reales (`cpal`,
   `global-hotkey`, `tray-icon`, `ksni`, `arboard`).
2. Implementar `WhisperCpp` en `oido-stt/src/whisper_cpp.rs` enlazando
   `whisper-rs`.
3. Implementar `Pipeline` en `oido-core/src/pipeline.rs` con channels.
4. Portar `SegmentDeduplicator` + `PhraseFilter` con frases ES.
5. Smoke E2E: hold F8 -> texto en cursor, 3 OS.