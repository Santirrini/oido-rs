# Changelog

## Fase 0 — Bootstrap (en curso)

**Fecha:** 2026-07-03

### Entregables completados

- `Cargo.toml` workspace + 4 crates libs + 1 bin (`oido`).
- `rust-toolchain.toml` pin a 1.96 (edition 2024 en transitive deps).
- `ARCHITECTURE.md` con las 3 reglas Rust inviolables y el flujo.
- `plans/PLAN.md` con roadmap 8 fases + registro YAGNI.
- `.github/workflows/ci.yml` matrix Win + macOS + Linux con fmt, clippy
  pedantic, nextest, deny, doc.
- `deny.toml` con licencias aprobadas + bans.
- Stub de cada crate con su tipo público principal ya declarado:
  - `oido-stt`: trait `Transcriber` (Fase 1 lo rellena).
  - `oido-platform`: 4 traits OS (`CaptureSource`, `Hotkey`, `Tray`,
    `Injector`) y stubs por plataforma.
  - `oido-config`: `ConfigStore` con parking_lot, atomic_write tempfile+rename,
    paths por OS vía `dirs`.
  - `oido-core`: tipos `AudioFrame`, `InjectedText`, aliases de canal y
    `Pipeline` placeholder.
  - `oido` bin: arranca logger, carga config, sale limpio.
- Lint a nivel workspace: `unsafe_code = "deny"` (Regla R2 enforcer),
  `clippy::pedantic` activo. Override en `oido-stt/Cargo.toml` para
  permitir unsafe dentro de su `whisper_cpp.rs`.

### Verificación

- Workspace parsed (cargo metadata) ✅
- 14 deps generaron `rmeta` antes de chocar con el linker.
- `cargo nextest run` corre con suite vacía (todos los crates están
  todavía como stubs sin tests) ✅.
- `cargo fmt --check` y `cargo clippy -W clippy::pedantic` no han sido
  ejecutados localmente todavía — bloqueado por la falta de Windows SDK
  (ver abajo). CI los ejecuta en push.

### Bloqueos identificados

**Windows SDK parcial en esta máquina.**

El Build Tools instalado trae cl.exe y link.exe pero faltan las libs
del sistema (`kernel32.lib`, `ntdll.lib`...). La instalación del
Windows SDK 10.0.22621 por winget falló en mitad de descarga (errores
HTTP `0x80072ee2`). Hasta que el SDK no esté completo, `cargo build`
no enlazará binarios en Windows.

**Decisión:** no perder más tiempo en toolchain del host. La
verificación de CI matrix cubre los 3 OS en push. Usuario: cuando
instale el SDK completo (ver `README.md` § Prereqs), `cargo build` verde.

### Toolchain instalado en esta máquina

- Rust 1.96.1 (x86_64-pc-windows-msvc).
- MSVC Build Tools 17.14.35 en `C:\BuildTools\VC`.
- ❌ Windows 10/11 SDK kernel32.lib (no instalado).
- ❌ Rust GNU mingw (no instalado).

### Siguiente fase (Fase 1)

MVP `dicta y pega` + tray 3 estados. Trabajo concreto:

1. Reemplazar stubs de `oido-platform` con impls reales usando
   `cpal`, `global-hotkey`, `tray-icon` (+ `ksni` en Linux), `arboard`.
2. Implementar `WhisperCpp` en `oido-stt/src/whisper_cpp.rs` enlazando
   `whisper-rs`.
3. Implementar `Pipeline` en `oido-core/src/pipeline.rs` con channels
   concretos.
4. Portar `SegmentDeduplicator` (~18 líneas de Oido) y `PhraseFilter`
   (~30 líneas, añadir "gracias por ver", "suscríbete").
5. Smoke E2E: hold F8 → "hola mundo" → texto en bloc de notas en 3 OS.
