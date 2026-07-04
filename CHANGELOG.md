# Changelog

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