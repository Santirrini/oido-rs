# ARCHITECTURE.md — Oido 2.0

> Resumen ejecutivo de arquitectura. Decisiones detalladas en
> [`plans/PLAN.md`](plans/PLAN.md). Este doc vive en la raíz y es lectura
> obligatoria para PR review.

## Una frase

`Audio (cpal) → STT (whisper-rs) → Filtro (dedup + phrase) → Inyección
(arboard + paste)`, orquestado por `oido-core` con channels. Sin estado
compartido mutable entre etapas.

## Reglas Rust inviolables

Estas tres reglas son **contrato del proyecto**. Si una contribución las
viola, el PR se cierra sin discusión hasta reformular.

### R1 — Sólo channels entre threads

- `crossbeam::channel` para el camino síncrono (audio → stt → filtro).
- `tokio::sync::mpsc` para async (UI / comandos Tauri).
- **Prohibido** `Arc<Mutex<T>>` o `Arc<RwLock<T>>` fuera de `oido-config`, con la única excepción autorizada de `Arc<Mutex<BufferState>>` en `oido-core` (necesaria para acumular las muestras de audio desde el callback de captura de `cpal` y sincronizarlo con el ciclo de vida del pipeline).
- Cada etapa posee su estado. La composición es `Sender → Receiver` entre
  threads workers.
- Si descubres que necesitas estado compartido, el problema está mal
  planteado. Replantea antes de añadir un mutex.

```rust
// oido-core/src/pipeline.rs (esquema)
let (audio_tx, audio_rx) = crossbeam::channel::bounded::<AudioFrame>(32);
let (text_tx, text_rx) = crossbeam::channel::bounded::<String>(8);

// Cada worker en su thread, conectado por channels:
std::thread::spawn(move || capture::run(audio_tx));
std::thread::spawn(move || stt::run(audio_rx, text_tx));
std::thread::spawn(move || filter::run(text_rx, inject_tx));
```

### R2 — FFI aislada en una unidad

- `unsafe` y punteros crudos sólo en `oido-stt/src/whisper_cpp.rs`.
- El trait `Transcriber` (`oido-stt/src/lib.rs`) es la única superficie
  expuesta y es 100% safe Rust.
- `cargo geiger` debe reportar 0 usos de unsafe fuera de `oido-stt`.
- Si necesitas unsafe fuera de `oido-stt`, primero justifica en este
  archivo con un párrafo y consigue aprobación de review.

```rust
// oido-stt/src/lib.rs
pub trait Transcriber: Send {
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError>;
    fn load(model_path: &Path) -> Result<Self, SttError>
    where Self: Sized;
}

// oido-stt/src/whisper_cpp.rs   (único archivo con unsafe)
struct WhisperImpl { ctx: NonNull<whisper_context> }
unsafe impl Send for WhisperImpl { } // justificado en doc-comment
impl Transcriber for WhisperImpl { /* wrap a FFI */ }
```

### R3 — `parking_lot::Mutex` para el único mutex del workspace

- `oido-config::ConfigStore` envuelve `parking_lot::Mutex<Inner>`.
- Cero `std::sync::Mutex` en el workspace.
- Lock acquisition nunca pánica: si quieres panic explícito, devuélvelo
  como `Err(Global)`; nunca `.lock().unwrap()`.
- Tests usan `parking_lot::Mutex` también, para forzar patrón consistente.

```rust
// oido-config/src/store.rs (esquema)
use parking_lot::Mutex;

pub struct ConfigStore {
    inner: Mutex<Inner>,
}

impl ConfigStore {
    pub fn with<R>(&self, f: impl FnOnce(&Inner) -> R) -> R {
        f(&self.inner.lock())
    }
}
```

## Mapa de crates

> **Refactor modular profundo (Fase 6):** el antiguo crate monolítico
> `oido-platform` se dividió en 4 crates granulares por dominio
> (audio, hotkey, input, tray). Cada uno tiene una sola razón de
> cambio. `oido-updater` se extrajo como crate hermano. `oido-stt`
> quedó puramente dedicado a STT (sin código de UI).

```
                       ┌────────────────────────────┐
                       │            bin             │
                       │ oido (CLI) + oido-tauri    │
                       └──────────┬────────┬────────┘
                                  │        │
              ┌───────────────────┘        └───────────────────┐
              ▼                                               ▼
       ┌──────────────────┐                          ┌──────────────────┐
       │   oido-core      │                          │   oido-config    │
       │ pipeline + filter│                          │ ConfigStore +    │
       │ + state events   │                          │ atomic write     │
       └──────┬───────────┘                          └──────────────────┘
              │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
              ├─►│ oido-audio  │  │ oido-hotkey │  │ oido-input  │
              │  │ 100% Safe   │  │ 100% Safe   │  │ 100% Safe   │
              │  │  Capture +  │  │  Hotkey +   │  │  Injector   │
              │  │ Resampler   │  │  GatedHotkey│  │ (arboard)   │
              │  └─────────────┘  └─────────────┘  └─────────────┘
              │  ┌─────────────────────────────────────────────┐
              ├─►│              oido-stt                      │
              │  │  WhisperCpp (FFI aislado, R2) + LocalAgre- │
              │  │  ement streamer. Sin UI, sin hotkey, sin   │
              │  │  tray.                                     │
              │  └─────────────────────────────────────────────┘
              │  ┌─────────────────────────────────────────────┐
              └─►│              oido-tray                     │
                 │  Único crate con `unsafe` fuera de oido-stt│
                 │  (popup GDI Win32 + MessageBox + DPI).     │
                 │  Bandeja + popup + icon + dialog + helpers │
                 │  Win32 UI (`pump_windows_message_loop`,    │
                 │  `set_windows_menu_theme`).                │
                 └─────────────────────────────────────────────┘

       ┌──────────────────┐         ┌──────────────────┐
       │  oido-models     │         │  oido-updater    │
       │  catálogo +      │         │  (hermano)       │
       │  descarga HF     │         │  self_update +   │
       └──────────────────┘         │  reqwest + sha2  │
                                    └──────────────────┘
```

Dependencias permitidas entre crates:

```
oido-core          ──> oido-{audio, hotkey, input} (traits)
oido-core          ──> oido-stt (trait)
oido-core          ──> oido-config (read config al boot)
oido-stt           ──> (sólo stdlib + whisper-rs)        [post-Fase 6]
oido-audio         ──> oido-config (types), cpal, rubato
oido-hotkey        ──> oido-config (types), rdev, global-hotkey
oido-input         ──> (arboard + enigo, sin deps internas)
oido-tray          ──> oido-config, oido-models, muda, tray-icon, ksni, dark-light
oido-config        ──> (sólo stdlib + serde + dirs + tempfile)
oido-models        ──> oido-config, reqwest, sha2, hex
oido-updater       ──> (self_update, reqwest, sha2, hex, serde)
bin (oido)         ──> oido-{core, config, models, audio, hotkey, input, tray, stt}
                      oido-updater? (optional, gate feature `updater`)
```

**Aislamiento de `unsafe` (R2):** tras el refactor, los únicos lugares
del workspace con bloques `unsafe` son:

1. `oido-stt/src/whisper_cpp.rs` — FFI whisper.cpp + helpers Win32.
2. `oido-tray/src/dialog.rs` — `MessageBoxW` Win32.
3. `oido-tray/src/dpi.rs` — `SetProcessDpiAwarenessContext` Win32.
4. `oido-tray/src/tray/popup_window.rs` — ventana Win32 borderless + GDI.
5. `oido-tray/src/win_helper.rs` — `PeekMessageW`/`DispatchMessageW`
   y `SetPreferredAppMode` (loadlibrary + GetProcAddress por ordinal).

Todos llevan `#![allow(unsafe_code)]` a nivel de archivo. `cargo
geiger` debe confirmar este set; cualquier otro `unsafe` requiere
justificación en este `ARCHITECTURE.md` + aprobación de review.

No se permite dependencia cíclica. Si la piensas, refactoriza.

## Frontend (Tauri + Svelte)

- Tauri 2.x como shell.
- Svelte 5 + TypeScript para UI.
- IPC tipado con `tsify` (R exporta tipos TypeScript automáticamente desde
  structs `serde`).
- Comandos Tauri viven en `ipc/src/commands.rs` y se reexportan en
  `ui/src/lib/tauri.ts`.
- `cargo nextest` prueba los comandos. UI sólo se smoke-testea manual al
  cerrar cada fase.

## Estado de los modos de dictado

El enum `SttMode` (en `crates/oido-config/src/lib.rs`) admite tres variantes, pero **solo `Batch` es estable y de uso diario**. Las otras dos se conservan accesibles para experimentación y se señalizan explícitamente como "en prueba" — tanto en el menú nativo de bandeja (sufijo `· en prueba`) como en un `tracing::warn!` en el arranque.

| Modo       | Pipeline                                | Estado     | Notas para developers                                                                          |
|------------|-----------------------------------------|------------|-------------------------------------------------------------------------------------------------|
| `Batch`    | `oido-core::pipeline::Pipeline`         | ✅ Estable | Default (`SttMode::Batch`). Hold-to-talk clásico. MVP original; base recomendada.               |
| `Streaming`| `oido-core::streaming_pipeline::…`      | 🧪 En prueba | LocalAgreement-2 con ticker de 1s. Verificado en su openspec pero no es uso diario.            |
| `Chunked`  | `oido-core::chunked_pipeline::…`        | 🧪 En prueba | Bloques de ~5s con timestamps por palabra. Carryover eliminado por races (commit 95bbb9d).    |

Cómo se señaliza:

- **UI de usuario:** `crates/oido-tray/src/tray/i18n.rs` añade los sufijos `· estable`, `· en prueba` (y variantes bilingüe/EN) a los labels del submenú "Modo de dictado". Los items siguen siendo seleccionables; no se deshabilitan.
- **Runtime:** en `crates/oido/src/main.rs`, dentro del `match mode` que arranca el pipeline, las ramas `Streaming` y `Chunked` emiten un `tracing::warn!(mode = …)` la primera vez que se construyen (tanto en arranque como en cambio en caliente vía menú).
- **Docs:** el doc-comment de `SttMode` lista el estado de cada variante; esta tabla es la referencia canónica.

**Importante:** no hay Cargo feature flags ni gating de compile-time. El gating es puramente señalización (label + log). Quien edite `config.json` a mano puede seguir activando cualquier modo.

```
                ┌────────────────────────────────────────────────────────────┐
                │                  Tauri UI / commands                       │
                └────────────────────────────────────────────────────────────┘
                                       │  reads
                                       ▼
                              ┌──────────────────┐
                              │   oido-config    │
                              │   ConfigStore    │
                              └──────────────────┘
                                       ▲  writes (atomic, parking_lot)
                                       │
                ┌──────────────────────┴───────────────────────────┐
                │                                                 │
                │  crossbeam::channel audio (bounded 32)          │
                │                                                 │
[global-hotkey]──on/off──►[CaptureSource]──────────────────────►[WhisperCpp]
                (cpal)                              audio batch  (whisper-rs)
                                                                          │
                ┌─────────────────────────────────────────────────┐
                │  crossbeam::channel text (bounded 8)            │
                │                                                 │
                │   output ──►[SegmentDeduplicator]──►[PhraseFilter]
                │                              (anti-hallucination)
                │                                                 │
                │                                                 ▼
                │                                    [arboard + paste]
                │                                                 │
                │                                                 ▼
                │                                            clipboard
                │                                       + Ctrl/Cmd+V send
                └─────────────────────────────────────────────────┘
                                       │
                                       ▼
                              tracing events ◄────── todo el pipeline
                                       │
                                       ▼
                            Tauri (estado visible UI / tray)
```

## Tipos de error

- **Crates (`-core`, `-stt`, `-platform`, `-config`):** `thiserror` con enums
  por dominio (`SttError`, `HotkeyError`, `InjectError`, `ConfigError`).
- **Bins (`oido`, `oido-tauri`):** `anyhow` con `.context()` en cada
  operación de IO/FFI que puede fallar.
- Nunca `Result<T, String>`. Nunca `panic!` para control de flujo.

```rust
// crate ejemplo
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("audio buffer too short: {0} samples")]
    AudioTooShort(usize),
    #[error("whisper backend error: {0}")]
    Backend(String),
}
```

## Concurrencia — guía rápida

| Caso | Solución |
|------|----------|
| Audio thread → STT thread | `crossbeam::channel` bounded |
| Hotkey callback (no bloquear) | callback encola evento; procesamiento en thread dedicado |
| UI lee estado | `ConfigStore::with(\|c\| c.snapshot())` |
| UI pide reload config | command Tauri → bin publica en canal de comando |
| Tray pide "show panel" | tray emite evento; UI handler en Svelte |
| Cancelar descarga modelo | `tokio::sync::oneshot` o `tokio_util::sync::CancellationToken` |

Si tu caso no está aquí, es señal de que la abstracción no encaja.

## Observabilidad

- `tracing` en todo el pipeline (spans por etapa).
- `tracing-subscriber` con env filter (`OIDO_LOG=debug` para verbose).
- Logs a archivo rotado en dir configurado por `oido-config`.
- Errores estructurados como eventos, no strings.

## Testing — guía rápida

| Qué                                                       | Cómo                                          |
|-----------------------------------------------------------|-----------------------------------------------|
| Pure function (`text filter`, `dedup`)                    | `#[test]` en `tests/` del módulo              |
| Lectura / escritura config                                | `proptest` roundtrip JSON                      |
| Pipeline end-to-end sin OS                                | inyecta trait mock para `CaptureSource`/`Injector`; audio sintético |
| FFI whisper                                              | smoke test real; no mockeamos FFI             |
| Comportamiento tray / hotkey                              | integration test con trait mock en `oido-platform` |
| Estados accesibles / i18n                                 | smoke manual + `axe-core` en CI               |

## Decisiones tomadas (registro inmutable)

| ID | Decisión | Razón | Fecha |
|----|----------|-------|-------|
| D1 | Rust + Tauri + whisper-rs | Mejor trade-off cross-platform / native / IA conocido | 2026-07-03 |
| D2 | Modelo default base multilingüe q5_1 | tiny.en no soporta ES; base cubre target | 2026-07-03 |
| D3 | Svelte 5 (no React) | Bundle pequeño, sintaxis simple | 2026-07-03 |
| D4 | Tray en Fase 1 (no Fase 2) | Sin feedback visual MVP no es usable | 2026-07-03 |
| D5 | rust-i18n + i18n día 1 | Barato temprano, caro tarde | 2026-07-03 |
| D6 | Portar dedup + phrase-filter + atomic_io | ~100 líneas resuelven problemas reales | 2026-07-03 |
| D7 | Cursor animado drop | Tray overlay comunica igual, 10× menos código | 2026-07-03 |
| D8 | Virtual desktop drop | Win-only, nadie lo usa | 2026-07-03 |
| D9 | Perf tuner beam grid drop | whisper.cpp ya auto-ajusta | 2026-07-03 |
| D10 | IPC JSON files + polling drop | Race conditions + 200ms poll, reemplazado por Tauri nativo | 2026-07-03 |
| D11 | paridad fases 1-8 todas críticas | Decisión de usuario 2026-07-03 | 2026-07-03 |

## Decisiones pendientes

| ID | Pregunta | Bloqueante para | Resolver en |
|----|----------|-----------------|-------------|
| P1 | Repo: nuevo en GitHub vs local primero | Fase 0 | ahora |
| P2 | Mismo proceso Tauri vs sidecar | Fase 3 | Fase 3 kickoff |
| P3 | Apple Developer account disponible | Fase 5 | ahora |
| P4 | Cert Win EV/OV tramitado | Fase 5 | ahora |
| P5 | Activar Fase 6 (translate) | post-Fase 5 | revisión post-Fase 5 |
