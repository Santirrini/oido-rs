# PLAN.md — Oido 2.0

> Estado: Fase 5a (Auto-updater & MSI Installer para Windows) completada.

## Visión

Dictado por voz **100% local**, **cross-platform** (Windows, macOS, Linux),
paridad funcional con [Oido](https://github.com/Santirrini/Oido). Sin servicios
en la nube para el núcleo (STT, inyección). Conexión opcional sólo si el usuario
configura traducción por API.

## Principios arquitectónicos

1. **Local primero.** STT y traducción offline por defecto. Red = opt-in.
2. **Una dirección de flujo.** `Audio → STT → Filtro → Inyección`. Cero estado
   mutable compartido entre etapas. Channels únicamente.
3. **Ladder ponytail.** Speculative need → skip. Stdlib antes que dependencia.
   Dependencia antes que código propio. Una línea antes que cincuenta.
4. **Una responsabilidad por crate.** Si un crate importa >2 traits externos,
   partir.
5. **Cada fase es shippable.** Al cerrar una fase, el binario corre en los 3 OS.
6. **Calidad no negociable.** `nextest` verde, `clippy::pedantic` limpio,
   `cargo deny` sin advisories, smoke manual en Win + 1 unix por fase.

## Stack (versiones a fijar en Fase 0)

| Capa              | Tecnología                                | Versión pin |
|-------------------|-------------------------------------------|-------------|
| Lenguaje backend  | Rust                                      | edición 2021, MSRV 1.75 |
| STT               | whisper.cpp vía `whisper-rs`              | última stable |
| Audio             | `cpal`                                    | última stable |
| Hotkey global     | `global-hotkey`                           | última stable |
| Tray              | `tray-icon` (Win/mac) + `ksni` (Linux D-Bus) | última stable |
| Clipboard / paste | `arboard` + mapeo OS Ctrl/Cmd+V           | última stable |
| Concurrencia sync | `crossbeam`                               | 0.13 |
| Runtime async     | `tokio`                                   | 1.x |
| Mutex             | `parking_lot` (sólo `oido-config`)        | 0.12 |
| Errores           | `thiserror` (crates) + `anyhow` (bin)     | 1.x / 1.x |
| Logging           | `tracing` + `tracing-subscriber`          | 0.1 |
| Serialización     | `serde` + `serde_json`                    | 1.x |
| Paths             | `dirs`                                    | 5.x |
| HTTP (downloader) | `reqwest` con `rustls`                    | 0.12 |
| Atomic write      | `tempfile`                                | 3.x |
| UI shell          | Tauri                                     | 2.x |
| UI front          | Svelte 5 + TypeScript                     | 5.x |
| i18n              | `rust-i18n` (back) + `svelte-i18n` (front) | 3.x |
| Updater           | `tauri-plugin-updater`                    | 2.x |
| IPC               | Tauri commands + eventos tipados          | — |

## Reglas Rust inviolables

Estas tres reglas viven también en `ARCHITECTURE.md`. PR que las viola = bloqueado.

### Regla 1 — Sólo channels entre threads

`crossbeam::channel` para el camino síncrono (audio → stt). `tokio::sync::mpsc`
para async (UI / Tauri). **Prohibido `Arc<Mutex<T>>` fuera de `oido-config`.**

Si necesitas compartir estado mutable entre etapas, replantea: probablemente
no lo necesitas; el estado de cada etapa es local o pasa por canal.

### Regla 2 — FFI aislada en una sola unidad

`unsafe` y punteros crudos sólo en `oido-stt/src/whisper_cpp.rs`. El trait
`Transcriber` es la única superficie y es 100% safe Rust. Ningún crate aparte
de `oido-stt` debe incluir `unsafe`.

### Regla 3 — `parking_lot::Mutex` (no `std::sync`)

Nunca `std::sync::Mutex` en el workspace. `parking_lot::Mutex` no envenena.
Único sitio donde se permite un mutex: `oido-config::ConfigStore`. En cualquier
otro punto, refactoriza hacia channel.

## Estructura del workspace

```
oido/
├── crates/
│   ├── oido-core/        # pipeline + dedup + phrase-filter + orch
│   │   └── src/{lib,pipeline,dedup,phrase_filter,state}.rs
│   ├── oido-stt/         # trait Transcriber + impl WhisperCpp (whisper-rs)
│   │   └── src/{lib,whisper_cpp,vad,registry}.rs
│   ├── oido-platform/    # traits OS + impls por plataforma
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs   # CaptureSource, Hotkey, Tray, Injector
│   │       └── {windows,macos,linux}/
│   └── oido-config/      # atomic JSON, schema, paths, ConfigStore
│       └── src/{lib,atomic,paths,schema,store}.rs
├── ui/                   # Tauri + Svelte + TS
│   └── src/{routes,lib,locales}/
├── ipc/                  # tipos compartidos core↔ui (serde + tsify)
├── assets/               # iconos tray (idle/listening/proc) + .strings ES/EN
├── models/               # .gitignore: ggml *.bin
├── plans/                # PLAN.md + sub-planes por fase
│   └── PLAN.md
├── .github/workflows/    # CI matrix (Win + macOS + Linux)
├── ARCHITECTURE.md
├── Cargo.toml            # workspace
├── rust-toolchain.toml   # MSRV pin
└── README.md
```

## Flujo de datos

```
                    ┌───────── Tauri command (state read) ─────────┐
                    ▼                                                │
[global-hotkey F8] →[gate]→ [cpal CaptureSource] →crossbeam→ [WhisperCpp]
                                                                     │
                                                          crossbeam  │
                                                                     ▼
                                            [SegmentDeduplicator] → [PhraseFilter]
                                                                     │
                                                          crossbeam  │
                                                                     ▼
                                                  [arboard Injector] → Ctrl/Cmd+V
                                                                     │
                                          tracing event "injected" ───┘
```

- Hotkey = puerta on/off. Audio graba mientras está held.
- Release → STT proceso batch → filtro → inyección.
- Sin streaming complejo en MVP. Streaming como optimización en Fase 8 si
  medimos latencia como problema real.

## Estrategia de testing

| Tipo                | Dónde                       | Cuándo                                | Cantidad       |
|---------------------|-----------------------------|---------------------------------------|----------------|
| Self-check          | cada crate no trivial       | siempre                               | 1-3 por crate  |
| Property test       | `oido-config`               | roundtrip JSON                        | 1              |
| Integración         | `oido-core/tests/`          | audio vacío → ""; "Thank you" descartado | 2-3         |
| Mock platform       | `oido-platform`             | trait mock, no OS real                | 1 por trait    |
| E2E smoke           | script manual               | fin de cada fase                      | hold F8 → texto |
| Sin mocks innecesarios | —                         | si necesitas `_Fake*`, abstracción mala | —           |

Reglas:
- Trivy (getter, formato string) no necesita test.
- Rama / loop / parser / money / security → 1 check mínimo.
- `tracing-test` para asserts sobre eventos emitidos.

## CI matrix (Fase 0)

```yaml
# .github/workflows/ci.yml (esquema)
matrix:
  os: [windows-latest, macos-latest, ubuntu-latest]
  target:
    - x86_64-pc-windows-msvc
    - aarch64-apple-darwin
    - x86_64-apple-darwin
    - x86_64-unknown-linux-gnu
steps:
  - cargo fmt --check
  - cargo clippy -W clippy::pedantic -- -D warnings
  - cargo nextest run
  - cargo deny check
  - cargo doc --no-deps --quiet
```

## Fases (paridad total — todas críticas)

### Fase 0 — Bootstrap (1-2 días)

**Objetivo:** workspace compila en 3 OS, CI verde, doc de arquitectura escrito.

Tareas:
- `cargo new --workspace` + 4 crates vacíos con `lib.rs` stub
- `rust-toolchain.toml` MSRV 1.75
- Tauri init vacío con template Svelte
- `.gitignore` (`target/`, `models/*.bin`, `ui/node_modules/`)
- `ARCHITECTURE.md` con 3 reglas Rust + flujo de datos
- CI `ci.yml` funcionando (build vacío verde en 3 OS)
- `README.md` mínimo (qué es, cómo build, prerequisitos toolchain)

**DoD:**
- `cargo build` verde en Win + macOS + Linux.
- CI verde en push.
- Repo creado en GitHub con `ARCHITECTURE.md`, `plans/PLAN.md`, `.github/workflows/ci.yml`.

**Riesgos:**
- `cpal` y `whisper-rs` requieren CMake + compilador C en Windows. Mitigar:
  documentar toolchain en README Fase 0 (instaladores Visual Studio Build Tools).

**Checkpoint:** ¿CI cruza los 3 OS? ¿Toolchain documentado?

---

### Fase 1 — MVP "dicta y pega" + tray (1-2 semanas)

**Objetivo:** hold F8 → hablas → sueltas → texto aparece en cursor activo.
Tray con 3 estados (idle / listening / procesando).

Tareas:
- `oido-config`:
  - `ConfigStore` (`parking_lot::Mutex`, único mutex del workspace)
  - escritura atómica (`tempfile` + `fs::rename`, mismo directorio)
  - paths por OS vía `dirs` (`%APPDATA%/Oido`, `~/Library/Application Support/Oido`, `~/.config/oido`)
  - defaults hardcoded por ahora
- `oido-platform`:
  - traits `CaptureSource`, `Hotkey`, `Tray`, `Injector`
  - impls mínimas por OS (`windows/`, `macos/`, `linux/`)
  - `cpal` capture, `global-hotkey` F8, `tray-icon` + `ksni`, `arboard` + paste
- `oido-stt`:
  - trait `Transcriber` (safe API)
  - impl `WhisperCpp` vía `whisper-rs`
  - carga modelo base q5_1 desde `models/`
- `oido-core`:
  - pipeline (`crossbeam::channel` audio → stt → inject)
  - `SegmentDeduplicator` (port Oido)
  - `PhraseFilter` (port Oido + frases ES: "gracias por ver", "suscríbete")
- bin `oido` (CLI sin Tauri todavía):
  - carga config, levanta pipeline, registra hotkey + tray
- modelos:
  - `ggml-base.bin` q5_1 en `models/` (gitignored)
  - `scripts/fetch-model.ps1` + `scripts/fetch-model.sh`

**DoD:**
- Win: hold F8 → dices "hola mundo" → aparece en bloc de notas.
- macOS / Linux: mismo flujo.
- Hold F8 + silencio → no inyecta nada (PhraseFilter descarta alucinación).
- Tray cambia idle → listening → procesando → idle.
- `cargo nextest run` verde.
- `cargo clippy -W clippy::pedantic -- -D warnings` limpio.
- `cargo deny check` sin advisories.
- Smoke manual en Win + 1 unix, registrado en CHANGELOG de fase.

**Riesgos:**
- macOS hotkey requiere entitlement Accessibility. Mitigar: documentar
  paso manual `tccutil reset Accessibility` + Info.plist con
  `NSAppleEventsUsageDescription`.
- Linux Wayland: `global-hotkey` no captura sin portal XDG. Mitigar: Fase 1
  soporta X11 únicamente; Wayland entra en Fase 8.
- whisper-rs build Windows MSVC requiere `cmake` en PATH. Documentar en README.

**Checkpoint:** ¿latencia <3s para frase corta (≤3 palabras)? ¿Tray
refleja estado correctamente en los 3 OS?

---

### Fase 2 — Config + hotkey configurable (1 semana)

**Objetivo:** hotkey y modelo configurables sin tocar código. Recarga en caliente.

Tareas:
- `oido-config`: schema `Config { hotkey, model, language_ui, ... }`, validación
- bin lee hotkey desde config (default F8)
- `ConfigStore::reload()` con notification vía channel de comando al pipeline
- estado persistente: último modelo usado, última hotkey

**DoD:**
- Editar JSON config → reiniciar bin → hotkey/modelo activos.
- Roundtrip JSON property test verde (`oido-config`).
- Validación rechaza config inválida (tecla mal formada, modelo inexistente).

---

### Fase 3 — Tauri panel + wizard + i18n (2 semanas)

**Objetivo:** UI settings (hotkey, modelo, idioma UI). Wizard primer arranque.
i18n ES/EN.

Tareas:
- Tauri 2.x integrado. Bin mismo proceso o sidecar (decidir al inicio).
- `ipc/`: tipos `serde` + `tsify` para exportar a `ui/src/lib/types.ts`.
- Comandos Tauri: `get_config`, `set_config`, `test_mic`, `list_models`,
  `download_model`, `set_language`.
- UI Svelte 5:
  - ruta `/settings` (hotkey picker, modelo select, idioma)
  - ruta `/wizard` (mic test, modelo download opcional, idioma UI)
- `rust-i18n` (back) + `svelte-i18n` (front), archivos `locales/es.json`,
  `locales/en.json`.
- Tray: click ahora abre panel (no sólo menú nativo).

**DoD:**
- Wizard corre en primer arranque (detecta primer launch vía flag config).
- Settings persisten en disco atómicamente.
- Cambio idioma UI en vivo sin reinicio.
- Roles ARIA básicos en componentes.

---

### Fase 4 — Model downloader (1 semana)

**Objetivo:** UI lista modelos instalados (tamaño disco, free space). Descarga
on-demand desde HuggingFace. Cancel + progreso en tray.

Tareas:
- `oido-stt`: `ModelRegistry` (catálogo hardcoded:
  tiny/base/small × {en, multilingual} × q5_1/q8_0).
- descarga vía `reqwest` (rustls), streaming a `tempfile` + rename atómico.
- progreso → `tracing` event → Tauri → UI + tray icon overlay.
- cancel token vía channel.
- verificación SHA256 del `.bin` post-descarga.

**DoD:**
- Descarga base multilingüe desde UI.
- Cancel a mitad sin corromper modelo existente.
- Reanudable (HTTP Range).
- Modelo corrupto (sha mal) → error claro, no se carga.

---

### Fase 5 — Auto-update + signing (Completada para Windows MSI)

**Objetivo:** Auto-updater con verificación de firma y empaquetador MSI completado.

Tareas:
- [x] Configuración de updater con endpoint de GitHub Releases
- [x] Generación y verificación de llaves criptográficas (Ed25519) embebidas
- [x] WiX installer para Windows (`oido.wxs`) generando MSI per-user
- [x] Script de empaquetado y firmado (`build-msi.ps1`) con checksum SHA256
- [x] CI/CD release workflow con compilación estática (`+crt-static`), smoke tests de instalación silenciosa, y publicación automática
- [ ] macOS notarization + Developer ID signing (Fase posterior)
- [ ] Linux AppImage + deb packaging (Fase posterior)

**DoD:**
- Tag `vX.Y.Z` → CI construye + firma → user recibe notificación → update
  en sitio.
- Update sin firma = rechazado.
- Rollback funcional (versión anterior se restaura si update falla).

**Riesgos:**
- macOS notarización requiere Apple Developer account ($99/año). Confirmar
  antes de iniciar fase.
- Cert EV/OV Windows requiere proceso burocrático. Empezar trámite pronto.

---

### Fase 6 — Translation popup (2 semanas)

> **Estado:** activación condicional post-Fase 5 según uso medido.

**Objetivo:** selección de texto → hotkey → popup Tauri hijo con traducción.
Backend offline (libretranslate local) o API (toggle).

Tareas:
- `oido-translation` crate (nuevo) o módulo en core (decidir al inicio).
- backend trait `Translator` + impls `LocalLibreTranslate` (HTTP a proceso
  local) + `GoogleApi` + `DeepLApi`.
- popup: ventana Tauri sin bordes, posición cerca cursor.
- `mouse_monitor` port (detecta selección texto por OS).
- toggle config: offline vs API + API key storage seguro (`keyring` crate).

**DoD:**
- Seleccionar "hello" en navegador → hotkey → popup "hola" cerca cursor.
- Toggle offline/API funciona sin reinicio.
- API keys persistidas de forma segura, no en JSON plano.

---

### Fase 7 — Accesibilidad WCAG 2.1 AA (1 semana)

**Objetivo:** panel completo cumple WCAG 2.1 AA. Screen readers funcional.

Tareas:
- Roles ARIA en todos componentes Svelte.
- Live regions para cambios de estado (`listening`, `procesando`, `done`).
- Contraste verificado ≥4.5:1 texto.
- Navegación teclado completa (Tab, Esc, Enter, Espacio).
- Labels `aria-label` en iconos tray / acciones.
- Testing manual con NVDA (Win), VoiceOver (mac), Orca (Linux).

**DoD:**
- Screen reader anuncia "Escuchando" cuando state cambia a listening.
- Tab llega a todos los controles sin trampas de foco.
- Auditoría automatizada (`axe-core` en CI) sin violaciones serious/critical.
- Contraste verificado en todos los temas.

---

### Fase 8 — Polish cross-platform (1-2 semanas)

**Objetivo:** Wayland, distribución nativa por OS, edge cases.

Tareas:
- Wayland: hotkey + capture vía `org.freedesktop.portal.*` (InputCapture +
  ScreenCast).
- macOS: `Info.plist` completo, icono `.icns`, universal binary
  (x86_64 + aarch64).
- Linux: AppImage + `.deb` + AUR PKGBUILD.
- Win: NSIS installer con auto-start opcional.
- i18n: revisar strings hardcoded, migrar a `t!()` / `$_()`.
- Decisión registrada: **cursor animado skip** — tray overlay comunica igual.

**DoD:**
- Build nativo en los 3 OS.
- Wayland funciona end-to-end (Ubuntu 22.04 GNOME + KDE).
- Iconos correctos por plataforma en taskbar / dock.

---

## Registro YAGNI (módulos dropeados explícito)

Decisiones de "no construir" para evitar over-engineering. Si alguien propone
re-introducir uno, este registro es el pushback.

| Módulo Oido                  | Razón drop                                      | Alternativa en Oido 2.0 |
|------------------------------|-------------------------------------------------|-------------------------|
| `virtual_desktop.py`         | Win-only, no portable, nadie usa                | nada                    |
| `perf_tuner.py` beam grid    | whisper.cpp ya auto-ajusta a hardware           | nada                    |
| `cursor_fx.py` + RMS feedback| 3 comunidades para feedback visual              | tray icon 3 estados     |
| `tray_panel_flet.py`         | Flet reemplazado entero                         | Tauri panel (Fase 3)    |
| `translator_flet*.py`        | Flet reemplazado entero                         | Tauri popup (Fase 6)    |
| `mouse_monitor.py`           | Sólo útil si traducción popup activa            | re-port Fase 6          |
| IPC por JSON files + polling | Race conditions + poll 200ms en Oido            | Tauri commands nativos  |
| `icons.py` precompute        | Específico de pystray                           | assets/ PNG/SVG estáticos |

## Lecciones portadas del Oido original

Patrones del Python actual que **sí** valen la pena y se portan literal.

| Patrón Oido             | Líneas | Acción en 2.0                | Lugar                                |
|-------------------------|--------|------------------------------|--------------------------------------|
| `atomic_io.py` tmp+rename | 46    | port Rust                    | `oido-config/src/atomic.rs`          |
| `text_processor.py` phrase-filter | 28 | port + frases ES añadidas | `oido-core/src/phrase_filter.rs`     |
| `SegmentDeduplicator`    | 18     | port                         | `oido-core/src/dedup.rs`             |
| `Transcriber` Protocol   | 7      | port como trait Rust         | `oido-stt/src/lib.rs` (sección trait)|
| `workers.py` 3 colas     | —      | port como channels           | `oido-core/src/pipeline.rs`          |
| `ModelSpec` / `VADConfig` dataclasses | — | port como `struct` serde | `oido-stt/src/lib.rs`                |
| `ConfigManager` load/save JSON | — | port con atomic write     | `oido-config/src/store.rs`           |

## Decisiones diferidas (checkpoint post-Fase 5)

| Decisión                              | Trigger para reabrir                        |
|---------------------------------------|---------------------------------------------|
| Activar Fase 6 (translate popup)      | uso medido del dictado sugiere demanda       |
| Streaming whisper.cpp                 | latencia >3s en frases >10 palabras          |
| GPU accel (CUDA / Metal / Vulkan)     | CPU insuficiente en hardware target          |
| Comandos de voz ("borrar", "nueva línea") | usuarios piden acciones por voz         |
| Wake-word / always-listening          | si el consumo de batería se vuelve queja     |

## Glosario

- **Pipeline:** cadena `CaptureSource → STT → Filtro → Injector` orquestada por
  `oido-core`.
- **Holding mode:** hotkey mantenido presionado graba; release transcribe.
- **Frase descartable:** output de whisper.cpp reconocido como alucinación
  (`PhraseFilter`).
- **Atomic write:** tmp file en mismo directorio + `fs::rename` atómico cross-OS.
- **Shape del cursor activo:** ventana / control que recibe el `paste` simulado.

## Estado del documento

| Sección | Versión | Última edición |
|---------|---------|----------------|
| Stack   | 1.0     | 2026-07-03     |
| Reglas  | 1.0     | 2026-07-03     |
| Fases   | 1.0     | 2026-07-03     |
