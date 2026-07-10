## Plan: Grabar la tecla de activación de Oido

### Diagnóstico

El campo `Config.hotkey: String` (default `"F8"`) existe en `crates/oido-config/src/lib.rs:30` pero **nunca se consume**. En `crates/oido-platform/src/hotkey.rs:16,59` la tecla está **hard-coded** como constante `DEFAULT_HOTKEY_CODE: Code = Code::F8`. `crates/oido/src/main.rs:72` llama a `GhHotkey::new()` sin argumentos y en la línea 170 loggea literalmente `"hold F8, dicta, suelta. …"`. No existe ninguna UI ni código de "grabación" de tecla; el comentario `MVP: tecla F8 fija. Fase 2 lo hace configurable desde Config` lo confirma.

La funcionalidad está **pendiente de implementar**, no rota en sentido estricto, pero el resultado observable para el usuario es exactamente "no puedo grabar mi tecla de activación".

### Objetivo

Solución profesional y escalable:

1. **Parser** `&str → (Modifiers, Code)` que cubra cualquier tecla/modalidad que `global-hotkey` soporte (F1-F12, letras A-Z, números 0-9, modificadores Ctrl/Shift/Alt/Meta, símbolos básicos). Cadena canónica `"Ctrl+Shift+D"` o `"F8"`.
2. **Plumbing**: `GhHotkey` recibe el binding desde `Config.hotkey`, lo parsea, y registra esa combinación en lugar de la constante. Reemplazar el log hard-coded por el binding real.
3. **Captura global interactiva**: nueva sub-funcionalidad `oido --set-hotkey` que escucha la siguiente tecla que el usuario pulse a nivel OS (vía `rdev`), la normaliza al formato canónico, y la persiste en `Config.hotkey` mediante `ConfigStore::replace` + `save()` (escritura atómica ya implementada).
4. **Robustez**: validación al arrancar, mensaje claro si el parser falla, fallback a `Config::default()` con `tracing::warn!`.

### Cambios por archivo

#### `Cargo.toml` (raíz, workspace deps)
Añadir `rdev = "0.5"` a `[workspace.dependencies]` (event-driven, distingue modificadores, cross-platform Win/macOS/Linux).

#### `crates/oido-platform/Cargo.toml`
Añadir `rdev.workspace = true` como dependencia opcional o directa según `Cargo.lock`. (Nota: `rdev` requiere libs nativas; documentar en el comentario del módulo.)

#### `crates/oido-platform/src/hotkey.rs`
- Nueva función `pub fn parse(binding: &str) -> Result<(Modifiers, Code), PlatformError>` con `PlatformError::Hotkey("parse: …")` ante input inválido.
  - Split por `+`, trim, lowercase-insensitive.
  - Tabla de modificadores: `"ctrl"|"control"`, `"shift"`, `"alt"`, `"meta"|"super"|"win"|"cmd"`.
  - Tabla de teclas: F1..F24, A..Z, 0..9, y un set razonable de símbolos (`Space`, `Escape`, `Tab`, `Enter`, `Backspace`, etc.). Lo que `global_hotkey::hotkey::Code` no exponga → `PlatformError::Hotkey`.
- Reemplazar `DEFAULT_HOTKEY_CODE` por un binding recibido en `register`. Cambiar la firma del trait para que reciba el binding:

  ```rust
  // traits.rs
  fn register(
      &mut self,
      binding: &str,
      on_press: Box<dyn Fn() + Send + 'static>,
      on_release: Box<dyn Fn() + Send + 'static>,
  ) -> Result<(), PlatformError>;
  ```

  Decisión consciente: cambiar la firma del trait es preferible a mantener estado mutable interno, porque el binding puede cambiar entre arranques (al usuario le place re-configurar y reiniciar; esto escala a la GUI Tauri futura que solo cambiará config + reinicio).
- `GhHotkey::register` hace `let (mods, code) = parse(binding)?;` y registra esa combinación.
- Tests unitarios inline: `parse("F8")`, `parse("Ctrl+Shift+D")`, `parse("ctrl+shift+d")` case-insensitive, `parse("BogusKey")` falla limpio, `parse("")` falla.

#### `crates/oido-platform/src/key_grab.rs` (nuevo módulo)
- `pub fn grab_next_key() -> Result<(Modifiers, Code), PlatformError>` que:
  1. Spawna un hilo con `rdev::listen` (bloqueante).
  2. Captura el primer `EventType::KeyPress(K)` no-modificador puro (filtra presses sueltos de Ctrl/Shift/Alt para no capturar "solo el modificador").
  3. Combina los `EventType::KeyPress` de modificadores que ocurrieron antes en una ventana de 500 ms como modificadores del binding final.
  4. Devuelve por `crossbeam_channel::bounded(1)` y se cierra el hilo al recibir.
  5. Manejo de errores de `rdev` (permisos macOS, fallos X11) → `PlatformError::Hotkey("grab: …")`.
- Documenta la limitación macOS (igual que global-hotkey).
- Re-exportar desde `crates/oido-platform/src/lib.rs`: `pub mod key_grab;`.

#### `crates/oido-platform/src/lib.rs`
- `pub mod key_grab;` y mantener los `pub use` existentes.

#### `crates/oido/src/main.rs`
- Procesar flags CLI con `std::env::args()` (sin clap para no añadir deps): detectar `--set-hotkey` antes de arrancar el pipeline.
- Si `--set-hotkey`:
  1. `tracing::info!("pulsa la tecla que quieras usar como activador (Esc para cancelar)…")`.
  2. Llamar `key_grab::grab_next_key()`; mapear `(Modifiers, Code)` → string canónico via un helper `serialize_binding(mods, code) -> String` (espejo del parser, en `oido-platform/src/hotkey.rs`).
  3. `ConfigStore::replace(Config { hotkey: new, .. })` + `save()` (atómico, ya funciona).
  4. Salir con `Ok(())` sin arrancar el pipeline.
- En el flujo normal (sin flag): pasar `snap.hotkey.as_str()` a `pipeline_cfg.hotkey.register(…)`. Cambia la composición del `PipelineConfig` para que el binding se entregue al `register` del trait.
- Reemplazar `tracing::info!("hold F8, dicta, suelta. …")` por `tracing::info!(hotkey = %snap.hotkey, "hold {hotkey}, dicta, suelta. Ctrl+C para salir.")`.
- Si el `parse` del binding falla al arrancar: `tracing::warn!` con el motivo y caer a `Config::default().hotkey` (no abortar).

#### `crates/oido-core/src/pipeline.rs`
- Ajustar el sitio donde se invoca `register(…)` para pasar `&self.config.hotkey_binding` (o el equivalente: probablemente exponer el binding desde `PipelineConfig`). Cambio mínimo, quirúrgico.

#### `crates/oido-core/tests/pipeline_e2e.rs`
- Añadir un test que use `parse("Ctrl+Shift+D")` para validar que el `Pipeline` levanta con un binding no-F8 sin entrar en panico (no requiere audio real; el hotkey ya está implementado como stub en tests si lo hay, si no se mockea a nivel del trait).

### Verificación

1. `cargo build --workspace` — sin warnings nuevos.
2. `cargo test --workspace` — pasa toda la suite existente más los tests nuevos del parser.
3. `cargo clippy --workspace --all-targets -- -D warnings` — limpio.
4. Manual (no automatizable aquí): ejecutar `cargo run -p oido -- --set-hotkey`, pulsar F9, verificar que `~/.config/oido/config.json` (o `%APPDATA%/oido/config.json` en Windows) ahora contiene `"hotkey": "F9"`. Ejecutar `cargo run -p oido` y confirmar que el log muestra `"hold F9, …"` y que F9 ahora activa el pipeline.

### Decisiones de diseño justificadas

- **Cambio de firma del trait `Hotkey::register`**: preferible a un `set_binding` separado porque (a) hay una sola combinación activa por proceso y (b) simplifica el flujo de Tauri futuro (la GUI reescribirá `Config` y reiniciará el bin).
- **`rdev` en lugar de `device_query`**: event-driven, distingue modificadores robustamente, callback nativo por SO, mejor encaje con la arquitectura de hilos del proyecto.
- **CLI flag en lugar de crate CLI**: una sola flag booleana no justifica `clap` (~600 KB compilados). Si en Fase 3 surge una segunda flag, se introduce `clap`.
- **Formato canónico de modificadores**: `Ctrl+Shift+Alt+Meta+<Key>`, generado siempre por `serialize_binding`, garantiza que el archivo `config.json` sea estable y que el parser inverso sea trivial.
- **Escritura atómica**: ya implementada y probada en `oido-config::atomic_write`; la reutilizamos para no duplicar.

### Riesgos / Notas

- `rdev` en macOS puede requerir los mismos permisos Accessibility que `global-hotkey` (documentado en comentario).
- Tests E2E del grab son difíciles sin GUI real; se cubre con tests del parser (input/output simétrico) y del `serialize_binding`. La captura global queda como verificación manual.
- Ningún cambio rompe la API pública existente salvo `Hotkey::register`, que es un trait interno al workspace; el `PipelineConfig` lo orquesta.