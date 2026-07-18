# Plan: Commit del Plan de Limpieza Arquitectónica para Producción

## Contexto

El plan de limpieza arquitectónica define 5 mejoras puntuales de robustez, lints y documentación para preparar el codebase de oido-rs para producción. **Todos los cambios ya están aplicados en el working tree** (no en commits); falta únicamente empaquetarlos en un commit enfocado siguiendo las reglas del workspace.

**Estado actual verificado:**
- 140 tests pasan (0 fallos, 3 ignorados por dependencias externas)
- `cargo clippy --all-targets -- -D warnings` → 0 warnings
- Sin usos huérfanos de `CpalCapture::default()` en el workspace

## Cambios a incluir en el commit

### 1. `.gitignore` (+7 líneas)
Añadir al final del archivo:
```gitignore
# Test audio and data files
*.mp3
*.wav
*.bin
*.npy
```
Justificación: los archivos `Grabación_prueba.mp3`, `audio_16k.{bin,npy,wav}` (~4 MB en total) ya no pueden subir accidentalmente al repo.

### 2. `ARCHITECTURE.md` (línea 22)
Aclarar la regla R1 con la excepción explícita:
```
- **Prohibido** `Arc<Mutex<T>>` o `Arc<RwLock<T>>` fuera de `oido-config`, con la única
  excepción autorizada de `Arc<Mutex<BufferState>>` en `oido-core` (necesaria para
  acumular las muestras de audio desde el callback de captura de `cpal` y sincronizarlo
  con el ciclo de vida del pipeline).
```
Justificación: alineamiento entre contrato y realidad del código.

### 3. `crates/oido-audio/src/capture.rs` (-18 líneas)
**Eliminar completamente** el bloque `impl Default for CpalCapture` (líneas 209-225 del archivo antiguo). El `default()` realizaba `.expect("sin dispositivo")` que paniqueaba instantáneamente en entornos sin dispositivo de audio (drivers rotos, VMs, sandboxes). `CpalCapture::new()` ya maneja este caso vía `Result<AudioError>`.

Verificación de seguridad: 0 usos de `CpalCapture::default()` en el workspace; único callsite es `main.rs:599` que usa `CpalCapture::new()`.

### 4. `crates/oido-core/tests/chunked_real_audio.rs` (2 fixes de lint)
- Línea 222: `fn load_audio_bin(path: &PathBuf)` → `fn load_audio_bin(path: &std::path::Path)` (fix `clippy::ptr-arg`, extensión del scope original para mantener consistencia con `load_whisper`)
- Línea 237: `fn load_whisper(model_path: &PathBuf)` → `fn load_whisper(model_path: &std::path::Path)` (ya aplicado, fix `clippy::ptr-arg`)

### 5. `crates/oido/src/models_setup.rs` (struct update syntax en tests)
En las dos pruebas que construyen `Config` con campos custom, usar `..Default::default()` en lugar de field reassignment (fix `clippy::field_reassign_with_default`):
- `sanitize_config_fixes_orphan_custom_prompt` (línea 212)
- `sanitize_config_keeps_custom_with_text` (línea 249)

## Archivos a NO incluir en el commit

Los siguientes archivos están modificados en el working tree pero **no** son parte de este plan:
- `.atl/.skill-registry.cache.json`
- `.atl/skill-registry.md`
- `crates/oido-models/src/lib.rs` (+163/-2)
- `crates/oido-tray/src/dialog.rs` (+125)
- `crates/oido-tray/src/lib.rs` (+3)
- `crates/oido-tray/src/tray.rs` (+28)
- `crates/oido/Cargo.toml` (+4)
- `crates/oido/src/model_lifecycle.rs` (+134)

Estos pertenecen a otro trabajo (probablemente features de tray/modelos) y deben commitearse por separado. Los archivos untracked en `.zcode/plans/*.md` tampoco se tocan.

## Ejecución del commit

Aplicar la skill `git-best-practices` para el flujo:

1. **Verificar rama y aislamiento**: confirmar que estamos en `main` (verificado) y que no hay cambios stash.
2. **Stage selectivo**: `git add .gitignore ARCHITECTURE.md crates/oido-audio/src/capture.rs crates/oido-core/tests/chunked_real_audio.rs crates/oido/src/models_setup.rs`
3. **Verificar staged**: `git diff --cached --stat` para confirmar que solo van los 5 archivos del plan.
4. **Verificar tests pre-commit**: `cargo test --workspace` y `cargo clippy --all-targets -- -D warnings` (ambos ya verificados: 140/0 y 0 warnings).
5. **Commit** con mensaje conventional:
   ```
   chore(cleanup): plan de limpieza arquitectónica para producción

   - .gitignore: ignorar *.mp3 *.wav *.bin *.npy (datos de prueba huérfanos)
   - ARCHITECTURE.md: aclarar excepción R1 (Arc<Mutex<BufferState>> en oido-core)
   - oido-audio: remover impl Default for CpalCapture (panic latente)
   - oido-core/tests: &PathBuf → &Path en load_audio_bin y load_whisper
   - oido/src/models_setup: struct update syntax en tests de sanitize_config

   Verificado: cargo test --workspace (140 ok) + cargo clippy -D warnings (0 warnings)
   Sin cambios de comportamiento en el pipeline.
   ```
6. **No push** (no autorizado en este turn).

## Verificación post-commit

- `git log -1 --stat` para confirmar el commit quedó con exactamente los 5 archivos y el mensaje esperado.
- `git status` para confirmar que los archivos no relacionados siguen modified y listos para su propio commit.