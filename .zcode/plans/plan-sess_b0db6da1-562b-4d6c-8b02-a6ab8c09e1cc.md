## Objetivo

El usuario dicta en español con el modelo `ggml-small.en.bin` (solo inglés) y `language_ui="es"`. Eso produce transcripciones corruptas (`Estes una provez un nido para ver como fucion a me applicación`). El fix profesional combina: (1) detección de la incompatibilidad modelo-inglés vs idioma-no-inglés, (2) aviso **no intrusivo** en el tray (tooltip + ítem de menú que descarga+activa el modelo multilingüe equivalente), (3) sanitización del preset huérfano (`Custom` + `system_prompt` vacío), y (4) integración con el flujo de fallback existente.

No se cambia el modelo a la fuerza: el usuario decide desde el menú.

## Cambios por archivo

### 1. `crates/oido-models/src/lib.rs` — helper de detección y resolución
Añadir dos funciones públicas puras (sin estado, fáciles de testear):
- `pub fn is_english_only_model(filename: &str) -> bool` → reemplaza la heurística `ends_with(".en.bin")` repartida por el código y la centraliza. Implementación: `find(filename).map(|e| e.language == Language::En).unwrap_or(filename.ends_with(".en.bin"))` (defensiva para modelos fuera de catálogo).
- `pub fn multilingual_counterpart(filename: &str) -> Option<&'static ModelEntry>` → dado un filename `*.en.bin`, busca en el catálogo el entry con el mismo `family` y `language == Language::Multi`. Para `ggml-small.en.bin` → `ggml-small.bin`. Devuelve `None` si el input no es `.en` o no hay contraparte (p.ej. fuera de catálogo).

### 2. `crates/oido-tray/src/tray/i18n.rs` — nuevos strings
Añadir al struct `Strings`:
- `pub model_lang_mismatch_tooltip: &'static str` — p.ej. ES: `"oido — ⚠ modelo solo inglés con idioma español; ábreme para corregir"`.
- `pub model_lang_mismatch_action: &'static str` — p.ej. ES: `"⚠ Cambiar a modelo multilingüe…"`.
Rellenar las 3 tablas (ES/EN/Bilingual) y el test `all_strings_are_non_empty` (añadir las 2 claves al array de barrido).

### 3. `crates/oido-tray/src/tray/sections.rs` — nueva sección de aviso condicional
- Nueva sección `ModelLangMismatchSection { ui_language, suggested_filename: String }` que, cuando el mismatch está activo, renderiza **un único** `Section::Item` con id `"fix_model_lang"` y label `"⚠ Cambiar a modelo multilingüe (ggml-small.bin)…"`. Si el filename sugerido es `None`, la sección devuelve `vec![]` (no renderiza nada).
- Añadir `"fix_model_lang" => MenuAction::FixModelLanguage(suggested_filename)` al `id_to_action`.
- Insertar la sección en `default_sections` **antes** de `HotkeySection` (visibilidad máxima) pero solo si `BuildContext` trae `model_lang_mismatch: Option<String>` con `Some(suggested)`. Esto evita ramificar el árbol para el caso común.
- Añadir `pub model_lang_mismatch: Option<String>` a `BuildContext` (default `None` en `initial()`).

### 4. `crates/oido-tray/src/traits.rs` — nueva acción
- Añadir variante `MenuAction::FixModelLanguage(String)` al enum `MenuAction` (el `String` es el filename sugerido, p.ej. `"ggml-small.bin"`).

### 5. `crates/oido-tray/src/tray.rs` — tooltip de aviso
- `state_tooltip(state)` se queda igual (no se toca el enum `TrayState`).
- Nueva función `pub fn mismatch_tooltip(ui_language: UiLanguage) -> String` que devuelve el string `model_lang_mismatch_tooltip` de la tabla i18n correspondiente.
- Exponer vía `lib.rs` para que el bin lo use directamente al hacer `set_tooltip`.

### 6. `crates/oido/src/main.rs` — detección + aplicación del aviso
- Tras el bloque de auto-recovery de modelo (línea ~177), añadir un cálculo:
  ```rust
  let model_lang_mismatch = oido_models::is_english_only_model(&snap.model)
      && !snap.language_ui.eq_ignore_ascii_case("en");
  let suggested_multi = if model_lang_mismatch {
      oido_models::multilingual_counterpart(&snap.model).map(|e| e.filename.clone())
  } else { None };
  ```
  Queda capturado por el closure `start_pipeline` y por el loop principal.
- En el `BuildContext` que se construye en **todos** los puntos (initial startup, `SetTheme`, `SetSttMode`, `SetUiLanguage`, `SetPromptPreset`, `RefreshMenu`), añadir el campo `model_lang_mismatch: suggested_multi.clone()`. Para no repetir el cálculo en 6 sitios, refactorizar a un helper `fn build_ctx(cfg, &snap) -> BuildContext` local que centralice. (Esto reduce el riesgo de inconsistencias — un patrón que el propio código ya sufre: el campo `model_active` se recalcula en cada site.)
- En el handler `MenuAction::FixModelLanguage(filename)` del `oido-menu-listener` thread: delegar al mismo `handle_model_click(&filename, ...)` existente. Si el modelo NO está instalado, `handle_model_click` ya lanza el thread `oido-downloader` y al terminar invoca `activate_after_download` que persiste + recarga. **Cero código nuevo de descarga**: reutilizamos el flujo probado.
- En el bloque donde se fija `initial_state` del tray (línea ~821-830), si `suggested_multi.is_some()` llamar a `t.set_tooltip(Some(mismatch_tooltip(snap.ui_language)))` además del `set_state`. El tooltip de aviso **convive** con el icono de estado.

### 7. `crates/oido/src/models_setup.rs` — sanitización del preset huérfano
- En `resolve_prompt_text`, rama `Custom` con `system_prompt.is_empty()`: además del `warn!` actual, devolver el bilingual **y** añadir un comentario documentando que el bin sanitiza (en main.rs ver abajo) para evitar recurrencia.
- Añadir `pub(crate) fn sanitize_config(cfg: &ConfigStore) -> bool` que, si detecta `prompt_preset == Custom && system_prompt.is_empty()`, reescribe persistentemente a `PromptPreset::BilingualEsEn` y devuelve `true` (hubo cambio). Esto separa "resolver texto en runtime" (defensivo, no muta) de "sanitizar disco" (muta, una sola vez). El `true` permite al bin loguear la corrección con `info!`.

### 8. `crates/oido/src/main.rs` — invocar la sanitización al startup
- Llamar `models_setup::sanitize_config(&cfg)` justo después de cargar `cfg` (línea ~85), antes del bloque de auto-recovery. Si devuelve `true`, loguear `info!("config sanitizada: prompt_preset Custom huérfano → BilingualEsEn")` y refrescar `snap = cfg.snapshot()`.

## Tests a añadir

- `oido-models`: `is_english_only_model("ggml-small.en.bin") == true`, `("ggml-small.bin") == false`, `("ggml-silero-v5.1.2.bin") == false`; `multilingual_counterpart("ggml-small.en.bin")` → `filename == "ggml-small.bin"`, `multilingual_counterpart("ggml-small.bin") == None`.
- `oido-tray::sections`: `default_sections` con `model_lang_mismatch = Some("ggml-small.bin".into())` produce un item con `id == "fix_model_lang"`; con `None` no produce ese item. `id_to_action("fix_model_lang")` mapea correctamente (caso general; el variante lleva el filename).
- `oido-tray::i18n`: las 2 nuevas claves existen y no están vacías en las 3 tablas.
- `models_setup`: `sanitize_config` sobre un `ConfigStore` con preset Custom + system_prompt vacío deja en disco `BilingualEsEn`. (Reusa `tempfile` para `OIDO_MODELS_DIR` / path de config.)

## Tests a actualizar

- `default_sections_produces_expected_tree`: ahora `BuildContext` lleva `model_lang_mismatch`. Actualizar los constructores en los tests existentes (añadir `model_lang_mismatch: None`). El assert de `sections.len() == 7` se mantiene (la sección de mismatch está vacía con `None`).
- `default_sections_cover_all_menu_actions`: opcionalmente añadir `"fix_model_lang"` a la lista de ids esperados **solo si** el test se cambia a `Some(...)`. Lo dejamos fuera del barrido por defecto porque la sección es condicional.

## No incluye (fuera de scope)

- No se reemplaza el modelo automáticamente (respeto a la decisión del usuario).
- No se añade lógica de "no volver a preguntar" ni flag en config: la condición desaparece sola cuando el modelo deja de ser `.en` o el idioma pasa a `en`.
- No se toca el `state_tooltip` existente (sigue gobernando el icono; el aviso va por separado en el tooltip, conviviendo o reemplazando según `set_tooltip` permita — en tray-icon 0.24 `set_tooltip(Some(...))` reemplaza, así que el bin elegirá: si hay mismatch, el tooltip es el de aviso; si no, el del estado). Para no perder info de estado, el string de aviso puede llevar el estado inline: p.ej. `"oido — idle · ⚠ modelo solo inglés con idioma español"`.

## Orden de implementación sugerido

1. `oido-models` helpers + tests (pure functions, base de todo).
2. `oido-tray` i18n + sections + traits + `mismatch_tooltip`.
3. `oido` `sanitize_config` + integración en main.rs (BuildContext helper, handler `FixModelLanguage`, tooltip).
4. `cargo build -p oido` y `cargo test --workspace` para validar.