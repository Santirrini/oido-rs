## Contexto / diagnóstico

Bug confirmado en `crates/oido-stt/src/whisper_cpp.rs`, método `transcribe_timed`, líneas **529–532**. El modo Chunked reconstruye el texto token a token (necesita los `t1` por token para el corte carryover), pero el espaciado está mal:

```rust
text_buf.push_str(trimmed);          // trim() borra el espacio-prefijo = marcador de límite de palabra
if !text_buf.ends_with(' ') {
    text_buf.push(' ');              // fuerza espacio tras CADA token
}
```

El tokenizador de whisper marca el inicio de palabra con un espacio prefijo (`" hol"`); subwords/continuaciones (`"a"`, `"ando"`) y puntuación (`","`) vienen sin él. `trim()` + espacio forzado destruye la señal → `"hol a"`, `"prob ando"`, `"Hola ,"`. El campo `starts_new_word` (línea 517) ya detecta el límite pero solo se usa para el corte, no para el espaciado.

Referencia de código correcto: `transcribe` batch (líneas 404–421) lee `seg.to_str_lossy()`; `streaming.rs::tokens_to_string` concatena bytes crudos. Ninguno tiene el bug.

## Plan de implementación

### 1. Crear rama (git best-practices)
- Partir de `main`. Crear `fix/chunked-token-spacing`.
- Commit final por rama → merge a `main` → borrado de la rama (según tu solicitud).

### 2. Extraer helper pura y testeable en `whisper_cpp.rs`
Definir una función libre pura (sin FFI, sin modelo) que reconstruya un token de texto dentro de un buffer respetando el marcador de límite nativo de whisper:

```rust
/// Reconstruye `text_buf` añadiendo un token de whisper respetando su
/// marcador de límite de palabra nativo (espacio prefijo o ▁ SentencePiece),
/// en lugar de forzar un espacio tras cada token (que rompe subwords y
/// adelanta puntuación: "hol a", "Hola ,").
///
/// Contrato:
/// - Tokens de inicio de palabra (" hola", " mundo") → añade un espacio
///   separador antes del texto (a menos que sea el primer token).
/// - Subwords/continuación ("a", "ando") y puntuación (",", ".") → se
///   concatenan sin separador, pegándose al token anterior.
/// - Tokens especiales ("[_TT_…]", blanks) → el caller ya los filtró;
///   esta función asume texto "real".
fn append_token_word_aware(text_buf: &mut String, token_text: &str) {
    let starts_new_word =
        token_text.starts_with(' ') || token_text.starts_with('\u{2581}'); // SentencePiece ▁
    if starts_new_word && !text_buf.is_empty() {
        text_buf.push(' ');
    }
    text_buf.push_str(token_text.trim());
}
```

### 3. Reescribir el bucle de `transcribe_timed` (líneas 489–537)
- Sustituir el bloque de líneas 517–532 por: calcular `starts_new_word`, hacer la lógica de corte candidato **antes** de mutar el buffer (igual que ahora), y luego llamar a `append_token_word_aware(&mut text_buf, token_text)`.
- Eliminar el `if !text_buf.ends_with(' ') { push(' ') }` defectuoso.
- El resto del método (corte "todo cabe" vs "corte candidato", coverage 50%, etc.) queda intacto: esos recortes operan por `text_buf.len()`/`cut_text_len` que siguen siendo válidos. Único cuidado: el `cut_text_len` debe tomarse **antes** de `append_token_word_aware` (como ya hace el código con `text_buf.len()`), para que el corte caiga en límite de palabra — sin cambio funcional ahí.
- Los `.trim()` finales (líneas 557, 570) se mantienen para limpiar el espacio prefijo del primer token o un espacio de cola.

### 4. Tests unitarios deterministas (sin modelo) en el `mod tests`
Cubrir el helper de forma aislada para atrapar regresiones en CI sin el `.bin`:

```rust
#[test]
fn append_token_word_aware_joins_subwords() {
    let mut b = String::new();
    append_token_word_aware(&mut b, " hol");   // nueva palabra
    append_token_word_aware(&mut b, "a");       // subword → se pega
    assert_eq!(b, "hola");
}

#[test]
fn append_token_word_aware_keeps_punctuation_glued() {
    let mut b = String::new();
    append_token_word_aware(&mut b, " Hola");
    append_token_word_aware(&mut b, ",");
    append_token_word_aware(&mut b, " mundo");
    assert_eq!(b, "Hola, mundo");   // NO "Hola , mundo"
}

#[test]
fn append_token_word_aware_multi_segment_words() {
    let mut b = String::new();
    for t in [" prob", "ando", " son", "ido", "."] {
        append_token_word_aware(&mut b, t);
    }
    assert_eq!(b, "probando sonido.");
}

#[test]
fn append_token_word_aware_first_token_no_leading_space() {
    let mut b = String::new();
    append_token_word_aware(&mut b, " Hola");
    append_token_word_aware(&mut b, ".");
    assert_eq!(b, "Hola.");
}

#[test]
fn append_token_word_aware_sentencepiece_marker() {
    let mut b = String::new();
    append_token_word_aware(&mut b, "\u{2581}Hola");  // ▁ marker
    assert_eq!(b, "Hola");
}
```

(Opcional, si aplica) un test de "regresión directa" que simule el flujo completo del bucle con una lista de `(token_text, t1_cs)` ficticia, replicando el corte, para validar `text_buf` final = `"Hola, probando sonido."`. Si extraer el bucle a su propia función pura añade demasiada superficie, me quedo con los tests del helper (que cubren el 100% de la lógica de espaciado).

### 5. Documentar el porqué
Comentario breve junto a la llamada al helper explicando que NO se fuerza espacio tras cada token para no romper subwords/puntuación, y referencia al bug real (`"hol a"`).

## Verificación
1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test -p oido-stt` (corre los nuevos tests puros; los smoke `#[ignore]` no corren solos)
4. **Smoke con modelo**: si `models/ggml-base.bin` existe, `cargo test -p oido-stt --features audio-smoke -- --ignored smoke_transcribe_timed_fits_all` y `smoke_transcribe_real_audio`, revisando que el `text=` salga bien espaciado. Si el modelo no está, lo reporto y dejo los smoke para que los corras tú.

## Git (según tu solicitud)
- Crear rama `fix/chunked-token-spacing` desde `main`.
- Commitear el fix + tests con mensaje claro (`fix(stt): reconstruir texto por límite de palabra en transcribe_timed`).
- Verificar gates anteriores.
- Fusionar a `main` (fast-forward o merge --no-ff según el flujo del repo) y **borrar la rama**.

## Alcance / no cambios
- No se toca `transcribe` (batch) ni `streaming.rs` (no tienen el bug).
- No se toca `chunked_pipeline.rs`: solo recorta `.trim()` al texto que recibe, el bug le llega de arriba.
- No se cambia el prompt (el bilingüe puede aumentar fragmentación, pero la causa raíz es el espaciado, no el prompt).