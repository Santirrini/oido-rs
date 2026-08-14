//! Loader y slicer de `voices-v1.0.bin` para Kokoro-82M.
//!
//! ## Formato del archivo (verificado contra `thewh1teagle/kokoro-onnx`)
//!
//! `voices-v1.0.bin` es un archivo **NPZ** (numpy `savez`, no
//! comprimido — usa `STORE` en ZIP) que contiene **un entry `.npy` por
//! voz**. El nombre del entry es el `voice_id` canónico
//! (`"af_heart"`, `"am_michael"`, `"bf_emma"`, ...).
//!
//! Cada entry es un array NPY 2D con dtype `<f4` y shape
//! `[max_token_len, 256]`. La dimensión 0 indexa por número de
//! tokens de fonemas; la dimensión 1 es el vector de estilo
//! (256 floats) que Kokoro espera como input `style` del ONNX.
//!
//! El upstream (`kokoro_onnx/__init__.py`) hace:
//!
//! ```python
//! self.voices = np.load(voices_path)        # mapping voz -> ndarray
//! voice = self.voices[name]                 # 2D [max_token_len, 256]
//! voice = voice[len(tokens)]                # 1D [256]
//! ```
//!
//! Esto significa que `voice_slice(voice_id, token_count)` tiene que
//! hacer exactamente eso: cargar el NPZ en memoria, buscar el entry
//! por nombre y devolver la fila `token_count`.
//!
//! ## `MAX_PHONEME_LENGTH` (del upstream)
//!
//! `kokoro_onnx/config.py` fija `MAX_PHONEME_LENGTH = 510` (deja sitio
//! para el pad 0 al inicio y al final). El `.bin` que distribuye
//! `thewh1teagle` v1.0 tiene por tanto shape `[511, 256]` por voz (índice
//! 0..=510). Si un build futuro usa otra cifra, lo leemos del `.npy`
//! header — no hardcodeamos.
//!
//! ## Padding
//!
//! El `synthesize` real pasa el `len(tokens)` SIN los pads `[0, *, 0]`
//! al slicer de voz — el pad se aplica DESPUÉS al tensor de tokens
//! pero el estilo se selecciona por la longitud de la secuencia
//! "real" (sin pads), igual que el upstream.
//!
//! ## Concurrencia
//!
//! La carga del NPZ se hace una vez (es O(megabytes) en CPU) y se
//! comparte via `Arc`. `voice_slice` se invoca bajo el `Mutex` de la
//! sesión durante la inferencia, así que NO añadimos un segundo lock
//! aquí — el `&[f32]` devuelto se clona a un `Array1` propio (256
//! floats = 1 KB, no es un hot spot).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use ndarray::Array1;
use zip::ZipArchive;

use crate::TtsError;

/// Dimensión fija del estilo por voz (verificado contra config.json de
/// Kokoro-82M: `style_dim = 128` en config, pero el `.bin` distribuye
/// vectores de 256; el modelo ONNX acepta `[1, 256]`). NO hardcodeamos
/// en el loader — la leemos del header NPY de la primera voz encontrada
/// y la validamos contra `EXPECTED_STYLE_DIM` para detectar `.bin`
/// corruptos o de otra versión.
const EXPECTED_STYLE_DIM: usize = 256;

/// Loader thread-safe de `voices-v1.0.bin`.
///
/// Internamente es un `Arc<BTreeMap<String, Vec<Vec<f32>>>>` (voz →
/// filas de 256 floats). `BTreeMap` para que `keys()` esté ordenado y
/// `voices()` del `Engine` sea determinista (no dependa del orden de
/// inserción del zip). El bin se lee **una sola vez** en
/// [`VoiceBank::load`] y se comparte via clonación de `Arc`.
///
/// `Clone` es barato (clona el `Arc`) — lo necesita `KokoroSession`
/// para poder derivar `Clone` y así el `Engine::synthesize` puede
/// soltar el lock exterior antes de inferir.
#[derive(Debug, Clone)]
pub struct VoiceBank {
    /// `voice_id` → filas `[max_token_len][256]`.
    inner: Arc<BTreeMap<String, Vec<Vec<f32>>>>,
}

impl VoiceBank {
    /// Lee el NPZ desde `path` y devuelve el `VoiceBank` listo para
    /// `voice_slice`.
    ///
    /// Errores:
    /// - `TtsError::Backend` si el archivo no se puede abrir/parsear
    ///   como ZIP o si el formato NPY no encaja con la spec.
    /// - `TtsError::InvalidVoiceConfig` si la dimensión de estilo no
    ///   es 256 (defensa contra archivos de otra versión del modelo).
    pub fn load(path: &Path) -> Result<Self, TtsError> {
        let file = File::open(path).map_err(|e| {
            TtsError::Backend(format!("abrir voices-v1.0.bin ({}): {e}", path.display()))
        })?;
        let mut archive = ZipArchive::new(BufReader::new(file))
            .map_err(|e| TtsError::Backend(format!("parse NPZ ({}): {e}", path.display())))?;

        let mut bank: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();

        // Itera sobre los entries. En un `.npz` cada entry termina en
        // `.npy` (numpy los numera si hay colisiones, ej.
        // `af_heart.npy`, `af_heart.1.npy`). Aquí asumimos un entry
        // por voz, así que el nombre "limpio" es el `voice_id`.
        for i in 0..archive.len() {
            let mut entry = archive
                .by_index(i)
                .map_err(|e| TtsError::Backend(format!("npz entry #{i}: {e}")))?;
            let raw_name = entry.name().to_string();
            // Normaliza `af_heart.npy` → `af_heart`. Si tiene sufijo
            // numérico (`af_heart.1.npy`) lo descartamos: voz duplicada,
            // un NPZ bien formado no debería tenerlos.
            let voice_id = match raw_name.strip_suffix(".npy") {
                Some(name) if !name.contains('.') => name.to_string(),
                _ => continue,
            };
            if voice_id.is_empty() {
                continue;
            }

            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry
                .read_to_end(&mut buf)
                .map_err(|e| TtsError::Backend(format!("leer entry {raw_name}: {e}")))?;

            let rows = parse_npy_float32_2d(&buf, EXPECTED_STYLE_DIM)?;
            bank.insert(voice_id, rows);
        }

        if bank.is_empty() {
            return Err(TtsError::InvalidVoiceConfig(format!(
                "voices-v1.0.bin no contiene entradas .npy válidas ({})",
                path.display()
            )));
        }

        Ok(Self {
            inner: Arc::new(bank),
        })
    }

    /// Lista los `voice_id` canónicos presentes en el banco, ordenados
    /// alfabéticamente. Es lo que el `Engine::voices()` consume para
    /// construir el submenú.
    #[must_use]
    pub fn voice_ids(&self) -> Vec<String> {
        self.inner.keys().cloned().collect()
    }

    /// Devuelve el vector de estilo (256 floats) correspondiente a la
    /// voz `voice_id` para una secuencia de `token_count` fonemas.
    ///
    /// `token_count` se clipea al rango `[0, max_token_len - 1]` del
    /// bin: si llega `0` devolvemos la fila 0 (vector "vacío", el
    /// modelo emite silencio) y si llega un valor por encima del
    /// máximo, devolvemos la última fila (el upstream también lo hace
    /// con un assert en `MAX_PHONEME_LENGTH`, pero ser defensivo
    /// aquí evita panics en tiempo de inferencia).
    ///
    /// Devuelve un `Array1<f32>` clonado (no una vista) para que el
    /// caller pueda moverlo a su `Tensor::from_array` sin
    /// restricciones de lifetime.
    pub fn voice_slice(&self, voice_id: &str, token_count: usize) -> Result<Array1<f32>, TtsError> {
        let rows = match self.inner.get(voice_id) {
            Some(r) => r,
            None => {
                let (fallback_name, fallback_rows) = self
                    .inner
                    .get_key_value("ef_dora")
                    .or_else(|| self.inner.get_key_value("af_heart"))
                    .or_else(|| self.inner.iter().next())
                    .ok_or_else(|| TtsError::UnknownVoice(voice_id.to_string()))?;

                tracing::warn!(
                    requested_voice = %voice_id,
                    fallback_voice = %fallback_name,
                    "voz Kokoro no encontrada en el banco; utilizando voz de fallback"
                );
                fallback_rows
            }
        };
        if rows.is_empty() {
            return Err(TtsError::InvalidVoiceConfig(format!(
                "voz '{voice_id}' tiene tabla de estilos vacía"
            )));
        }
        let idx = token_count.min(rows.len() - 1);
        let row = &rows[idx];
        debug_assert_eq!(
            row.len(),
            EXPECTED_STYLE_DIM,
            "voice row tiene dimensión inesperada: {} (esperaba {EXPECTED_STYLE_DIM})",
            row.len()
        );
        Ok(Array1::from(row.clone()))
    }
}

/// Parsea un NPY v1/v2/v3 little-endian con dtype `<f4` (float32) y
/// shape 2D `(rows, cols)`. Devuelve `Vec<Vec<f32>>` (cada fila es
/// un vector de estilo).
///
/// Implementación mínima: leemos el magic `\x93NUMPY`, la versión, el
/// header dict y el bloque de datos raw. Sólo aceptamos dtype `<f4`
/// y fortran_order `false` (el `.bin` distribuido cumple ambos). Si
/// en el futuro sale una variante con `|f4` big-endian o `>f4`,
/// añadiremos un swap.
fn parse_npy_float32_2d(buf: &[u8], expected_cols: usize) -> Result<Vec<Vec<f32>>, TtsError> {
    // Magic: 6 bytes `\x93NUMPY`
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(TtsError::Backend("NPY: magic header ausente".into()));
    }
    let major = buf[6];
    let minor = buf[7];
    // v1: header_len u16 en [8..10], header en [10..10+len].
    // v2/v3: header_len u32 en [8..12], header en [12..12+len].
    let (header_len, header_off) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize),
        2 | 3 => {
            let l = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
            (l, 12usize)
        }
        other => {
            return Err(TtsError::Backend(format!(
                "NPY: versión no soportada {other}.{minor}"
            )))
        }
    };
    let header_end = header_off
        .checked_add(header_len)
        .ok_or_else(|| TtsError::Backend("NPY: header_len overflow".into()))?;
    if header_end > buf.len() {
        return Err(TtsError::Backend(
            "NPY: header_len excede el archivo".into(),
        ));
    }
    let header_str = std::str::from_utf8(&buf[header_off..header_end])
        .map_err(|e| TtsError::Backend(format!("NPY: header no UTF-8: {e}")))?;
    let data = &buf[header_end..];

    // Parsea un dict minimalista. Sólo leemos tres claves:
    //   'descr': debe ser '<f4'
    //   'fortran_order': debe ser False
    //   'shape': tupla (rows, cols)
    let descr = header_field_str(header_str, "descr")
        .ok_or_else(|| TtsError::Backend("NPY: falta 'descr'".into()))?;
    if descr != "<f4" {
        return Err(TtsError::Backend(format!(
            "NPY: dtype '{descr}' no soportado (esperaba '<f4')"
        )));
    }
    let fortran = header_field_str(header_str, "fortran_order")
        .ok_or_else(|| TtsError::Backend("NPY: falta 'fortran_order'".into()))?;
    if fortran != "False" {
        return Err(TtsError::Backend(format!(
            "NPY: fortran_order={fortran} no soportado"
        )));
    }
    let shape_str = header_field_str(header_str, "shape")
        .ok_or_else(|| TtsError::Backend("NPY: falta 'shape'".into()))?;
    let dims = parse_shape_tuple(&shape_str)?;
    let (rows, cols) = match dims.as_slice() {
        [rows, cols] => (*rows, *cols),
        [rows, 1, cols] => (*rows, *cols),
        [1, rows, cols] => (*rows, *cols),
        _ => {
            return Err(TtsError::Backend(format!(
                "NPY: shape debe ser 2D o 3D con dimensión singleton, obtuve {dims:?}"
            )));
        }
    };
    if cols != expected_cols {
        return Err(TtsError::InvalidVoiceConfig(format!(
            "NPY: dimensión de estilo {cols} != {expected_cols} — \
             ¿archivo de otra versión de Kokoro?"
        )));
    }

    // Copia bytes → Vec<f32>. Asumimos little-endian host (todas las
    // targets soportadas: x86_64, aarch64).
    let expected_bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| TtsError::Backend("NPY: tamaño overflow".into()))?;
    if data.len() < expected_bytes {
        return Err(TtsError::Backend(format!(
            "NPY: data corta ({} bytes, esperaba {expected_bytes})",
            data.len()
        )));
    }
    let mut flat: Vec<f32> = Vec::with_capacity(rows * cols);
    // SAFETY: usamos `from_le_bytes` por chunks de 4 bytes; `data` es
    // un slice Rust (no hacemos punteros crudos). Esta función es 100%
    // safe Rust.
    for chunk in data[..expected_bytes].chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        flat.push(f32::from_le_bytes(bytes));
    }

    // Reorganiza flat (row-major) a Vec<Vec<f32>> (un Vec por fila).
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        out.push(flat[start..start + cols].to_vec());
    }
    Ok(out)
}

/// Extrae el valor string de un campo del header dict NPY (formato
/// Python `repr`/`str`). El header es algo como:
///
/// ```text
/// {'descr': '<f4', 'fortran_order': False, 'shape': (511, 256), }
/// ```
///
/// Buscamos la clave, saltamos los dos puntos y devolvemos el valor
/// hasta la próxima coma (o el cierre del dict). Para nuestro uso
/// basta con valores escalares y tuplas; no parseamos listas anidadas.
fn header_field_str(header: &str, key: &str) -> Option<String> {
    // Busca `'<key>':` o `"<key>":`. La clave en el dict suele ir con
    // comillas simples. Distinguimos la *primera* ocurrencia (no
    // buscamos dentro de un valor) con la asunción de que el header
    // es lineal.
    let needle1 = format!("'{key}':");
    let needle2 = format!("\"{key}\":");
    let value_off = if let Some(pos) = header.find(&needle1) {
        pos + needle1.len()
    } else if let Some(pos) = header.find(&needle2) {
        pos + needle2.len()
    } else {
        return None;
    };
    let rest = &header[value_off..];
    // Salta whitespace inicial.
    let rest = rest.trim_start();
    // El valor termina en la próxima coma de nivel-0 o el cierre `}`.
    // Es una aproximación suficiente para los valores escalares y
    // tuplas simples que emite numpy 1.x-2.x.
    let bytes = rest.as_bytes();
    let mut depth: i32 = 0;
    let mut end = rest.len();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth < 0 {
                    end = i;
                    break;
                }
            }
            b',' if depth == 0 => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    let mut value = rest[..end].trim().to_string();
    // Strip surrounding quotes (numpy usa repr() de string, que mete
    // comillas simples: `'descr': '<f4'`). Sin esto, el caller
    // compararía con `"<f4"` cuando el valor real es `"'<f4'"`.
    if (value.starts_with('\'') && value.ends_with('\''))
        || (value.starts_with('"') && value.ends_with('"')) && value.len() >= 2
    {
        value = value[1..value.len() - 1].to_string();
    }
    Some(value)
}

/// Parsea una tupla de enteros como `"(511, 256)"` → `vec![511, 256]`.
/// Acepta espacios arbitrarios. Para shape de 1D (no usado aquí pero
/// trivial de soportar) sería `"512,"` o `"(512,)"`.
fn parse_shape_tuple(s: &str) -> Result<Vec<usize>, TtsError> {
    let trimmed = s.trim();
    let inner = trimmed
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .unwrap_or(trimmed);
    let mut out = Vec::new();
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let n: usize = t
            .parse()
            .map_err(|e| TtsError::Backend(format!("NPY: shape no entero '{t}': {e}")))?;
        out.push(n);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// Tests unitarios sin tocar el archivo NPZ real (326 MB que F7
// descargará). Cubren:
//   - `parse_shape_tuple` con tuplas reales del .bin (511, 256).
//   - `header_field_str` sobre un header numpy real.
//   - `parse_npy_float32_2d` sobre un NPY sintético mínimo (3 filas ×
//     2 cols) para validar la pipeline completa.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shape_tuple_handles_typical_kokoro_shape() {
        let dims = parse_shape_tuple("(511, 256)").expect("tuple");
        assert_eq!(dims, vec![511, 256]);
    }

    #[test]
    fn parse_shape_tuple_handles_spaces_and_single_dim() {
        assert_eq!(parse_shape_tuple("( 3 , 2 )").unwrap(), vec![3, 2]);
        assert_eq!(parse_shape_tuple("512,").unwrap(), vec![512]);
    }

    #[test]
    fn parse_shape_tuple_rejects_non_integer() {
        assert!(parse_shape_tuple("(abc, 256)").is_err());
    }

    #[test]
    fn header_field_str_extracts_descr() {
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (511, 256), }";
        assert_eq!(header_field_str(h, "descr").as_deref(), Some("<f4"));
        assert_eq!(
            header_field_str(h, "fortran_order").as_deref(),
            Some("False")
        );
        assert_eq!(header_field_str(h, "shape").as_deref(), Some("(511, 256)"));
        assert_eq!(header_field_str(h, "missing"), None);
    }

    /// Construye un NPY sintético de 3 filas × 2 cols con valores
    /// `[0.0, 1.0]`, `[2.0, 3.0]`, `[4.0, 5.0]`. Valida que
    /// `parse_npy_float32_2d` lo deserializa correctamente.
    #[test]
    fn parse_npy_float32_2d_round_trips_synthetic_buffer() {
        // Construimos el header NPY v1.
        let header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 2), }";
        // NPY v1: header_len u16 + header padded a múltiplo de 64 con
        // espacios + 1 byte newline.
        let mut header_bytes = header_dict.as_bytes().to_vec();
        while !(header_bytes.len() + 10 + 1).is_multiple_of(64) {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        let header_len = header_bytes.len() as u16;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1); // major
        buf.push(0); // minor
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        for v in [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let rows = parse_npy_float32_2d(&buf, 2).expect("parse");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![0.0, 1.0]);
        assert_eq!(rows[1], vec![2.0, 3.0]);
        assert_eq!(rows[2], vec![4.0, 5.0]);
    }

    /// `parse_npy_float32_2d` rechaza un dtype != `<f4` para que un
    /// `.bin` futuro de Kokoro con `<f8` (float64) no se cargue
    /// silenciosamente con la mitad de memoria.
    #[test]
    fn parse_npy_float32_2d_rejects_wrong_dtype() {
        let header_dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 4), }";
        let mut header_bytes = header_dict.as_bytes().to_vec();
        while !(header_bytes.len() + 10 + 1).is_multiple_of(64) {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        let header_len = header_bytes.len() as u16;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.extend_from_slice(&[0u8; 64]); // 2 × 4 × 8 bytes dummy
        let res = parse_npy_float32_2d(&buf, 4);
        assert!(matches!(res, Err(TtsError::Backend(_))));
    }

    /// `parse_npy_float32_2d` rechaza una dimensión de estilo != 256.
    /// Defensa contra un `.bin` de otra versión de Kokoro (que podría
    /// tener `style_dim = 128` por ejemplo) cargado accidentalmente.
    #[test]
    fn parse_npy_float32_2d_rejects_wrong_style_dim() {
        let header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 128), }";
        let mut header_bytes = header_dict.as_bytes().to_vec();
        while !(header_bytes.len() + 10 + 1).is_multiple_of(64) {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        let header_len = header_bytes.len() as u16;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.extend_from_slice(&[0u8; 2 * 128 * 4]);
        let res = parse_npy_float32_2d(&buf, 256);
        assert!(matches!(res, Err(TtsError::InvalidVoiceConfig(_))));
    }

    /// `parse_npy_float32_2d` soporta shapes 3D con dimensión singleton como (3, 1, 2).
    #[test]
    fn parse_npy_float32_2d_supports_3d_singleton_shape() {
        let header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 1, 2), }";
        let mut header_bytes = header_dict.as_bytes().to_vec();
        while !(header_bytes.len() + 10 + 1).is_multiple_of(64) {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        let header_len = header_bytes.len() as u16;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        for v in [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let rows = parse_npy_float32_2d(&buf, 2).expect("parse 3D singleton");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![0.0, 1.0]);
        assert_eq!(rows[1], vec![2.0, 3.0]);
        assert_eq!(rows[2], vec![4.0, 5.0]);
    }
}
