//! Parser del archivo de configuración de voz Piper (`<voice>.onnx.json`).
//!
//! Cada voz de Piper (VITS) se distribuye como un par de archivos:
//!
//! - `<voice>.onnx` — el grafo ONNX que se carga con [`ort::Session`].
//! - `<voice>.onnx.json` — los metadatos de la voz: `audio.sample_rate`,
//!   `espeak.voice`, `inference.{noise_scale,length_scale,noise_w}` y el
//!   `phoneme_id_map` que mapea cada fonema a sus IDs de salida.
//!
//! Este módulo expone [`PiperVoiceConfig`], una vista tipada y robusta
//! sobre ese JSON, con defaults sensatos para los campos opcionales (los
//! modelos actuales siempre traen todos los campos, pero los antiguos /
//! los cuantizados pueden omitir `inference.*`).
//!
//! ## Diseño defensivo
//!
//! Los `.onnx.json` son artefactos generados por `python -m piper` y
//! pueden variar entre versiones (1.0, 1.1, 1.2 de Piper). Cualquier
//! campo ausente se rellena con el default del paper original de VITS y
//! de los `voice.py` del repo upstream (`rhasspy/piper`). Si el campo
//! crítico `phoneme_id_map` falta o no parsea como objeto, devolvemos
//! `TtsError::InvalidVoiceConfig(...)` — el caller puede elegir voz
//! alternativa sin panicar.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::TtsError;

/// Configuración de una voz Piper, deserializada desde el `.onnx.json`
/// que acompaña al grafo ONNX.
///
/// ## Estructura del JSON (verificada contra `rhasspy/piper` upstream)
///
/// ```json
/// {
///   "audio":        { "sample_rate": 22050 },
///   "espeak":       { "voice": "es" },
///   "inference":    { "noise_scale": 0.667,
///                     "length_scale": 1.0,
///                     "noise_w":      0.8   },
///   "phoneme_type": "espeak",
///   "phoneme_id_map": { " ": [2], "a": [25, 27], ... },
///   "num_symbols":  256,
///   "num_speakers": 1,
///   "speaker_id_map": {}
/// }
/// ```
///
/// Los campos opcionales se rellenan con defaults razonables (los del
/// paper de VITS) si faltan en el archivo — ver [`PiperVoiceConfig::from_raw`].
#[derive(Debug, Clone)]
pub struct PiperVoiceConfig {
    /// Sample rate nativo del modelo en Hz (típicamente 22050).
    pub audio_sample_rate: u32,
    /// Voice espeak solicitada (ej. `"es"`, `"es-419"`, `"en-us"`). Se
    /// guarda como referencia informativa; el G2P real se delega a
    /// `piper-plus-g2p`.
    pub espeak_voice: String,
    /// Multiplicador de duración (`length_scale`). 1.0 = normal,
    /// `<1.0` = habla más rápida, `>1.0` = más lenta. Cocinado al
    /// tensor `scales[1]` en cada inferencia.
    pub length_scale: f32,
    /// Ruido introducido en la duración predicha. Típicamente 0.667.
    pub noise_scale: f32,
    /// Ruido introducido en la predicción del waveform. Típicamente 0.8.
    pub noise_w: f32,
    /// Tamaño del vocabulario de salida (`phoneme_type` = `espeak` da
    /// 256). Útil como cota defensiva al validar IDs.
    pub num_symbols: u16,
    /// Número de hablantes del modelo multi-speaker (1 = single-speaker,
    /// el caso que v1.0 soporta). `>1` se acepta pero el caller debe
    /// pasar `sid` en la inferencia — fuera de scope para F2.
    pub num_speakers: u8,
    /// `phoneme_id_map` del `.onnx.json`. Claves son strings de 1+ chars
    /// (un único fonema espeak o un carácter IPA). Valores son listas
    /// de IDs (`Vec<i64>`) — un fonema puede mappear a varios IDs
    /// (PUA mapping para alófonos largos: `aː`, `eː`, etc.).
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
}

impl PiperVoiceConfig {
    /// Sample rate por defecto de Piper si el `.onnx.json` no lo trae.
    /// Todos los modelos de Piper (>= 2023) entrenan a 22050 Hz.
    pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 22_050;
    /// Defaults del paper de VITS + `piper/src/python_run/piper/voice.py`.
    pub const DEFAULT_NOISE_SCALE: f32 = 0.667;
    pub const DEFAULT_LENGTH_SCALE: f32 = 1.0;
    pub const DEFAULT_NOISE_W: f32 = 0.8;
    pub const DEFAULT_NUM_SYMBOLS: u16 = 256;

    /// Lee y parsea un `.onnx.json` desde `path`.
    ///
    /// Devuelve `TtsError::InvalidVoiceConfig` con un mensaje accionable
    /// si el archivo no existe, el JSON es inválido, o faltan campos
    /// críticos (`audio`, `phoneme_id_map`).
    pub fn load(path: &Path) -> Result<Self, TtsError> {
        let bytes = fs::read(path).map_err(|e| {
            TtsError::InvalidVoiceConfig(format!(
                "no se pudo leer '{}': {e}",
                path.display()
            ))
        })?;
        let raw: RawPiperVoiceConfig = serde_json::from_slice(&bytes).map_err(|e| {
            TtsError::InvalidVoiceConfig(format!(
                "JSON inválido en '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_raw(raw).map_err(|e| {
            // Envolvemos en InvalidVoiceConfig con contexto del path.
            TtsError::InvalidVoiceConfig(format!(
                "config '{}' inválida: {e}",
                path.display()
            ))
        })
    }

    /// Convierte el árbol deserializado por `serde_json` en la versión
    /// robusta con defaults aplicados. Punto único donde decidimos qué
    /// hacer cuando un campo opcional falta.
    fn from_raw(raw: RawPiperVoiceConfig) -> Result<Self, String> {
        let audio = raw.audio.ok_or_else(|| {
            "campo obligatorio 'audio' ausente".to_string()
        })?;
        if audio.sample_rate == 0 {
            return Err("audio.sample_rate debe ser > 0".into());
        }
        let phoneme_id_map = raw.phoneme_id_map.ok_or_else(|| {
            "campo obligatorio 'phoneme_id_map' ausente".to_string()
        })?;

        let inference = raw.inference.unwrap_or_default();
        Ok(Self {
            audio_sample_rate: audio.sample_rate,
            espeak_voice: raw
                .espeak
                .as_ref()
                .and_then(|e| e.voice.clone())
                .unwrap_or_default(),
            length_scale: inference.length_scale,
            noise_scale: inference.noise_scale,
            noise_w: inference.noise_w,
            num_symbols: raw.num_symbols.unwrap_or(Self::DEFAULT_NUM_SYMBOLS),
            num_speakers: raw.num_speakers.unwrap_or(1),
            phoneme_id_map,
        })
    }
}

// ---------------------------------------------------------------------------
// Raw deserialization
// ---------------------------------------------------------------------------
//
// Mantenemos las structs `Raw*` separadas para tolerar JSONs con campos
// faltantes o versiones distintas. `serde(default)` + `Option<T>` en cada
// campo nos da robustez sin escribir un visitor custom.

#[derive(Debug, Default, Deserialize)]
struct RawPiperVoiceConfig {
    #[serde(default)]
    audio: Option<RawAudio>,
    #[serde(default)]
    espeak: Option<RawEspeak>,
    #[serde(default)]
    inference: Option<RawInference>,
    /// Tipo de fonema (`"espeak"`, `"text"`, etc.). Se parsea para
    /// tolerar distintos dialectos de `.onnx.json` pero no se usa en
    /// F2: Piper siempre usa espeak-style, y el `phoneme_id_map`
    /// manda sobre la cadena exacta.
    #[serde(default)]
    #[allow(dead_code)]
    phoneme_type: Option<String>,
    #[serde(default)]
    phoneme_id_map: Option<HashMap<String, Vec<i64>>>,
    #[serde(default)]
    num_symbols: Option<u16>,
    #[serde(default)]
    num_speakers: Option<u8>,
    /// Mapa de speakers para modelos multi-speaker (F4+). Lo
    /// parseamos pero no lo usamos en F2 (single-speaker only).
    #[serde(default)]
    #[allow(dead_code)]
    speaker_id_map: Option<HashMap<String, u32>>,
}

#[derive(Debug, Default, Deserialize)]
struct RawAudio {
    sample_rate: u32,
}

#[derive(Debug, Default, Deserialize)]
struct RawEspeak {
    #[serde(default)]
    voice: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawInference {
    #[serde(default = "default_noise_scale")]
    noise_scale: f32,
    #[serde(default = "default_length_scale")]
    length_scale: f32,
    #[serde(default = "default_noise_w")]
    noise_w: f32,
}

impl Default for RawInference {
    fn default() -> Self {
        Self {
            noise_scale: default_noise_scale(),
            length_scale: default_length_scale(),
            noise_w: default_noise_w(),
        }
    }
}

fn default_noise_scale() -> f32 {
    PiperVoiceConfig::DEFAULT_NOISE_SCALE
}
fn default_length_scale() -> f32 {
    PiperVoiceConfig::DEFAULT_LENGTH_SCALE
}
fn default_noise_w() -> f32 {
    PiperVoiceConfig::DEFAULT_NOISE_W
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// JSON mínimo válido (es voz es_ES-davefx-medium).
    const MINIMAL_JSON: &str = r#"{
        "audio": { "sample_rate": 22050 },
        "espeak": { "voice": "es" },
        "phoneme_type": "espeak",
        "phoneme_id_map": { "^": [1], "_": [0], "$": [2], "a": [10] },
        "num_symbols": 256,
        "num_speakers": 1
    }"#;

    fn write_tmp(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("create tmpfile");
        f.write_all(content.as_bytes()).expect("write tmpfile");
        f
    }

    #[test]
    fn load_minimal_valid_json() {
        let tmp = write_tmp(MINIMAL_JSON);
        let cfg = PiperVoiceConfig::load(tmp.path()).expect("load ok");
        assert_eq!(cfg.audio_sample_rate, 22_050);
        assert_eq!(cfg.espeak_voice, "es");
        // Defaults aplicados a `inference.*` (campo ausente).
        assert!((cfg.length_scale - 1.0).abs() < f32::EPSILON);
        assert!((cfg.noise_scale - 0.667).abs() < f32::EPSILON);
        assert!((cfg.noise_w - 0.8).abs() < f32::EPSILON);
        assert_eq!(cfg.num_symbols, 256);
        assert_eq!(cfg.num_speakers, 1);
        assert_eq!(cfg.phoneme_id_map.get("^"), Some(&vec![1]));
        assert_eq!(cfg.phoneme_id_map.get("_"), Some(&vec![0]));
        assert_eq!(cfg.phoneme_id_map.get("$"), Some(&vec![2]));
        assert_eq!(cfg.phoneme_id_map.get("a"), Some(&vec![10]));
    }

    #[test]
    fn load_with_explicit_inference_overrides_defaults() {
        let json = r#"{
            "audio": { "sample_rate": 16000 },
            "inference": {
                "noise_scale": 0.5, "length_scale": 1.5, "noise_w": 1.0
            },
            "phoneme_id_map": {}
        }"#;
        let tmp = write_tmp(json);
        let cfg = PiperVoiceConfig::load(tmp.path()).expect("load ok");
        assert_eq!(cfg.audio_sample_rate, 16_000);
        assert!((cfg.length_scale - 1.5).abs() < f32::EPSILON);
        assert!((cfg.noise_scale - 0.5).abs() < f32::EPSILON);
        assert!((cfg.noise_w - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn load_missing_audio_returns_error() {
        let json = r#"{ "phoneme_id_map": {} }"#;
        let tmp = write_tmp(json);
        let err = PiperVoiceConfig::load(tmp.path()).unwrap_err();
        match err {
            TtsError::InvalidVoiceConfig(msg) => {
                assert!(
                    msg.contains("audio"),
                    "mensaje debería mencionar 'audio', obtuve: {msg}"
                );
            }
            other => panic!("esperaba InvalidVoiceConfig, obtuve: {other:?}"),
        }
    }

    #[test]
    fn load_missing_phoneme_id_map_returns_error() {
        let json = r#"{ "audio": { "sample_rate": 22050 } }"#;
        let tmp = write_tmp(json);
        let err = PiperVoiceConfig::load(tmp.path()).unwrap_err();
        match err {
            TtsError::InvalidVoiceConfig(msg) => {
                assert!(
                    msg.contains("phoneme_id_map"),
                    "mensaje debería mencionar 'phoneme_id_map', obtuve: {msg}"
                );
            }
            other => panic!("esperaba InvalidVoiceConfig, obtuve: {other:?}"),
        }
    }

    #[test]
    fn load_malformed_json_returns_error() {
        let tmp = write_tmp("{ esto no es JSON valido ");
        let err = PiperVoiceConfig::load(tmp.path()).unwrap_err();
        match err {
            TtsError::InvalidVoiceConfig(_) => (),
            other => panic!("esperaba InvalidVoiceConfig, obtuve: {other:?}"),
        }
    }

    #[test]
    fn load_missing_file_returns_error() {
        let err = PiperVoiceConfig::load(Path::new("/nonexistent.json"))
            .unwrap_err();
        match err {
            TtsError::InvalidVoiceConfig(_) => (),
            other => panic!("esperaba InvalidVoiceConfig, obtuve: {other:?}"),
        }
    }

    #[test]
    fn load_zero_sample_rate_returns_error() {
        let json = r#"{
            "audio": { "sample_rate": 0 },
            "phoneme_id_map": {}
        }"#;
        let tmp = write_tmp(json);
        let err = PiperVoiceConfig::load(tmp.path()).unwrap_err();
        match err {
            TtsError::InvalidVoiceConfig(msg) => {
                assert!(
                    msg.contains("sample_rate"),
                    "mensaje debería mencionar 'sample_rate', obtuve: {msg}"
                );
            }
            other => panic!("esperaba InvalidVoiceConfig, obtuve: {other:?}"),
        }
    }
}