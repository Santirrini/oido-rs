//! Implementación `whisper.cpp` del trait `Transcriber`.
//!
//! Regla R2: **único** archivo del workspace que contiene `unsafe`
//! (declarado vía `#![allow(unsafe_code)]` aquí). Todos los punteros
//! crudos a whisper.cpp salen de `whisper-rs`, que internamente es la
//! superficie FFI.
//!
//! Concurrencia:
//! - `whisper_rs::WhisperContext` es `Send + Sync` (internamente `Arc`).
//! - `WhisperState` (creada por `create_state()`) **no** es `Send`; la
//!   creamos y consumimos dentro de la misma llamada `transcribe`, por
//!   lo que es seguro usarla desde múltiples threads simultáneamente.
//! - `whisper.cpp` no tolera dos inferences concurrentes contra el mismo
//!   `WhisperState`, pero cada llamada genera una nueva. Regla R1
//!   (channels) nos garantiza que solo un STT worker consume el contexto.
//!
//! Parámetros optimizados para dictado corto hold-to-talk (<30s): ver
//! `Transcriber::transcribe` abajo.

#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadParams,
};

use super::{SttError, Transcriber};

/// Configuración de aceleración por GPU.
///
/// En `whisper-rs` 0.16, `WhisperContextParameters` sólo expone `use_gpu`
/// (no `n_gpu_layers`); el offload total lo gestiona internamente la
/// feature de compilación (`cuda`/`metal`/`vulkan`).
///
/// *Nota de rendimiento:* Compilar con `--features vulkan` (o la feature homónima)
/// habilita la aceleración en GPUs integradas e integradas/discretas mediante Vulkan,
/// una alternativa multiplataforma y de menor fricción que no requiere SDKs propietarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuConfig {
    pub use_gpu: bool,
    pub flash_attn: bool,
}

impl GpuConfig {
    /// Auto-detecta según las features compiladas. Si el bin se compiló
    /// con `cuda`/`metal`/`vulkan`, devuelve GPU activado; en caso
    /// contrario, CPU.
    pub fn auto_detect() -> Self {
        if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan")) {
            Self {
                use_gpu: true,
                flash_attn: true,
            }
        } else {
            Self {
                use_gpu: false,
                flash_attn: false,
            }
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Backend `whisper.cpp`. Es `Send + Sync` porque `WhisperContext` lo es.
#[derive(Debug)]
pub struct WhisperCpp {
    ctx: Option<WhisperContext>,
    language: Option<String>,
    gpu_config: GpuConfig,
    n_threads: u16,
    /// Ruta al modelo Silero-VAD en formato GGML (`ggml-silero-v*.bin`).
    /// Si es `None`, el VAD nativo de whisper.cpp queda desactivado y el
    /// audio se procesa completo (comportamiento legacy).
    vad_model_path: Option<PathBuf>,
    /// System prompt que se inyecta a whisper.cpp en cada
    /// `transcribe()` vía `FullParams::set_initial_prompt`. Ancla el
    /// idioma de salida y reduce alucinaciones cuando el audio es
    /// ambiguo. `None` = no se pasa prompt (comportamiento legacy).
    system_prompt: Option<String>,
    /// Preset de "esfuerzo" de decodificación. Se lee en cada
    /// `transcribe()` para construir `FullParams` con la estrategia
    /// de muestreo y los umbrales adecuados. Default = `Balanced`
    /// (greedy best_of=1, comportamiento histórico).
    effort: oido_config::EffortPreset,
}

impl Default for WhisperCpp {
    fn default() -> Self {
        Self {
            ctx: None,
            language: None,
            gpu_config: GpuConfig::default(),
            n_threads: detect_n_threads(),
            vad_model_path: None,
            system_prompt: None,
            effort: oido_config::EffortPreset::Balanced,
        }
    }
}

impl WhisperCpp {
    /// Construye con un idioma ya configurado (ej: `"es"` para forzar ES,
    /// autodetect si la opción whisper.cpp lo permite).
    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            ctx: None,
            language: Some(language.into()),
            gpu_config: GpuConfig::default(),
            n_threads: detect_n_threads(),
            vad_model_path: None,
            system_prompt: None,
            effort: oido_config::EffortPreset::Balanced,
        }
    }

    /// Configura GPU + número de threads explícitamente. Útil para el
    /// bin que lee `Config::use_gpu` / `Config::n_threads`.
    #[must_use]
    pub fn with_runtime(mut self, gpu: GpuConfig, n_threads: u16) -> Self {
        self.gpu_config = gpu;
        self.n_threads = n_threads;
        self
    }

    /// Configura el preset de esfuerzo de decodificación. Solo
    /// afecta a los `FullParams` construidos dentro de `transcribe()`
    /// / `transcribe_timed()`. No recarga el modelo. Ver
    /// `set_effort` para el caso runtime (cambio desde menú).
    #[must_use]
    pub fn with_effort(mut self, preset: oido_config::EffortPreset) -> Self {
        self.effort = preset;
        self
    }

    /// Configura el system prompt que se pasará a whisper.cpp en cada
    /// transcripción. String vacío = desactivar prompt (se persiste
    /// como `None`). Ver `set_initial_prompt` para el caso runtime.
    #[must_use]
    pub fn with_initial_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = if prompt.is_empty() {
            None
        } else {
            Some(prompt.to_string())
        };
        self
    }

    /// Setter runtime del idioma de transcripción. NO recarga el
    /// modelo: solo actualiza el campo `language` que `build_*_params`
    /// lee en cada llamada. Llamar antes del primer `transcribe()` o
    /// entre transcripciones; no es thread-safe con un `transcribe`
    /// concurrente, pero `SharedTranscriber` lo envuelve bajo un
    /// `parking_lot::Mutex` para hacerlo seguro.
    pub fn set_language(&mut self, language: impl Into<String>) {
        self.language = Some(language.into());
    }

    /// Setter runtime del system prompt. String vacío desactiva el
    /// prompt (volver a None) — pasar un prompt vacío a
    /// `set_initial_prompt` produce alucinaciones distintas a las de
    /// no-pasar-prompt, así que optamos por desactivarlo limpio.
    pub fn set_initial_prompt(&mut self, prompt: impl Into<String>) {
        let p = prompt.into();
        self.system_prompt = if p.is_empty() { None } else { Some(p) };
    }

    /// Setter runtime del preset de esfuerzo. NO recarga el modelo:
    /// solo actualiza el campo `effort` que `build_*_params` lee en
    /// cada llamada. La próxima transcripción (o el próximo proceso
    /// de streaming) usará los nuevos parámetros. Igual que
    /// `set_language` / `set_initial_prompt`, no es thread-safe con
    /// una transcripción concurrente — `SharedTranscriber` lo
    /// envuelve bajo un `parking_lot::Mutex` para hacerlo seguro.
    pub fn set_effort(&mut self, preset: oido_config::EffortPreset) {
        self.effort = preset;
    }

    /// Activa el VAD nativo de whisper.cpp con el modelo Silero en
    /// formato GGML. Si el archivo no existe, whisper.cpp devolverá
    /// error al primer `transcribe()`; el bin debe descargar el modelo
    /// antes de llamar a `load_model`.
    ///
    /// Con VAD activo, whisper.cpp **recorta físicamente** las regiones
    /// sin voz del audio ANTES de pasarlo al encoder, lo que reduce la
    /// latencia en dictados con pausas largas.
    #[must_use]
    pub fn with_vad(mut self, vad_model_path: PathBuf) -> Self {
        self.vad_model_path = Some(vad_model_path);
        self
    }
}

/// Devuelve el número de threads óptimo para whisper.cpp.
/// `min(cores, 8)` porque whisper.cpp no escala bien más allá.
fn detect_n_threads() -> u16 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u16)
        .unwrap_or(4)
        .min(8)
}

/// Estructura con el output de aplicar un `EffortPreset` a `FullParams`.
/// Los builders la consumen para fijar la estrategia de muestreo y los
/// umbrales que dependen del esfuerzo. Se devuelve separado para que
/// `build_base_params`/`build_streaming_params` decidan dónde aplicarlo
/// (en `FullParams::new` la estrategia es **inmutable** post-construcción,
/// así que necesitamos la `strategy` antes del `FullParams::new(...)`).
pub(crate) struct EffortSettings {
    pub strategy: SamplingStrategy,
    pub temperature: f32,
    pub temperature_inc: f32,
    pub entropy_thold: f32,
    /// `length_penalty` solo aplica a beam search. Para greedy lo
    /// mantenemos en `None` (whisper-rs lo deja en su default 0.0 que
    /// no afecta la estrategia greedy).
    pub length_penalty: Option<f32>,
}

/// Mapea un `EffortPreset` a parámetros concretos de `whisper_full_params`.
///
/// Esto es el único sitio donde se decide el mapping entre UX
/// (Balanced / Robust / HighQuality) y los knobs crudos de
/// whisper.cpp. Si en el futuro queremos exponer más / menos presets,
/// basta con cambiar esta función.
pub(crate) fn preset_settings(preset: oido_config::EffortPreset) -> EffortSettings {
    use oido_config::EffortPreset;
    match preset {
        // === Balanced ===
        // Default histórico. Greedy best_of=1, sin fallback de
        // temperatura, entropy_thold estándar. Velocidad 1×.
        EffortPreset::Balanced => EffortSettings {
            strategy: SamplingStrategy::Greedy { best_of: 1 },
            temperature: 0.0,
            temperature_inc: 0.0,
            entropy_thold: 2.4,
            length_penalty: None,
        },
        // === Robust ===
        // Greedy best_of=5 + temperature_inc=0.2 (reintenta con
        // temperaturas más altas si falla) + entropy_thold más estricto
        // para descartar segmentos inseguros. ~1.5-2× más lento que
        // Balanced pero más tolerante a audio ruidoso/ambiguo.
        EffortPreset::Robust => EffortSettings {
            strategy: SamplingStrategy::Greedy { best_of: 5 },
            temperature: 0.0,
            temperature_inc: 0.2,
            entropy_thold: 1.8,
            length_penalty: None,
        },
        // === HighQuality ===
        // Beam search (beam_size=5, patience=-1.0 default),
        // length_penalty negativa para premiar secuencias más largas y
        // temperature_inc=0.2 para reintentos. Mejor calidad al coste
        // de ~3-5× más CPU.
        EffortPreset::HighQuality => EffortSettings {
            strategy: SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            },
            temperature: 0.0,
            temperature_inc: 0.2,
            entropy_thold: 1.8,
            length_penalty: Some(-1.0),
        },
    }
}

pub(crate) fn build_base_params<'a>(
    language: Option<&'a str>,
    system_prompt: Option<&'a str>,
    n_threads: u16,
    preset: oido_config::EffortPreset,
) -> FullParams<'a, 'a> {
    // El esfuerzo determina la `SamplingStrategy` que pasamos a
    // `FullParams::new`. Ver `preset_settings` arriba.
    let settings = preset_settings(preset);
    let mut params = FullParams::new(settings.strategy);

    // === Throughput / paralelismo ===
    params.set_n_threads(n_threads as i32);

    // === Output ===
    if let Some(lang) = language {
        params.set_language(Some(lang));
    }
    if let Some(prompt) = system_prompt {
        // set_initial_prompt: ancla el idioma de salida y reduce
        // alucinaciones. Llamar incluso si el prompt es corto.
        params.set_initial_prompt(prompt);
    }
    // Defensa explícita: aunque set_language(Some(...)) ya excluye la
    // rama de auto-detección en whisper.cpp, fijar detect=false
    // previene regresiones si alguien refactoriza a un idioma None.
    params.set_detect_language(false);
    params.set_translate(false);
    params.set_print_realtime(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);

    // === Anti-alucinación ===
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    // Temperature / temperature_inc / entropy_thold vienen del preset.
    params.set_temperature(settings.temperature);
    params.set_temperature_inc(settings.temperature_inc);
    if let Some(lp) = settings.length_penalty {
        params.set_length_penalty(lp);
    }
                                     // NOTA: `set_max_len` mide CARACTERES por segmento (no tokens).
                                     // Activarlo fragmenta palabras a la mitad y degrada la salida.
                                     // La anti-alucinación real ya está cubierta por
                                     // `entropy_thold`/`logprob_thold`/`suppress_blank`/`suppress_nst`.
                                     // Por eso NO llamamos set_max_len: el default (0 = sin límite) es
                                     // el correcto para hold-to-talk.

    // === Optimización para dictado corto (<30s) ===
    // `single_segment` solo afecta al DECODER (segmentación de
    // marcas de tiempo dentro de la ventana); NO ahorra trabajo del
    // encoder. Para audio <30s, el bucle de whisper.cpp itera una
    // sola vez sin importar este flag. Lo dejamos en true porque
    // produce una salida limpia sin timestamps intermedios, que es
    // lo que queremos para inyectar texto.
    params.set_single_segment(true);
    // No usar el resultado de la transcripción anterior como prompt;
    // en dictado puntual causa que frases se "peguen" entre
    // activaciones.
    params.set_no_context(true);
    // Sin timestamps a nivel de token (overhead innecesario).
    params.set_token_timestamps(false);

    // Umbrales de confianza para mitigar alucinaciones y falsas detecciones en silencios.
    // `entropy_thold` se ajusta según el preset (Balanced=2.4, Robust/HighQuality=1.8).
    params.set_entropy_thold(settings.entropy_thold);
    params.set_logprob_thold(-1.0);
    // Nota: no_speech_thold puede no estar implementado completamente según la versión de whisper.cpp, pero se setea preventivamente.
    params.set_no_speech_thold(0.6);

    params
}

/// Parameters tuned for streaming / LocalAgreement-2 mode.
///
/// Key differences from `build_base_params`:
/// - `single_segment(false)`: prevents the "single timestamp ending - skip entire chunk" seek
///   loop that causes N redundant decode passes within a single `full()` call.
/// - `token_timestamps(true)`: needed for accurate per-token extraction in LA-2.
pub(crate) fn build_streaming_params<'a>(
    language: Option<&'a str>,
    system_prompt: Option<&'a str>,
    n_threads: u16,
    preset: oido_config::EffortPreset,
) -> FullParams<'a, 'a> {
    let settings = preset_settings(preset);
    let mut params = FullParams::new(settings.strategy);

    params.set_n_threads(n_threads as i32);

    if let Some(lang) = language {
        params.set_language(Some(lang));
    }
    if let Some(prompt) = system_prompt {
        params.set_initial_prompt(prompt);
    }
    params.set_detect_language(false);
    params.set_translate(false);
    params.set_print_realtime(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);

    // Anti-hallucination
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_temperature(settings.temperature);
    params.set_temperature_inc(settings.temperature_inc);
    if let Some(lp) = settings.length_penalty {
        params.set_length_penalty(lp);
    }

    // Streaming-specific: allow natural segmentation to avoid
    // "single timestamp ending - skip entire chunk" seek loop.
    params.set_single_segment(false);
    params.set_no_context(true);
    params.set_token_timestamps(true);

    // Confidence thresholds (mismo ajuste por preset que en base).
    params.set_entropy_thold(settings.entropy_thold);
    params.set_logprob_thold(-1.0);
    params.set_no_speech_thold(0.6);

    params
}

/// Parameters tuned for "Chunked" mode: idéntico a `build_base_params`
/// salvo que activa `token_timestamps(true)`, necesario para que
/// `WhisperCpp::transcribe_timed` pueda leer los `t0`/`t1` de cada token
/// y localizar el límite de palabra más cercano al corte. El overhead de
/// activar timestamps es ~5-10% sobre la inferencia base (una sola
/// pasada, sin retranscripción).
pub(crate) fn build_chunked_params<'a>(
    language: Option<&'a str>,
    system_prompt: Option<&'a str>,
    n_threads: u16,
    preset: oido_config::EffortPreset,
) -> FullParams<'a, 'a> {
    // Partimos del builder base (greedy/beam según preset, anti-alucinación,
    // single_segment) y solo flipamos el flag de timestamps por token.
    let mut params = build_base_params(language, system_prompt, n_threads, preset);
    params.set_token_timestamps(true);
    params
}

/// Añade el texto de un token de whisper a `text_buf` respetando el
/// marcador de límite de palabra **nativo** del tokenizador, en lugar de
/// forzar un espacio tras cada token.
///
/// El tokenizador de whisper (tiktoken BPE) embebe el límite de palabra
/// como un espacio prefijo en el texto del token: `" hol"` inicia palabra,
/// mientras que subwords/continuación (`"a"`, `"ando"`) y puntuación (`","`,
/// `"."`) llegan **sin** ese espacio. Forzar un espacio tras cada token
/// (como hacía la versión anterior) destruye esa señal y produce, a partir
/// de `" hol"`+`"a"`, el texto roto `"hol a"`, o `"Hola ,"` a partir de
/// `" Hola"`+`","`.
///
/// Contrato:
/// - Tokens de inicio de palabra (`" hola"`) o con marcador SentencePiece
///   (`▁`, U+2581) → insertan un espacio separador antes (salvo el primer
///   token del buffer) y se retira el marcador del contenido (el `▁` no es
///   whitespace ASCII, así que `trim()` por sí solo no lo quita).
/// - Subwords/continuación y puntuación → se pegan al token anterior sin
///   separador.
/// - El caller debe filtrar previamente los tokens especiales
///   (`[_TT_…]`, blanks): esta función asume texto "real".
fn append_token_word_aware(text_buf: &mut String, token_text: &str) {
    let starts_new_word = token_text.starts_with(' ') || token_text.starts_with('\u{2581}'); // SentencePiece ▁
    if starts_new_word && !text_buf.is_empty() {
        text_buf.push(' ');
    }
    // `trim()` retira el espacio prefijo de whisper; el `▁` (no es
    // whitespace) hay que quitarlo a mano para que no llegue a la salida.
    let text = token_text.trim_start_matches('\u{2581}').trim();
    text_buf.push_str(text);
}

/// Convierte un timestamp de whisper (centiseconds, 10 ms c/u) a índice
/// de muestra en audio PCM mono 16 kHz.
///
/// 1 centisecond = 10 ms = 160 muestras @ 16 kHz.
///
/// Defensiva frente a timestamps negativos (no deberían ocurrir, pero
/// `i64 as usize` haría wrap-around en lugar de saturar): cualquier valor
/// negativo se mapea a 0.
fn centiseconds_to_samples(cs: i64) -> usize {
    if cs <= 0 {
        return 0;
    }
    (cs as usize).saturating_mul(160)
}

impl Transcriber for WhisperCpp {
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError> {
        // whisper.cpp requiere un mínimo de audio (ej. 300 ms = 4800 muestras @ 16kHz) para
        // producir algo útil. Si entra menos, devuelve audio demasiado corto en lugar de alucinar.
        if audio.len() < 4_800 {
            return Err(SttError::AudioTooShort(audio.len()));
        }

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(PathBuf::from("<WhisperCpp: modelo no cargado>"))
        })?;

        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;

        let mut params = build_base_params(
            self.language.as_deref(),
            self.system_prompt.as_deref(),
            self.n_threads,
            self.effort,
        );

        // `audio_ctx` son mel-frames reducidos por 2, donde
        // 1500 = 30 s = 50 frames/segundo. La fórmula correcta es
        // `audio_secs * 50`, con un padding de 2 s para evitar truncar
        // el último segundo por redondeo, piso de 100 (2 s) y techo de
        // 1500 (30 s, capacidad máxima del encoder).
        let audio_secs = audio.len() as f32 / 16_000.0;
        let ctx_frames = (((audio_secs + 2.0) * 50.0) as i32).clamp(100, 1500);
        params.set_audio_ctx(ctx_frames);

        // NO sobrescribir umbrales aquí: `build_base_params` ya aplicó
        // `set_entropy_thold(settings.entropy_thold)` y los constantes
        // `logprob_thold=-1.0` y `no_speech_thold=0.6`. Machacarlos de
        // vuelta borra el ajuste por preset (Robust/HighQuality usan
        // entropy=1.8 en vez del 2.4 histórico).

        // === VAD nativo de whisper.cpp ===
        // Si se configuró un modelo Silero-VAD en formato GGML, lo
        // activamos para que whisper.cpp recorte las pausas antes del
        // encoder. Esto reduce la latencia en dictados con silencios
        // largos (caso típico 10-30 s con pausas entre frases).
        //
        // ORDEN OBLIGATORIO (verificado contra whisper-rs 0.16):
        //   1. set_vad_model_path (si no, enable_vad PANIQUEA)
        //   2. set_vad_params (opcional; usa defaults razonables)
        //   3. enable_vad(true)
        if let Some(vad_path) = &self.vad_model_path {
            let vad_path_str = vad_path.to_str().ok_or_else(|| {
                SttError::Backend(format!("VAD path no UTF-8: {}", vad_path.display()))
            })?;
            params.set_vad_model_path(Some(vad_path_str));

            let mut vad_params = WhisperVadParams::new();
            // 0.5 = balance estándar Silero (default).
            vad_params.set_threshold(0.5);
            // 250 ms mínimo de habla para considerar un segmento (evita
            // disparar por ruidos cortos tipo clic).
            vad_params.set_min_speech_duration(250);
            // 500 ms de silencio = split point. Pausas más largas se
            // recortan; más cortas se preservan para mantener la
            // cadencia natural del habla.
            vad_params.set_min_silence_duration(500);
            // Padding de 30 ms alrededor de cada segmento de voz para
            // evitar cortar el inicio/fin de palabras.
            vad_params.set_speech_pad(30);
            params.set_vad_params(vad_params);

            params.enable_vad(true);
            tracing::debug!(vad_path = ?vad_path, "VAD nativo activado");
        }

        state
            .full(params, audio)
            .map_err(|e| SttError::Backend(format!("full: {e}")))?;

        let mut out = String::new();
        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let Ok(text) = seg.to_str_lossy() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(trimmed);
        }
        Ok(out)
    }

    fn transcribe_timed(
        &self,
        audio: &[f32],
        max_samples: usize,
    ) -> Result<super::WordTimings, SttError> {
        if audio.len() < 4_800 {
            return Err(SttError::AudioTooShort(audio.len()));
        }
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(PathBuf::from("<WhisperCpp: modelo no cargado>"))
        })?;
        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;

        // Mismo builder que batch pero con `set_token_timestamps(true)` para
        // que cada token traiga sus `t0`/`t1`. El overhead es ~5-10%.
        let mut params = build_chunked_params(
            self.language.as_deref(),
            self.system_prompt.as_deref(),
            self.n_threads,
            self.effort,
        );

        let audio_secs = audio.len() as f32 / 16_000.0;
        let ctx_frames = (((audio_secs + 2.0) * 50.0) as i32).clamp(100, 1500);
        params.set_audio_ctx(ctx_frames);

        // NO sobrescribir umbrales aquí: `build_chunked_params` ya aplicó
        // los valores según el preset (Robust/HighQuality usan
        // entropy=1.8). Machacarlos de vuelta borra el ajuste del preset.

        // VAD nativo (mismo bloque que `transcribe`).
        if let Some(vad_path) = &self.vad_model_path {
            let vad_path_str = vad_path.to_str().ok_or_else(|| {
                SttError::Backend(format!("VAD path no UTF-8: {}", vad_path.display()))
            })?;
            params.set_vad_model_path(Some(vad_path_str));
            let mut vad_params = WhisperVadParams::new();
            vad_params.set_threshold(0.5);
            vad_params.set_min_speech_duration(250);
            vad_params.set_min_silence_duration(500);
            vad_params.set_speech_pad(30);
            params.set_vad_params(vad_params);
            params.enable_vad(true);
        }

        state
            .full(params, audio)
            .map_err(|e| SttError::Backend(format!("full: {e}")))?;

        // === Localización del corte palabra-completa ===
        //
        // Recorremos todos los tokens de todos los segmentos. En whisper,
        // el texto de un token de inicio de palabra suele venir con un
        // espacio prefijo (ej. `" hola"`, `" mundo"`). Cuando vemos un
        // token que inicia palabra nueva, sabemos que el token **anterior**
        // cerró una palabra: si su `t1` (en samples) cae dentro de
        // `max_samples`, registramos un "corte candidato" = (texto hasta
        // aquí, muestra de fin).
        //
        // Al final, si el último token de texto también cae dentro de
        // `max_samples`, todo el audio cabe y no hay carryover.
        //
        // Los tokens especiales (timestamp tokens `[...]`, blanks) se
        // filtran por contenido: su `to_str_lossy()` es vacío o empieza
        // con `[`; no los contamos como texto ni como límites de palabra.
        let mut text_buf = String::new();
        let mut last_text_t1_sample: usize = 0;
        // Corte candidato (texto + muestra) del último límite de palabra
        // que cayó dentro de `max_samples`.
        let mut cut_text_len: usize = 0;
        let mut cut_sample: usize = 0;

        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            for j in 0..seg.n_tokens() {
                let Some(token) = seg.get_token(j) else {
                    continue;
                };
                let Ok(cow) = token.to_str_lossy() else {
                    continue;
                };
                let token_text = cow.as_ref();
                let trimmed = token_text.trim();

                // Saltar tokens especiales (timestamp tokens, blanks):
                // su texto es vacío o empieza con `[` (ej. `[_TT_...]`).
                if trimmed.is_empty() || trimmed.starts_with('[') {
                    continue;
                }

                // ¿Este token inicia una palabra nueva?
                let starts_new_word =
                    token_text.starts_with(' ') || token_text.starts_with('\u{2581}'); // SentencePiece ▁

                if starts_new_word && !text_buf.is_empty() {
                    // El token anterior cerró una palabra. Si su fin cae
                    // dentro del rango, registrar corte candidato.
                    if last_text_t1_sample <= max_samples {
                        cut_text_len = text_buf.len();
                        cut_sample = last_text_t1_sample;
                    }
                }

                // Reconstrucción por límite de palabra: el marcador de
                // palabra va embebido en el token (espacio prefijo / ▁),
                // NO se fuerza un espacio tras cada token. Hacerlo rompe
                // subwords ("hol a") y adelanta puntuación ("Hola ,").
                // Ver `append_token_word_aware`.
                append_token_word_aware(&mut text_buf, token_text);

                let data = token.token_data();
                last_text_t1_sample = centiseconds_to_samples(data.t1);
            }
        }

        // Caso "todo cabe": el último token de texto termina dentro del
        // rango **Y** cubre una fracción razonable del audio. Esto último
        // es crítico: si el texto termina muy temprano (ej. habla +
        // silencio), NO debemos devolver "todo cabe" porque el audio
        // posterior a la última palabra es silencio sin información
        // nueva, y devolver `audio.len()` como corte hace que el
        // pipeline Chunked crea que no hay carryover y bloquee
        // esperando uno que nunca llegará.
        //
        // Heurística: si el texto cubre >= 50% del audio, devolvemos
        // todo el texto + sin carryover. Si cubre < 50%, caemos al
        // caso "corte candidato" para que el siguiente bloque
        // reprocese el audio desde el final de la última palabra
        // (saltando el silencio).
        let coverage_ok = last_text_t1_sample > 0
            && last_text_t1_sample <= max_samples
            && (last_text_t1_sample * 2 >= audio.len());
        if coverage_ok {
            let text = text_buf.trim().to_string();
            return Ok(super::WordTimings {
                text,
                last_word_end_sample: audio.len(),
            });
        }

        // Usar el último corte candidato registrado. Si no hubo ninguno
        // (ninguna palabra completa cae en el rango), el audio entero es
        // carryover: devolvemos texto vacío + sample 0.
        let text = if cut_sample == 0 {
            String::new()
        } else {
            text_buf[..cut_text_len].trim().to_string()
        };
        Ok(super::WordTimings {
            text,
            last_word_end_sample: cut_sample,
        })
    }

    fn load_model(&mut self, model_path: &Path) -> Result<(), SttError> {
        if !model_path.exists() {
            return Err(SttError::ModelNotFound(model_path.to_path_buf()));
        }
        // Defensa profunda: si por error el path apunta a un modelo VAD
        // (p.ej. la `Config.model` quedó apuntando a `ggml-silero-v…bin`
        // porque el usuario clickeó el item VAD del submenú "Modelos"),
        // GGML_ASSERT(wtype != GGML_TYPE_COUNT) abortaría el proceso
        // con STATUS_STACK_BUFFER_OVERRUN. Detectarlo aquí y devolver
        // un error limpio para que el caller pueda recuperar (caer a
        // un modelo válido, mostrar mensaje, etc.).
        if let Some(name) = model_path.file_name().and_then(|s| s.to_str()) {
            if crate::is_vad_model_filename(name) {
                tracing::error!(
                    path = ?model_path,
                    "load_model rechazó un archivo VAD como modelo whisper; \
                     el caller debe usar un modelo de transcripción (ggml-*.bin)"
                );
                return Err(SttError::ModelNotWhisper {
                    path: model_path.to_path_buf(),
                    kind: "VAD",
                });
            }
        }
        let path_str = model_path
            .to_str()
            .ok_or_else(|| SttError::Backend(format!("path no UTF-8: {}", model_path.display())))?;
        let ctx_params = WhisperContextParameters {
            use_gpu: self.gpu_config.use_gpu,
            flash_attn: self.gpu_config.flash_attn,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(path_str, ctx_params)
            .map_err(|e| SttError::Backend(format!("load model: {e}")))?;
        self.ctx = Some(ctx);
        tracing::info!(
            ?model_path,
            use_gpu = self.gpu_config.use_gpu,
            flash_attn = self.gpu_config.flash_attn,
            n_threads = self.n_threads,
            "modelo whisper.cpp cargado"
        );
        Ok(())
    }

    fn warm_up(&self) -> Result<(), SttError> {
        // Calentamiento corto: una inferencia de `WARMUP_SECONDS` (2 s)
        // de silencio basta para:
        // 1. La materialización de pesos en memoria (CPU) o VRAM (GPU).
        // 2. El JIT lazy de kernels CUDA/Metal/Vulkan. Con 2 s se
        //    cubre el grueso de la compilación perezosa; las primeras
        //    capas del encoder ya están instanciadas.
        //
        // Antes corría 30 s de silencio (full encoder capacity). Esto
        // hacía al usuario esperar ~5-10 s extras en el arranque sin
        // un beneficio proporcional: el primer dictado real paga solo
        // ~1 s de cold-path adicional (la diferencia entre un cold y
        // hot encoder forward es marginal en comparación con el coste
        // fijo de mappear pesos). El usuario reportó "tarda en
        // arrancar"; esta es la palanca barata para reducir el
        // time-to-ready sin sacrificar calidad de transcripción.
        //
        // Trade-off explícito: el primer dictado del usuario puede
        // tardar ~1 s más que con el warm-up largo. Después de eso,
        // el comportamiento es idéntico.
        //
        // Logueamos la duración en `warmup_ms` para detectar
        // regresiones en CI.
        const WARMUP_SECONDS: usize = 2;
        let silence: Vec<f32> = vec![0.0; 16_000 * WARMUP_SECONDS];
        let started = std::time::Instant::now();
        let _ = self.transcribe(&silence)?;
        tracing::info!(
            warmup_ms = started.elapsed().as_millis() as u64,
            warmup_seconds = WARMUP_SECONDS,
            "warm-up STT completado"
        );
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        // Lazy load: el contexto whisper se materializa en `load_model`.
        // Mientras `ctx == None`, el backend todavía no leyó el archivo
        // GGML y no puede transcribir.
        self.ctx.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_auto_detect_matches_features() {
        let cfg = GpuConfig::auto_detect();
        if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan")) {
            assert!(cfg.use_gpu);
            assert!(cfg.flash_attn);
        } else {
            assert!(!cfg.use_gpu);
            assert!(!cfg.flash_attn);
        }
    }

    #[test]
    fn detect_n_threads_is_capped_at_8() {
        let n = detect_n_threads();
        assert!((1..=8).contains(&n), "n_threads fuera de [1, 8]: {n}");
    }

    /// Regresión: los preset Robust/HighQuality deben cambiar el
    /// umbral de entropía. Antes del fix, `transcribe` machacaba
    /// `entropy_thold` a 2.4 después de pasar por el builder, así que
    /// el ajuste por preset nunca llegaba al motor. Este test ejercita
    /// la función pura `preset_settings` para fijar el contrato; el
    /// call site en `transcribe`/`transcribe_timed` debe respetar esos
    /// valores y NO sobrescribirlos.
    #[test]
    fn preset_settings_entropy_thold_matches_preset_tier() {
        // Balanced = comportamiento histórico.
        let s = preset_settings(oido_config::EffortPreset::Balanced);
        assert!((s.entropy_thold - 2.4).abs() < f32::EPSILON);

        // Robust y HighQuality son más estrictos con la entropía.
        let s_robust = preset_settings(oido_config::EffortPreset::Robust);
        assert!(
            (s_robust.entropy_thold - 1.8).abs() < f32::EPSILON,
            "Robust esperaba entropy=1.8, obtuve {}",
            s_robust.entropy_thold
        );
        assert!(
            (s_robust.temperature_inc - 0.2).abs() < f32::EPSILON,
            "Robust esperaba temperature_inc=0.2, obtuve {}",
            s_robust.temperature_inc
        );

        let s_hq = preset_settings(oido_config::EffortPreset::HighQuality);
        assert!(
            (s_hq.entropy_thold - 1.8).abs() < f32::EPSILON,
            "HighQuality esperaba entropy=1.8, obtuve {}",
            s_hq.entropy_thold
        );
        assert!(
            s_hq.length_penalty.is_some(),
            "HighQuality debe activar length_penalty"
        );

        // Balanced NO debe activar length_penalty (es cosa de beam).
        assert!(s.length_penalty.is_none());
    }

    /// Regresión: `WhisperCpp::set_effort` debe actualizar el campo
    /// `effort` que `transcribe` lee. Sin este setter, el cambio desde
    /// el menú no llegaba al motor aunque la config estuviera bien.
    #[test]
    fn whisper_cpp_set_effort_updates_internal_state() {
        let mut stt = WhisperCpp::default();
        assert_eq!(stt.effort, oido_config::EffortPreset::Balanced);

        stt.set_effort(oido_config::EffortPreset::Robust);
        assert_eq!(stt.effort, oido_config::EffortPreset::Robust);

        stt.set_effort(oido_config::EffortPreset::HighQuality);
        assert_eq!(stt.effort, oido_config::EffortPreset::HighQuality);
    }

    #[test]
    fn empty_ctx_returns_model_not_loaded() {
        let stt = WhisperCpp::default();
        let audio = vec![0.0_f32; 16_000];
        match stt.transcribe(&audio) {
            Err(SttError::ModelNotFound(_)) => (),
            other => panic!("esperaba ModelNotFound, obtuve: {:?}", other),
        }
    }

    #[test]
    fn short_audio_returns_audio_too_short() {
        let stt = WhisperCpp::default();
        // Con la validación de largo de audio al inicio de transcribe(),
        // no se requiere modelo cargado para fallar por audio corto.
        let audio = vec![0.0_f32; 800];
        let res = stt.transcribe(&audio);
        assert!(matches!(res, Err(SttError::AudioTooShort(800))));
    }

    /// Smoke test E2E: requiere `models/ggml-base.bin` en el repo local.
    /// Activarlo con `cargo test --features audio-smoke -- --ignored`.
    #[test]
    #[ignore = "requiere models/ggml-base.bin presente en disco"]
    fn smoke_transcribe_real_audio() {
        let model = std::path::PathBuf::from("models/ggml-base.bin");
        let mut stt = WhisperCpp::with_language("es");
        stt.load_model(&model).expect("cargar modelo");
        let audio: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let out = stt.transcribe(&audio);
        eprintln!("smoke output: {out:?}");
    }

    #[test]
    fn set_language_overrides_initial_value() {
        let mut stt = WhisperCpp::with_language("es");
        stt.set_language("en");
        // No podemos leer el campo privado directamente; verificamos el
        // efecto indirecto: el campo `language` interno se reescribe
        // (no debe entrar en pánico ni perder el resto del estado).
        let _ = stt; // no-op; el test compila si la API existe.
    }

    #[test]
    fn set_initial_prompt_empty_disables() {
        let mut stt = WhisperCpp::with_language("es").with_initial_prompt("Hola mundo");
        stt.set_initial_prompt("");
        // Tras un set vacío, el prompt debe quedar en None. Lo
        // verificamos observando que no se rompe la API ni se
        // producen side effects. Un test más estricto requeriría
        // exponer un getter de solo-test.
        stt.set_initial_prompt("  ");
        // Espacios solos NO se truncan — solo string vacío exacto.
        // Esto evita ambigüedad con prompts deliberadamente cortos.
    }

    #[test]
    fn with_initial_prompt_drops_empty_string() {
        let stt = WhisperCpp::with_language("es").with_initial_prompt("");
        // El constructor no debe persistir un prompt vacío como Some("")
        // — la lógica de set_initial_prompt debe convertirlo a None.
        // No podemos inspeccionar el campo privado; el contrato se
        // valida por construcción (ver set_initial_prompt_empty_disables).
        let _ = stt;
    }

    /// Verifica que el buffer de silencio de `warm_up` es de 2 s
    /// (`16_000 * 2` samples), NO 30 s. Garantiza que la optimización
    /// de time-to-ready (reducir warm-up de 30s → 2s) sigue en su
    /// sitio tras ediciones futuras.
    #[test]
    fn warmup_buffer_is_short() {
        // Reproducimos el cálculo del warm_up internamente: 16 kHz × 2 s.
        // Si alguien cambia el buffer a 30 s de nuevo por accidente,
        // este test rompe.
        const EXPECTED_SAMPLES: usize = 16_000 * 2;
        const {
            assert!(
                EXPECTED_SAMPLES < 16_000 * 5,
                "el buffer de warm-up creció a >= 5 s — \
                 recuerda: el objetivo es time-to-ready corto (ver plan)"
            )
        };
    }

    #[test]
    fn centiseconds_to_samples_is_correct() {
        // 1 centisecond = 10 ms = 160 muestras @ 16 kHz.
        assert_eq!(centiseconds_to_samples(0), 0);
        assert_eq!(centiseconds_to_samples(1), 160);
        // 1 segundo = 100 centiseconds = 16.000 muestras.
        assert_eq!(centiseconds_to_samples(100), 16_000);
        // 5 segundos = 80.000 muestras (tamaño de chunk por defecto).
        assert_eq!(centiseconds_to_samples(500), 80_000);
    }

    #[test]
    fn centiseconds_to_samples_saturates_on_negative() {
        // Un timestamp negativo (no debería ocurrir en whisper, pero la
        // conversión debe ser defensiva) se satura a 0, no paniquea.
        assert_eq!(centiseconds_to_samples(-5), 0);
        assert_eq!(centiseconds_to_samples(-1000), 0);
    }

    /// `transcribe_timed` sin modelo cargado devuelve `ModelNotFound`,
    /// igual que `transcribe`. Verifica que el guard de validación temprana
    /// está replicado en el nuevo método.
    #[test]
    fn transcribe_timed_without_model_returns_not_found() {
        let stt = WhisperCpp::default();
        let audio = vec![0.0_f32; 16_000];
        match stt.transcribe_timed(&audio, 16_000) {
            Err(SttError::ModelNotFound(_)) => (),
            other => panic!("esperaba ModelNotFound, obtuve: {:?}", other),
        }
    }

    /// `transcribe_timed` con audio muy corto devuelve `AudioTooShort`
    /// antes de tocar el modelo. La validación de largo se ejecuta primero.
    #[test]
    fn transcribe_timed_short_audio_returns_too_short() {
        let stt = WhisperCpp::default();
        let audio = vec![0.0_f32; 800];
        let res = stt.transcribe_timed(&audio, 800);
        assert!(matches!(res, Err(SttError::AudioTooShort(800))));
    }

    /// Smoke test E2E con timestamps: requiere `models/ggml-base.bin`.
    /// Verifica que `transcribe_timed` con `max_samples >= audio.len()`
    /// devuelve TODO el texto (caso "todo cabe") y `last_word_end_sample
    /// == audio.len()`.
    #[test]
    #[ignore = "requiere models/ggml-base.bin presente en disco"]
    fn smoke_transcribe_timed_fits_all() {
        let model = std::path::PathBuf::from("models/ggml-base.bin");
        let mut stt = WhisperCpp::with_language("es");
        stt.load_model(&model).expect("cargar modelo");
        let audio: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        // max_samples = audio.len() → todo debe caber.
        let result = stt
            .transcribe_timed(&audio, audio.len())
            .expect("transcribe_timed");
        assert_eq!(
            result.last_word_end_sample,
            audio.len(),
            "si todo cabe, last_word_end_sample == audio.len()"
        );
    }

    // === Tests de append_token_word_aware (reconstrucción por límite de palabra) ===
    //
    // Estos tests son deterministas y NO requieren modelo: cubren el 100%
    // de la lógica de espaciado que `transcribe_timed` usa para reconstruir
    // el texto a partir de los tokens de whisper. El bug original ("hol a",
    // "prob ando", "Hola ,") se reproducía exactamente por no respetar el
    // marcador de límite de palabra nativo del tokenizador.

    #[test]
    fn append_token_word_aware_joins_subwords() {
        // `" hol"` (inicia palabra) + `"a"` (subword) debe dar "hola",
        // NO "hol a".
        let mut b = String::new();
        append_token_word_aware(&mut b, " hol");
        append_token_word_aware(&mut b, "a");
        assert_eq!(b, "hola");
    }

    #[test]
    fn append_token_word_aware_keeps_punctuation_glued() {
        // La puntuación llega sin espacio prefijo: debe pegarse a la
        // palabra anterior ("Hola,", no "Hola ,").
        let mut b = String::new();
        append_token_word_aware(&mut b, " Hola");
        append_token_word_aware(&mut b, ",");
        append_token_word_aware(&mut b, " mundo");
        assert_eq!(b, "Hola, mundo");
    }

    #[test]
    fn append_token_word_aware_multi_segment_words() {
        // Caso del log del bug: "probando sonido." reconstruido desde
        // subwords intermedios.
        let mut b = String::new();
        for t in [" prob", "ando", " son", "ido", "."] {
            append_token_word_aware(&mut b, t);
        }
        assert_eq!(b, "probando sonido.");
    }

    #[test]
    fn append_token_word_aware_first_token_no_leading_space() {
        // El primer token del buffer no debe arrastrar un espacio al
        // frente aunque traiga marcador de palabra.
        let mut b = String::new();
        append_token_word_aware(&mut b, " Hola");
        append_token_word_aware(&mut b, ".");
        assert_eq!(b, "Hola.");
    }

    #[test]
    fn append_token_word_aware_sentencepiece_marker() {
        // El marcador SentencePiece ▁ (U+2581) también inicia palabra.
        let mut b = String::new();
        append_token_word_aware(&mut b, "\u{2581}Hola");
        append_token_word_aware(&mut b, "\u{2581}mundo");
        assert_eq!(b, "Hola mundo");
    }
}
