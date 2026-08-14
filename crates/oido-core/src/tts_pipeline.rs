//! Pipeline TTS (lectura de selección de cursor).
//!
//! ## Diseño
//!
//! Threading: `oido-tts-worker` (espejo de `oido-stt-{i}` en
//! `pipeline.rs`). Comunicación por canales `crossbeam` (regla R1):
//!
//! ```text
//! hotkey "leer selección"
//!       │
//!       ▼
//! [read_selection] ── texto ──▶ tts_tx (bounded 8) ──▶ oido-tts-worker
//!                                                       │
//!                                              chunker por oración
//!                                                       │
//!                                              engine.synthesize(chunk)
//!                                                       │
//!                                          [convertir AudioChunk]    │
//!                                                       ▼
//!                                              playback.enqueue(pcm)
//! ```
//!
//! ## Diferencias con `Pipeline` (STT)
//!
//! - **Sin captura de audio**: la entrada es texto (no PCM). Un solo
//!   thread worker basta: `ort::Session::run` ya paraleliza internamente
//!   con su threadpool intra-op, y serializar `Session::run` desde varios
//!   workers añadiría lock contention sin throughput.
//! - **Sin mutex compartido entre etapas**: el único estado mutable es
//!   `cancellation: Arc<AtomicBool>`.
//! - **Chunking por oración**: el texto se parte con
//!   `unicode_segmentation::UnicodeSegmentation::split_sentence_bounds`
//!   para limitar la latencia del primer audio y permitir cancelación
//!   inmediata entre oraciones.
//!
//! ## Conversión de AudioChunk
//!
//! Los dos crates (`oido-tts` y `oido-audio`) defienden un struct
//! `AudioChunk` con los mismos campos pero **distinto tipo** (no se
//! introdujo `oido-audio → oido-tts` dep para evitar acoplarlos — ver
//! F1 nota del agente). Esta pipeline convierte manualmente campo a
//! campo. Cuando F7 o una iteración futura unifique los tipos, esta
//! conversión desaparece.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use oido_audio::PlaybackSink as AudioPlaybackSink;
use oido_config::TtsEngineKind;
use oido_tts::{AudioChunk as TtsAudioChunk, Engine as TtsEngine};
use tracing::{debug, error, info, warn};

/// Convierte un `AudioChunk` de `oido-tts` a uno de `oido-audio`.
/// Trivial: misma estructura `{ samples: Vec<f32>, sample_rate_hz: u32 }`.
/// Ver doc del módulo.
fn audio_chunk_into_audio(tts: TtsAudioChunk) -> oido_audio::AudioChunk {
    oido_audio::AudioChunk {
        samples: tts.samples,
        sample_rate_hz: tts.sample_rate_hz,
    }
}

/// Configuración para arrancar el `TtsPipeline`.
#[derive(Debug)]
pub struct TtsPipelineConfig {
    /// Engine TTS activo. Ya cargado por el thread de lazy-loader
    /// (en `main.rs`) antes de que el pipeline arranque.
    pub engine: Arc<dyn TtsEngine>,
    /// Sink de audio sobre el que se reproducen los chunks sintetizados.
    pub playback: Arc<dyn AudioPlaybackSink>,
}

#[derive(Debug)]
pub struct TtsPipeline {
    cfg: TtsPipelineConfig,
    /// Canal hotkey → worker. Bounded 8: margen para pulsaciones
    /// consecutivas durante una lectura larga.
    text_tx: Sender<String>,
    text_rx: Receiver<String>,
    /// Flag de cancelación entre chunks. `AtomicBool` para no abrir un
    /// canal adicional: el worker chequea entre chunks; el setter lo
    /// hace `store(true)` desde cualquier thread (e.g. nueva selección
    /// del usuario).
    cancel: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl TtsPipeline {
    /// Crea el pipeline. NO arranca el worker todavía — eso lo hace
    /// `start()`. El engine debe estar cargado.
    pub fn new(cfg: TtsPipelineConfig) -> Self {
        let (text_tx, text_rx) = crossbeam_channel::bounded(8);
        Self {
            cfg,
            text_tx,
            text_rx,
            cancel: Arc::new(AtomicBool::new(false)),
            worker: None,
        }
    }

    /// `Sender` que el bin conecta al handler `TtsReadSelection(text)`
    /// del control loop.
    #[must_use]
    pub fn text_sink(&self) -> Sender<String> {
        self.text_tx.clone()
    }

    /// Comparte el handle de cancelación. El bin lo mueve al callback
    /// del hotkey TTS para cancelar lo pendiente entre re-selecciones.
    #[must_use]
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    /// Arranca el worker. Idempotente.
    pub fn start(&mut self) -> anyhow::Result<()> {
        if self.worker.is_some() {
            debug!("TtsPipeline::start llamado dos veces; ignorando");
            return Ok(());
        }

        // `Arc<dyn ...>` se clona barato: no hay movimiento del trait
        // object. El cfg en `self` queda intacto.
        let engine = Arc::clone(&self.cfg.engine);
        let playback = Arc::clone(&self.cfg.playback);
        let text_rx = self.text_rx.clone();
        let cancel = Arc::clone(&self.cancel);

        let worker = thread::Builder::new()
            .name("oido-tts-worker".into())
            .spawn(move || run_tts_worker(engine, playback, text_rx, cancel))?;
        self.worker = Some(worker);
        Ok(())
    }

    /// Detén el pipeline. Pone el flag de cancelación (para no quedarnos
    /// bloqueados en un chunk en vuelo) y dropea el canal.
    pub fn shutdown(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        // cerrar el canal → worker sale del recv()
    }
}

fn run_tts_worker(
    engine: Arc<dyn TtsEngine>,
    playback: Arc<dyn AudioPlaybackSink>,
    text_rx: Receiver<String>,
    cancel: Arc<AtomicBool>,
) {
    let engine_kind = engine.engine_kind();
    info!(
        engine = ?engine_kind,
        "arrancando worker TTS"
    );

    loop {
        let text = match text_rx.recv() {
            Ok(t) => t,
            Err(_) => {
                info!("canal TTS cerrado; saliendo del worker");
                return;
            }
        };

        // Reset cancelación para este lote.
        cancel.store(false, Ordering::SeqCst);

        // Salvaguarda: si el engine es Kokoro y el texto parece español,
        // advertimos. El bin debe haber pre-sustituido a Piper para
        // casos de español; esto es sólo logging — Kokoro mismo es el
        // que devolverá un error en su `synthesize` si no puede
        // phonemizar.
        if engine_kind == TtsEngineKind::Kokoro && looks_spanish(&text) {
            tracing::info!(
                "Kokoro sintetizando texto en español: {} chars",
                text.chars().count()
            );
        }

        // Chunking por oración + sub-chunking por longitud (para que
        // ningún chunk exceda MAX_PHONEMES del engine Piper/Kokoro).
        let mut chunker = SentenceChunker::new(&text);
        for chunk in chunker.by_ref() {
            // Checkpoint de cancelación entre chunks.
            if cancel.load(Ordering::SeqCst) {
                warn!("lectura cancelada por nueva selección");
                break;
            }

            // Síntesis. Bloquea (ort::Session::run es síncrono).
            let started = std::time::Instant::now();
            let pcm = match engine.synthesize(&chunk) {
                Ok(p) => p,
                Err(e) => {
                    error!(
                        ?e,
                        chunk_chars = chunk.chars().count(),
                        "síntesis TTS falló"
                    );
                    continue;
                }
            };
            let elapsed = started.elapsed();

            // Reproducción.
            let playback_chunk = audio_chunk_into_audio(pcm);
            if let Err(e) = playback.enqueue(playback_chunk) {
                error!(?e, "playback falló");
                continue;
            }

            info!(
                chunk_chars = chunk.chars().count(),
                elapsed_ms = elapsed.as_millis() as u64,
                "chunk sintetizado y encolado"
            );
        }
    }
}

/// Estado del chunker por oración. Vive en el closure del worker
/// (no necesita compartirse entre threads).
///
/// Estrategia: cortar en el primer `.`/`!`/`?`/`…` que vaya seguido de
/// un espacio o fin de string. Es deliberadamente simple — no usa
/// `unicode_segmentation` (cuya API de oraciones es costosa y sobrada
/// para nuestro caso). Un solo punto final suelto tras trim produce un
/// chunk vacío y se skipea.
///
/// **Sub-chunking por longitud**: si una oración excede
/// [`MAX_CHUNK_CHARS`], se subdivide adicionalmente en comas/puntos
/// medios/espacios para que el engine TTS no trunque los fonemas
/// internamente (Piper tiene `MAX_PHONEMES=128`; una oración de 305
/// chars produce ~260 fonemas y se truncaría sin este reparto).
struct SentenceChunker<'a> {
    remaining: &'a str,
    /// Sub-chunks pendientes de una oración que excedió
    /// `MAX_CHUNK_CHARS`. Se rellena en `next()` cuando una oración
    /// es demasiado larga; los siguientes `next()` los drena antes
    /// de volver a leer `remaining`.
    pending: std::collections::VecDeque<String>,
}

impl<'a> SentenceChunker<'a> {
    /// Construye un chunker nuevo. API equivalente a la anterior
    /// (`SentenceChunker { remaining: text }`), pero ahora requiere
    /// `::new(text)` para inicializar `pending` vacío.
    fn new(text: &'a str) -> Self {
        Self {
            remaining: text,
            pending: std::collections::VecDeque::new(),
        }
    }
}

const SENTENCE_TERMINATORS: &[char] = &['.', '!', '?', '…'];

/// Máximo de chars (UTF-8) por chunk emitido al engine. Calculado para
/// no traspasar el `MAX_PHONEMES` de Piper (256 IDs ≈ 126 fonemas
/// reales, ya que cada fonema emite 1 ID + 1 PAD). Empíricamente
/// medimos ~0.84 fonemas/char en español mezclado con timestamps/logs:
/// `126 / 0.84 ≈ 150`. Dejamos 150 como límite conservador para que ni
/// el peor caso (todo consonantes, fonema por char) trunque.
const MAX_CHUNK_CHARS: usize = 150;

/// Sub-delimitadores blandos: cuando una oración supera
/// `MAX_CHUNK_CHARS`, cortamos en uno de estos (en orden de
/// preferencia). `,` y `;` producen cortes naturales; `:` también. Si
/// ninguno aparece, cortamos en el primer espacio tras el límite.
const SOFT_SUBDELIMITERS: &[char] = &[',', ';', ':', '—', '–', '-'];

/// ¿el carácter `c` cuenta como final de oración?
#[inline]
fn is_terminator(c: char) -> bool {
    SENTENCE_TERMINATORS.contains(&c)
}

impl<'a> Iterator for SentenceChunker<'a> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        // 0) Drenar sub-chunks pendientes primero (de una oración larga
        //    que se subdividió en la iteración anterior).
        if let Some(chunk) = self.pending.pop_front() {
            return Some(chunk);
        }

        // 1) Saltar whitespace al principio de `remaining`.
        let trimmed = self.remaining.trim_start();
        if trimmed.is_empty() {
            return None;
        }
        // Ajustar el offset: el chunk que devolvamos pertenece al
        // `rest` (trimmed), pero el `remaining` siguiente empieza donde
        // terminaba el chunk dentro de `self.remaining` ORIGINAL. Lo
        // más simple: trabajar siempre con sub-slices de `self.remaining`
        // y ajustar offsets al final.
        let leading_ws_len = self.remaining.len() - trimmed.len();
        let rest_start_offset = leading_ws_len;

        // 2) Buscar el primer terminador en `trimmed`, donde el siguiente
        // carácter es whitespace O fin de string.
        let mut cut: Option<usize> = None;
        for (i, c) in trimmed.char_indices() {
            if is_terminator(c) {
                // ¿qué viene después?
                let after = i + c.len_utf8();
                let next_is_ws_or_end = trimmed[after..]
                    .chars()
                    .next()
                    .is_none_or(|n| n.is_whitespace());
                if next_is_ws_or_end {
                    cut = Some(after);
                    break;
                }
            }
        }

        let raw_sentence: String = match cut {
            // 3) Terminador encontrado: cortar inmediatamente después
            //    del `.`/`!`/`?` (sin comerse el espacio siguiente — el
            //    espacio lo absorbe el próximo `trim_start` del método).
            Some(local_end) => {
                let absolute_end = rest_start_offset + local_end;
                let chunk = self.remaining[..absolute_end].trim().to_owned();
                self.remaining = &self.remaining[absolute_end..];
                chunk
            }
            // 4) Sin terminador: chunk = todo `trimmed`. Si quedó algo
            //    no-vacío y no-solo-puntuación, lo emitimos; si no,
            //    terminamos.
            None => {
                let chunk = trimmed.to_owned();
                self.remaining = "";
                chunk
            }
        };

        // Skip si la oración es vacía o solo puntuación/whitespace.
        if raw_sentence.is_empty()
            || raw_sentence
                .chars()
                .all(|c| c.is_whitespace() || is_terminator(c))
        {
            return self.next();
        }

        // 5) Sub-chunking por longitud: si la oración cabe en
        //    `MAX_CHUNK_CHARS`, devolverla tal cual. Si no,
        //    subdividirla en bloques buscando sub-delimitadores blandos
        //    (`,`, `;`, etc.) y, como último recurso, espacios.
        if raw_sentence.chars().count() <= MAX_CHUNK_CHARS {
            return Some(raw_sentence);
        }
        let sub = subdivide_long_sentence(&raw_sentence, MAX_CHUNK_CHARS);
        // El primer sub-chunk lo devolvemos ahora; el resto entra a
        // `pending` para que las siguientes llamadas a `next()` los
        // drenen (paso 0 de arriba).
        let mut iter = sub.into_iter();
        let first = iter.next().unwrap_or(raw_sentence);
        for s in iter {
            self.pending.push_back(s);
        }
        Some(first)
    }
}

/// Subdivide una oración larga en chunks de como máximo `max_chars`
/// caracteres (UTF-8). Estrategia:
///
/// 1. Buscar el último sub-delimitador blando (`,`, `;`, `:`, etc.)
///    antes de `max_chars` → corte natural.
/// 2. Si no hay sub-delimitador, cortar en el último espacio antes
///    de `max_chars` (no partir palabras).
/// 3. Si no hay espacio (palabra larga > max_chars), cortar forzado
///    en `max_chars`.
///
/// Cada chunk emitido preserva el delimitador al final (mejor
/// entonación en el TTS — "Hola, mundo," no "Hola, mun").
fn subdivide_long_sentence(sentence: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = sentence;
    while rest.chars().count() > max_chars {
        // Buscar el índice en chars del mejor corte dentro de [0, max_chars].
        let window: String = rest.chars().take(max_chars).collect();
        // byte-offset del final del window dentro de `rest`.
        let window_byte_len = window.len();

        // 1) Último sub-delimitador blando dentro del window.
        // 2) Último espacio dentro del window.
        // 3) Corte forzado al final del window (palabra muy larga sin
        //    espacios dentro del límite).
        let cut_opt = window
            .rfind(SOFT_SUBDELIMITERS)
            .or_else(|| window.rfind(char::is_whitespace));

        // `chunk_end`: byte-offset del final del chunk dentro de `rest`.
        // Si encontramos delimitador o espacio, lo incluimos en este
        // chunk (`+1`) — produce cortes naturales ("Hola, mundo," en
        // vez de "Hola, mun"). Si no (corte forzado), cortamos exacto.
        let chunk_end = match cut_opt {
            Some(cut_byte) => (cut_byte + 1).min(rest.len()),
            None => window_byte_len,
        };
        let chunk = rest[..chunk_end].trim().to_owned();
        if !chunk.is_empty() {
            out.push(chunk);
        }
        rest = rest[chunk_end..].trim_start();
    }
    // Resto final.
    let final_chunk = rest.trim().to_owned();
    if !final_chunk.is_empty() {
        out.push(final_chunk);
    }
    // Si por alguna razón no emitiéramos nada (e.g. todo whitespace),
    // devolvemos al menos la oración original truncada — peor caso
    // silencioso que colgar el worker.
    if out.is_empty() {
        out.push(sentence.chars().take(max_chars).collect());
    }
    out
}

/// Heurística de "es texto español" para advertir cuando Kokoro
/// (que sólo tiene G2P EN en v1) recibe texto español. Mejora futura:
/// CLD3 o fasttext.
pub(crate) fn looks_spanish(text: &str) -> bool {
    let mut spanish_hits = 0usize;
    for c in text.chars().take(200) {
        match c {
            'á' | 'é' | 'í' | 'ó' | 'ú' | 'ñ' | '¿' | '¡' => spanish_hits += 1,
            _ => {}
        }
    }
    spanish_hits > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_chitter_splits_on_basic_punctuation() {
        let text = "Hola mundo. Adiós.";
        let mut chunker = SentenceChunker::new(text);
        let a = chunker.next().unwrap();
        let b = chunker.next().unwrap();
        assert_eq!(a.trim(), "Hola mundo.");
        assert_eq!(b.trim(), "Adiós.");
        assert!(chunker.next().is_none());
    }

    #[test]
    fn sentence_chitter_handles_text_without_terminator() {
        let text = "una sola frase sin punto";
        let mut chunker = SentenceChunker::new(text);
        let a = chunker.next().unwrap();
        assert_eq!(a.trim(), "una sola frase sin punto");
        assert!(chunker.next().is_none());
    }

    #[test]
    fn sentence_chitter_skips_punctuation_only_chunks() {
        // El primer bound cae en `".   "` — sólo puntuación → skip.
        let text = "   .   hola";
        let mut chunker = SentenceChunker::new(text);
        let a = chunker.next().unwrap();
        assert_eq!(a.trim(), "hola");
        assert!(chunker.next().is_none());
    }

    /// Una oración corta (≤ MAX_CHUNK_CHARS) se emite como un solo
    /// chunk — sin sub-división.
    #[test]
    fn short_sentence_is_single_chunk() {
        let text = "Hola mundo corto.";
        let mut chunker = SentenceChunker::new(text);
        assert_eq!(chunker.next().unwrap().trim(), "Hola mundo corto.");
        assert!(chunker.next().is_none());
    }

    /// Una oración larga se subdivide en chunks de ≤ MAX_CHUNK_CHARS.
    /// El caso real: el usuario selecciona un párrafo de ~300 chars sin
    /// puntos intermedios (e.g. una línea de log) y Piper trunca los
    /// fonemas sin este reparto.
    #[test]
    fn long_sentence_without_terminator_is_subdivided() {
        // 500 chars, sin puntuación fuerte ni débil: nos lleva al corte
        // forzado por espacio.
        let text = "a".repeat(500);
        let mut chunker = SentenceChunker::new(&text);
        let mut chunks = Vec::new();
        for c in chunker.by_ref() {
            chunks.push(c);
        }
        assert!(chunks.len() >= 2, "oración de 500 chars debe subdividirse");
        // Todos los chunks caben dentro del límite.
        for c in &chunks {
            assert!(
                c.chars().count() <= MAX_CHUNK_CHARS,
                "chunk de {} chars excede el límite {}",
                c.chars().count(),
                MAX_CHUNK_CHARS
            );
        }
        // La concatenación preserva todo el texto (pérdida de whitespace
        // entre chunks es OK — Piper lo ignora).
        let total: String = chunks.iter().cloned().collect();
        assert_eq!(total.chars().filter(|c| *c == 'a').count(), 500);
    }

    /// Sub-división respeta comas como puntos de corte naturales.
    /// Salida: cada chunk termina en `,` cuando cae en un corte.
    #[test]
    fn subdivided_long_sentence_prefers_comma_as_cut() {
        // 300 chars con comas cada ~50.
        let mut s = String::new();
        for i in 0..6 {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&"a".repeat(45));
            s.push(',');
        }
        // Total ~280 chars + comas.
        let chunks = subdivide_long_sentence(&s, MAX_CHUNK_CHARS);
        assert!(chunks.len() >= 2, "oración larga debe subdividirse");
        // Al menos un chunk debe terminar en coma (preferencia por corte natural).
        let any_comma_terminated = chunks.iter().any(|c| c.trim_end().ends_with(','));
        assert!(
            any_comma_terminated,
            "al menos un chunk debe cortar en coma: {:?}",
            chunks
        );
    }

    /// Sub-división nunca excede `max_chars` por chunk (límite duro).
    #[test]
    fn subdivided_chunks_never_exceed_max() {
        let text = "palabra muy larga sin espacios ".repeat(20);
        let chunks = subdivide_long_sentence(&text, 50);
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                c.chars().count() <= 50,
                "chunk {} de {} chars excede 50: {:?}",
                i,
                c.chars().count(),
                c
            );
        }
    }

    #[test]
    fn looks_spanish_rejects_pure_english() {
        assert!(!looks_spanish("hello world"));
        assert!(!looks_spanish(""));
    }

    #[test]
    fn looks_spanish_accepts_accents() {
        assert!(looks_spanish("El niño come mañana"));
        assert!(looks_spanish("¿Cómo estás?"));
    }

    #[test]
    fn audio_chunk_conversion_is_field_to_field() {
        let tts = oido_tts::AudioChunk {
            samples: vec![1.0, 2.0],
            sample_rate_hz: 24000,
        };
        let aud = audio_chunk_into_audio(tts.clone());
        assert_eq!(aud.samples, tts.samples);
        assert_eq!(aud.sample_rate_hz, tts.sample_rate_hz);
    }
}
