//! Pipeline "Chunked": fragmenta audios largos en bloques de ~5s y los
//! transcribe al vuelo, pegando incrementalmente cada bloque.
//!
//! ## Motivación
//!
//! El modo `Batch` transcribe TODO el audio de la sesión al soltar la
//! tecla. Para audios >30s, la latencia percibida es alta (la inferencia
//! de whisper escala linealmente) y el usuario no ve nada hasta el final.
//! El modo `Streaming` retranscribe el buffer completo cada 1s (coste
//! cuadrático) y sólo confirma prefijos estables.
//!
//! `Chunked` parte la grabación en bloques independientes de `chunk_secs`
//! y los transcribe a medida que se llenan. La transcripción de cada
//! bloque se pega en el cursor (`Ctrl+V`), dándole al usuario feedback
//! en tiempo real mientras sigue hablando.
//!
//! ## Corte palabra-completa (carryover)
//!
//! El corte entre bloques no es a tiempo fijo: usa `transcribe_timed`
//! para localizar el límite de palabra completa más cercano a
//! `CHUNK_SIZE`. El audio sobrante (la palabra cortada a medias) se pasa
//! como "carryover" al inicio del siguiente bloque, evitando truncar
//! palabras en la frontera.
//!
//! ```text
//! cpal → [audio_rx] → consumer thread ─┐
//!                                       │ (cuando buffer ≥ CHUNK_SIZE)
//!                                       ▼
//!                             [chunk_tx bounded 4]
//!                                       │
//!                                       ▼
//!                               worker STT (1 hilo)
//!                              transcribe_timed()
//!                              injector.inject(text)
//!                                       │
//!                              carryover = chunk[cut..]
//!                                       │
//!                                       ▼
//!                            [carryover_tx bounded 1]
//!                                       │
//!                        ◄───────────────┘ (se prepone al próximo chunk)
//! ```
//!
//! ## Backpressure
//!
//! El consumer **espera** al carryover del bloque anterior antes de
//! cortar el siguiente. Si el worker tarda más de `chunk_secs` (CPU
//! lenta), el audio se sigue acumulando y el siguiente chunk crece más
//! allá de 5s. Un cap de `MAX_BUFFER_SAMPLES` (30s) descarta audio viejo
//! con un `warn` antes de agotar memoria.
//!
//! ## Reglas del proyecto
//!
//! - **R1:** todo el paso de mensajes es `crossbeam::channel`. El único
//!   mutex (`BufferState`) está acotado a `samples`/`recording`, igual
//!   que en `Pipeline` y `StreamingPipeline`.
//! - **R2:** sin `unsafe`. `transcribe_timed` usa la API safe de
//!   whisper-rs.
//! - **R3:** `parking_lot::Mutex`, sin `.unwrap()` en locks.

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;

use oido_audio::{AudioRx, AudioTx, CaptureSource, Resampler};
use oido_hotkey::Hotkey;
use oido_input::Injector;
use oido_stt::{Transcriber, WordTimings};

use crate::phrase_filter;

// Reusamos los tipos de estado del pipeline batch.
use crate::pipeline::{PipelineEvent, PipelineState};

/// Configuración para arrancar el pipeline "Chunked".
#[derive(Debug)]
pub struct ChunkedPipelineConfig {
    pub capture: Box<dyn CaptureSource>,
    pub transcriber: Arc<dyn Transcriber>,
    pub injector: Arc<dyn Injector>,
    pub hotkey: Box<dyn Hotkey>,
    /// Binding canónico del hotkey (ej. "F8").
    pub hotkey_binding: String,
    /// Duración de cada bloque en segundos. Default recomendado: 5.0
    /// (80.000 muestras @ 16 kHz). Valores más bajos aumentan el
    /// overhead de inferencia; más altos aumentan la latencia percibida.
    pub chunk_duration_secs: f32,
}

/// Capacidad del canal de chunks (consumer → worker). 4 da margen para
/// ráfagas sin riesgo de OOM ni bloqueo del consumer.
const CHUNK_QUEUE_CAP: usize = 4;

/// Cap anti-OOM: si el buffer excede 30s de audio (porque el worker
/// está atascado o la CPU no da abasto), se descarta lo más viejo. 30s
/// es también el techo del encoder de whisper (`audio_ctx` máx = 1500
/// mel-frames); no tiene sentido acumular más.
const MAX_BUFFER_SECS: usize = 30;

/// Tipo del canal de chunks: vector de muestras listas para transcribir.
type ChunkJob = Vec<f32>;

#[derive(Default, Debug)]
struct BufferState {
    samples: Vec<f32>,
    recording: bool,
}

/// Pipeline que fragmenta audios largos en bloques transcritos al vuelo.
#[derive(Debug)]
pub struct ChunkedPipeline {
    cfg: ChunkedPipelineConfig,
    chunk_size: usize,
    recording: Arc<Mutex<BufferState>>,
    audio_tx: AudioTx,
    audio_rx: AudioRx,
    /// Canal consumer → worker: bloques de audio listos para STT.
    chunk_tx: Sender<ChunkJob>,
    chunk_rx: Receiver<ChunkJob>,
    /// Canal worker → consumer: carryover del bloque anterior que se
    /// prepone al inicio del próximo chunk.
    carryover_tx: Sender<Vec<f32>>,
    carryover_rx: Receiver<Vec<f32>>,
    event_tx: Sender<PipelineEvent>,
    event_rx: Receiver<PipelineEvent>,
    audio_consumer: Option<JoinHandle<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ChunkedPipeline {
    /// Crea el pipeline sin arrancarlo. Calcula `chunk_size` a partir de
    /// `chunk_duration_secs` (redondeado a muestras @ 16 kHz).
    pub fn new(cfg: ChunkedPipelineConfig) -> Self {
        let (audio_tx, audio_rx) = crossbeam_channel::bounded(1024);
        let (chunk_tx, chunk_rx) = crossbeam_channel::bounded(CHUNK_QUEUE_CAP);
        let (carryover_tx, carryover_rx) = crossbeam_channel::bounded(1);
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let chunk_size = (cfg.chunk_duration_secs * 16_000.0) as usize;
        Self {
            cfg,
            chunk_size,
            recording: Arc::new(Mutex::new(BufferState::default())),
            audio_tx,
            audio_rx,
            chunk_tx,
            chunk_rx,
            carryover_tx,
            carryover_rx,
            event_tx,
            event_rx,
            audio_consumer: None,
            worker: None,
        }
    }

    /// Receiver de eventos para el observador (bin → tray).
    #[must_use]
    pub fn events(&self) -> Receiver<PipelineEvent> {
        self.event_rx.clone()
    }

    /// Abre captura + arranca el consumer + el worker STT + registra el
    /// hotkey.
    pub fn start(&mut self) -> anyhow::Result<()> {
        // 1) capture.open() cablea cpal → audio_tx.
        self.cfg
            .capture
            .open(self.audio_tx.clone())
            .map_err(|e| anyhow::anyhow!("capture.open: {e}"))?;
        self.cfg
            .capture
            .start()
            .map_err(|e| anyhow::anyhow!("capture.start: {e}"))?;

        // Resampler (mismo patrón que Pipeline::start).
        let input_rate = self.cfg.capture.sample_rate_hz();
        let resampler = if Resampler::is_identity(input_rate) {
            None
        } else {
            match Resampler::new(input_rate) {
                Some(r) => {
                    tracing::info!(input_rate, output_rate = 16_000, "resampling activo");
                    Some(r)
                }
                None => {
                    tracing::warn!(input_rate, "no pude crear resampler");
                    None
                }
            }
        };

        let chunk_size = self.chunk_size;
        let max_buffer_samples = MAX_BUFFER_SECS * 16_000;

        // 2) Consumer thread: acumula audio y corta bloques.
        let recording_c = Arc::clone(&self.recording);
        let audio_rx = self.audio_rx.clone();
        let chunk_tx = self.chunk_tx.clone();
        let carryover_rx = self.carryover_rx.clone();
        let event_tx_c = self.event_tx.clone();
        let audio_consumer = thread::Builder::new()
            .name("oido-audio-chunked".into())
            .spawn(move || {
                let mut resampler = resampler;
                // Carryover del bloque anterior, devuelto por el worker.
                // Se prepone al inicio del próximo chunk. Vacío = todavía
                // no hay carryover disponible (primer chunk o el worker
                // aún no terminó el bloque anterior).
                let mut pending_carryover: Vec<f32> = Vec::new();
                let mut is_first_chunk = true;

                while let Ok(frame) = audio_rx.recv() {
                    // Paso 1: acumular el frame en el buffer si estamos
                    // grabando. Lock breve.
                    let buf_len = {
                        let mut s = recording_c.lock();
                        if !s.recording {
                            continue;
                        }
                        let samples = if let Some(r) = resampler.as_mut() {
                            match r.process(&frame.samples) {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::error!(?e, "resampler falló; descartando frame");
                                    continue;
                                }
                            }
                        } else {
                            frame.samples
                        };
                        s.samples.extend(samples);

                        // Cap anti-OOM.
                        if s.samples.len() > max_buffer_samples {
                            let excess = s.samples.len() - max_buffer_samples;
                            s.samples.drain(0..excess);
                            tracing::warn!(
                                excess,
                                "buffer excede {MAX_BUFFER_SECS}s; descartando audio viejo"
                            );
                        }
                        s.samples.len()
                    };

                    // Paso 2: ¿hay suficiente para cortar un chunk?
                    if buf_len < chunk_size {
                        continue;
                    }

                    // Paso 3: obtener el carryover del bloque anterior.
                    // Si es el primer chunk, no hay carryover.
                    //
                    // Para chunks subsiguientes, esperamos al carryover
                    // con un timeout de 30 segundos. Por qué 30s:
                    //  - En CPU con modelo small, un chunk de 5s de audio
                    //    tarda típicamente 5-8s en transcribirse (más
                    //    que el audio mismo por overhead del decoder).
                    //  - 30s es absurdo pero descarta el timeout como
                    //    causa de duplicación; cubre CPU muy lenta.
                    //  - El worker usa `send` bloqueante, así que SI hay
                    //    carryover, llega. Si el timeout expira, significa
                    //    que el worker descartó el carryover (todo
                    //    silencio) y NO envió nada — cortar sin
                    //    carryover es correcto.
                    //
                    // Si el timeout expira frecuentemente en producción,
                    // considerar reducir chunk_duration_secs (chunks más
                    // cortos = worker más rápido) o usar GPU.
                    if !is_first_chunk && pending_carryover.is_empty() {
                        match carryover_rx.recv_timeout(Duration::from_secs(30)) {
                            Ok(co) => pending_carryover = co,
                            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                                tracing::warn!(
                                    "carryover no entregado tras 30s; cortando sin él \
                                     (worker más lento de lo esperado)"
                                );
                            }
                            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                                break; // shutdown
                            }
                        }
                    }

                    // Paso 4: cortar el chunk. Prepone carryover + drena
                    // `chunk_size` muestras del inicio del buffer. El
                    // sobrante queda para el próximo chunk. Lock breve.
                    let chunk = {
                        let mut s = recording_c.lock();
                        if s.samples.len() < chunk_size {
                            // Edge case: on_release pudo haber drenado el
                            // buffer entre el paso 2 y aquí.
                            continue;
                        }
                        let carryover_len = pending_carryover.len();
                        let mut chunk = std::mem::take(&mut pending_carryover);
                        chunk.extend_from_slice(&s.samples[..chunk_size]);
                        s.samples.drain(0..chunk_size);
                        tracing::debug!(
                            carryover_len,
                            chunk_total = chunk.len(),
                            "corte de chunk"
                        );
                        chunk
                    };

                    is_first_chunk = false;

                    let _ = event_tx_c.send(PipelineEvent::State(PipelineState::Processing));
                    // `send` bloqueante: si la cola de chunks está llena
                    // (worker rezagado), esperamos. Mejor que perder
                    // audio. El buffer seguirá acumulándose mientras.
                    if chunk_tx.send(chunk).is_err() {
                        break; // shutdown
                    }
                }
            })?;
        self.audio_consumer = Some(audio_consumer);

        // 3) Worker STT: transcribe bloques y devuelve carryover.
        let chunk_rx = self.chunk_rx.clone();
        let carryover_tx = self.carryover_tx.clone();
        let event_tx_w = self.event_tx.clone();
        let transcriber = Arc::clone(&self.cfg.transcriber);
        let injector = Arc::clone(&self.cfg.injector);
        let chunk_size_w = self.chunk_size;
        let worker = thread::Builder::new()
            .name("oido-stt-chunked".into())
            .spawn(move || {
                while let Ok(chunk) = chunk_rx.recv() {
                    process_chunk(
                        chunk,
                        chunk_size_w,
                        &transcriber,
                        &injector,
                        &carryover_tx,
                        &event_tx_w,
                    );
                }
            })?;
        self.worker = Some(worker);

        // 4) Callbacks de hotkey (máquina de estados anti-reentrada
        //    igual que Pipeline).
        let event_tx_p = self.event_tx.clone();
        let recording_p = Arc::clone(&self.recording);
        let on_press = Box::new(move || {
            let mut s = recording_p.lock();
            if !s.recording {
                s.recording = true;
                s.samples.clear();
                drop(s);
                let _ = event_tx_p.send(PipelineEvent::State(PipelineState::Recording));
            }
        });

        let event_tx_r = self.event_tx.clone();
        let recording_r = Arc::clone(&self.recording);
        let chunk_tx_r = self.chunk_tx.clone();
        let on_release = Box::new(move || {
            // Guard anti-reentrada: release en Idle = rebote.
            let remainder = {
                let mut s = recording_r.lock();
                if !s.recording {
                    return;
                }
                s.recording = false;
                std::mem::take(&mut s.samples)
            };
            if remainder.is_empty() {
                let _ = event_tx_r.send(PipelineEvent::State(PipelineState::Idle));
                return;
            }
            let _ = event_tx_r.send(PipelineEvent::State(PipelineState::Processing));
            // El resto (lo que no llegó a llenar un chunk) se envía como
            // último chunk. El worker lo transcribe completo (max_samples
            // = chunk.len(), sin recorte) y el carryover final se ignora
            // (no hay próximo chunk).
            if let Err(e) = chunk_tx_r.send(remainder) {
                tracing::warn!(?e, "no se pudo encolar el resto final");
                let _ = event_tx_r.send(PipelineEvent::State(PipelineState::Idle));
            }
        });

        self.cfg
            .hotkey
            .register(&self.cfg.hotkey_binding, on_press, on_release)
            .map_err(|e| anyhow::anyhow!("hotkey.register: {e}"))?;
        Ok(())
    }

    /// Detiene el pipeline: para captura, desregistra hotkey, cierra
    /// canales. Los threads terminan solos al cerrarse los receivers.
    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.cfg.capture.stop();
        let _ = self.cfg.hotkey.unregister();
        let _ = self.event_tx.send(PipelineEvent::Shutdown);
        Ok(())
    }
}

impl Drop for ChunkedPipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Lógica del worker: transcribe un bloque, filtra, inyecta, devuelve
/// carryover. `chunk_size` es el límite de corte para
/// `transcribe_timed`; si el chunk es más corto (resto final), se pasa
/// `chunk.len()` para que todo el texto quepa.
fn process_chunk(
    chunk: ChunkJob,
    chunk_size: usize,
    transcriber: &Arc<dyn Transcriber>,
    injector: &Arc<dyn Injector>,
    carryover_tx: &Sender<Vec<f32>>,
    event_tx: &Sender<PipelineEvent>,
) {
    let samples = chunk.len();
    let audio_seconds = samples as f64 / 16_000.0;
    // max_samples: si el chunk es más corto que chunk_size (resto
    // final), no queremos recortar nada.
    let max_samples = chunk_size.min(samples);

    // STT con timestamps.
    let started = Instant::now();
    let timings: WordTimings = {
        let _span = tracing::info_span!("chunked.infer").entered();
        match transcriber.transcribe_timed(&chunk, max_samples) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(?e, samples, audio_seconds, "STT falló en chunked");
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                return;
            }
        }
    };
    let stt_latency = started.elapsed();

    let text = timings.text.trim();
    if text.is_empty() {
        // El chunk completo es carryover (ninguna palabra completa
        // cayó en el rango). Devolver todo como carryover, recortando
        // silencio final.
        tracing::debug!(
            samples,
            "chunk produjo texto vacío; devolviendo todo como carryover"
        );
        let carryover = trim_trailing_silence(&chunk);
        if !carryover.is_empty() {
            // `send` bloqueante: si el canal está lleno, el consumer
            // aún no leyó el carryover anterior. Esperamos microsegundos
            // (bounded 1) — mejor que descartar y re-procesar audio.
            if let Err(e) = carryover_tx.send(carryover.to_vec()) {
                tracing::warn!(?e, "no se pudo enviar carryover (shutdown)");
            }
        }
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        return;
    }

    // Anti-alucinación.
    if phrase_filter::filter(text).is_none() {
        tracing::info!(
            ?text,
            stt_latency_ms = stt_latency.as_millis() as u64,
            audio_seconds,
            "chunk descartado por filtro"
        );
        // Aún así devolvemos el carryover para no perder la palabra
        // cortada, recortando silencio.
        let carryover = trim_trailing_silence(&chunk[timings.last_word_end_sample..]);
        if !carryover.is_empty() {
            if let Err(e) = carryover_tx.send(carryover.to_vec()) {
                tracing::warn!(?e, "no se pudo enviar carryover (shutdown)");
            }
        }
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        return;
    }

    // Inyección (Ctrl+V incremental).
    let inject_started = Instant::now();
    if let Err(e) = injector.inject(text) {
        tracing::error!(?e, "injector falló en chunked");
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        return;
    }
    let inject_latency = inject_started.elapsed();

    tracing::info!(
        text = %text,
        samples,
        cut_sample = timings.last_word_end_sample,
        carryover_samples = chunk.len() - timings.last_word_end_sample,
        stt_ms = stt_latency.as_millis() as u64,
        inject_ms = inject_latency.as_millis() as u64,
        "chunk transcrito e inyectado"
    );

    // Carryover: audio desde el corte hasta el fin del chunk. Recortamos
    // el silencio final para no arrastrar ms inútiles al próximo bloque
    // (el caso típico es "habla + pausa antes de los 5s"). Si todo es
    // silencio, descartamos el carryover entero.
    //
    // Usamos `send` bloqueante (no `try_send`) para garantizar que el
    // consumer recibe el carryover. El canal es `bounded(1)`: si está
    // lleno, el worker espera microsegundos hasta que el consumer lea.
    // Esto evita la pérdida silenciosa de carryover que causaba
    // duplicación de palabras entre chunks.
    let raw_carryover = &chunk[timings.last_word_end_sample..];
    let carryover = trim_trailing_silence(raw_carryover);
    if !carryover.is_empty() {
        if let Err(e) = carryover_tx.send(carryover.to_vec()) {
            tracing::warn!(?e, "no se pudo enviar carryover (shutdown)");
        }
    }
    let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
}

/// Recorta el silencio al final de un slice de audio. Define silencio
/// como muestras con `|x| < 0.01` (amplitud ≈ -40 dBFS). Recorre desde
/// el final hacia atrás y devuelve el slice sin la cola silenciosa.
///
/// Si todo el audio es silencio, devuelve un slice vacío (el llamador
/// lo descarta). Conserva al menos una ventana de 200ms (3.200
/// muestras @ 16 kHz) para que el VAD del próximo bloque tenga
/// "contexto" antes del habla — sin eso, el primer fonema puede
/// perderse.
fn trim_trailing_silence(audio: &[f32]) -> &[f32] {
    const SILENCE_THRESHOLD: f32 = 0.01;
    const MIN_KEEP_SAMPLES: usize = 3_200; // 200ms

    if audio.len() <= MIN_KEEP_SAMPLES {
        return audio;
    }
    let mut end = audio.len();
    while end > MIN_KEEP_SAMPLES
        && audio[end - 1].abs() < SILENCE_THRESHOLD
        && audio[end - 2].abs() < SILENCE_THRESHOLD
        && audio[end - 4].abs() < SILENCE_THRESHOLD
    {
        end -= 1;
    }
    &audio[..end]
}
