//! Pipeline MVP `dicta-y-pega`. Hold F8 → hablas → release → texto en
//! cursor.
//!
//!
//! Regla R1 (canales-only): ahora **estricta**. El búfer de muestras
//! sigue en un `Arc<parking_lot::Mutex<_>>` compartido por (a) el
//! consumer que appende frames mientras graba y (b) los callbacks de
//! hotkey que snapshot+borran al release. Es el único mutex del
//! workspace fuera de `oido-config`, y está acotado a UN campo
//! (`samples`/`recording`) porque cpal requiere que el closure de
//! capture viva sobre el `Stream` desde el `open()`.
//!
//! La transcripción ya NO se ejecuta inline en el callback
//! `on_release`: se encola el buffer por canal a un thread worker
//! `"oido-stt"` dedicado, que libera el thread del hotkey en
//! microsegundos. Esto alinea el pipeline con la regla R1.
//!
//! ## Máquina de estados anti-reentrada
//!
//! Los callbacks `on_press` / `on_release` son idempotentes frente a
//! rebotes del sistema operativo (autorepeat, KeyRelease fantasma de
//! Windows al rotar foco, doble-click en raw input):
//!
//! - `Idle + press` → `Recording`
//! - `Recording + press` → no-op (ya estamos grabando)
//! - `Recording + release` → `Processing` (snapshot + enqueue STT)
//! - `Idle + release` → **no-op** (evita `AudioTooShort` espurios y la
//!   fragmentación que producía alucinaciones — antes, un release
//!   en Idle siempre generaba un job vacío y un cambio de estado).
//!
//! Además la tecla target está **suprimida a nivel OS** por
//! `rdev::grab` en `oido-platform::hotkey`, así que la app con foco
//! nunca la ve.
//!
//! Regla R3: el único mutex del workspace *fuera* de `oido-config` es
//! este.
//!
//! Regla R2: trivial, no hay `unsafe` aquí; el FFI vive sólo en
//! `oido-stt/src/whisper_cpp.rs`.

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;

use oido_platform::{AudioRx, AudioTx, CaptureSource, Hotkey, Injector, Resampler};
use oido_stt::Transcriber;

use crate::dedup::Dedup;
use crate::phrase_filter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    Idle,
    Recording,
    Processing,
    /// Error transitorio de STT o inyección. Vuelve a Idle tras emitirlo
    /// para que UI/tray puedan mostrar feedback diferenciado.
    Error,
}

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    State(PipelineState),
    Shutdown,
}

#[derive(Debug)]
pub struct PipelineConfig {
    pub capture: Box<dyn CaptureSource>,
    pub transcriber: Arc<dyn Transcriber>,
    pub injector: Arc<dyn Injector>,
    pub hotkey: Box<dyn Hotkey>,
    /// Binding canónico para el hotkey (ej. "F8", "Ctrl+Shift+D"). Se
    /// entrega a `Hotkey::register` al arrancar el pipeline.
    pub hotkey_binding: String,
}

#[derive(Default, Debug)]
struct BufferState {
    samples: Vec<f32>,
    recording: bool,
    dedup: Dedup,
}

/// Tipo del canal STT: vector de muestras listas para transcribir.
type SttJob = Vec<f32>;

/// Capacidad del canal STT. Hold-to-talk: 6 para dar margen extra sin
/// riesgo de OOM y soportar ráfagas de 3+ activaciones consecutivas
/// mientras los 2 workers procesan los buffers previos.
const STT_QUEUE_CAP: usize = 6;

#[derive(Debug)]
pub struct Pipeline {
    cfg: PipelineConfig,
    recording: Arc<Mutex<BufferState>>,
    audio_tx: AudioTx,
    audio_rx: AudioRx,
    /// Canal del hotkey → STT worker. Bounded 1.
    stt_tx: Sender<SttJob>,
    stt_rx: Receiver<SttJob>,
    event_tx: Sender<PipelineEvent>,
    event_rx: Receiver<PipelineEvent>,
    audio_consumer: Option<JoinHandle<()>>,
    stt_workers: Vec<JoinHandle<()>>,
}

impl Pipeline {
    /// Crear y construir el objeto Pipeline (no arranca todavía).
    pub fn new(cfg: PipelineConfig) -> Self {
        let (audio_tx, audio_rx) = crossbeam_channel::bounded(1024);
        let (stt_tx, stt_rx) = crossbeam_channel::bounded(STT_QUEUE_CAP);
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        Self {
            cfg,
            recording: Arc::new(Mutex::new(BufferState::default())),
            audio_tx,
            audio_rx,
            stt_tx,
            stt_rx,
            event_tx,
            event_rx,
            audio_consumer: None,
            stt_workers: Vec::new(),
        }
    }

    /// Receiver de eventos para los observadores (bin → tray).
    #[must_use]
    pub fn events(&self) -> Receiver<PipelineEvent> {
        self.event_rx.clone()
    }

    /// Abre captura + arranca el consumer thread + el worker STT +
    /// registra el hotkey.
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

        // Resampler: si la captura no está a 16 kHz, creamos uno. Vive
        // en el closure del consumer (no en el closure de cpal) porque
        // rubato::SincFixedIn mantiene estado entre llamadas y no es
        // Clone.
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
                    tracing::warn!(
                        input_rate,
                        "no pude crear resampler; audio podría no ser 16kHz"
                    );
                    None
                }
            }
        };

        // 2) consumer thread: drena audio_rx → buffer si está grabando.
        // Si hay resampler, aplica resampling por bloque antes de
        // appendear al buffer (el buffer siempre termina a 16 kHz).
        let recording = Arc::clone(&self.recording);
        let audio_rx = self.audio_rx.clone();
        let audio_consumer = thread::Builder::new()
            .name("oido-audio".into())
            .spawn(move || {
                let mut resampler = resampler;
                while let Ok(frame) = audio_rx.recv() {
                    let mut s = recording.lock();
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
                }
            })?;
        self.audio_consumer = Some(audio_consumer);

        // 3) STT workers: drena stt_rx → transcribe → filter → inject.
        //    Posee su propio clon de event_tx, transcriber, injector.
        const STT_WORKERS: usize = 2;
        for i in 0..STT_WORKERS {
            let stt_rx = self.stt_rx.clone();
            let event_tx = self.event_tx.clone();
            let transcriber = Arc::clone(&self.cfg.transcriber);
            let injector = Arc::clone(&self.cfg.injector);
            let worker = thread::Builder::new()
                .name(format!("oido-stt-{i}"))
                .spawn(move || {
                    while let Ok(buffer) = stt_rx.recv() {
                        process_one(buffer, &transcriber, &injector, &event_tx);
                    }
                })?;
            self.stt_workers.push(worker);
        }

        // 4) hotkey: cierra el ciclo press → record, release → enqueue.
        let event_tx = self.event_tx.clone();
        let recording_p = Arc::clone(&self.recording);
        let on_press = Box::new(move || {
            let mut s = recording_p.lock();
            if !s.recording {
                s.recording = true;
                s.samples.clear();
                s.dedup = Dedup::new();
                drop(s);
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Recording));
            }
        });

        let event_tx = self.event_tx.clone();
        let recording_r = Arc::clone(&self.recording);
        let stt_tx = self.stt_tx.clone();
        let on_release = Box::new(move || {
            // 4a) snapshot del buffer + cambio de estado.
            //
            // Guarda anti-reentrada: si NO estamos grabando, el
            // release es un rebote (autorepeat / KeyRelease fantasma
            // de Windows / cambio de foco). Lo ignoramos por completo
            // y NO emitimos estado: un release en `Idle` antes
            // provocaba un job vacío en STT que devolvía
            // `AudioTooShort` y fragmentaba la grabación.
            let buffer = {
                let mut s = recording_r.lock();
                if !s.recording {
                    return;
                }
                s.recording = false;
                std::mem::take(&mut s.samples)
            };
            if buffer.is_empty() {
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                return;
            }
            let _ = event_tx.send(PipelineEvent::State(PipelineState::Processing));

            // 4b) Encolar al worker STT. `try_send` con canal bounded
            // 1: si el worker está ocupado con el job anterior, el
            // nuevo buffer lo reemplaza (no acumulamos). Esto evita
            // bloquear el callback del hotkey.
            match stt_tx.try_send(buffer) {
                Ok(()) => {}
                Err(_) => {
                    // Cola llena Y sin reemplazo posible: descartamos el
                    // nuevo y volvemos a Idle.
                    tracing::warn!("cola STT llena; descartando buffer");
                    let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                }
            }
        });

        self.cfg
            .hotkey
            .register(&self.cfg.hotkey_binding, on_press, on_release)
            .map_err(|e| anyhow::anyhow!("hotkey.register: {e}"))?;
        Ok(())
    }

    /// Para el pipeline: para la captura, desregistra el hotkey,
    /// dropea los canales para que los workers terminen.
    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.cfg.capture.stop();
        let _ = self.cfg.hotkey.unregister();
        let _ = self.event_tx.send(PipelineEvent::Shutdown);
        // Dropear los Senders cierra los canales; los workers salen
        // del `recv()` cuando el canal se cierra.
        // No hacemos join: cada worker termina solo, no bloqueamos
        // shutdown.
        Ok(())
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct LatencyReport {
    audio_secs: f64,
    stt_ms: u64,
    inject_ms: u64,
    total_ms: u64,
}

/// Lógica del worker STT: procesa un buffer, emite eventos, inyecta.
fn process_one(
    buffer: SttJob,
    transcriber: &Arc<dyn Transcriber>,
    injector: &Arc<dyn Injector>,
    event_tx: &Sender<PipelineEvent>,
) {
    let samples = buffer.len();
    let audio_seconds = samples as f64 / 16_000.0;

    // STT. Bloquea (whisper.cpp es CPU/GPU-bound).
    let started = Instant::now();
    let text = {
        let _span = tracing::info_span!("stt.infer").entered();
        match transcriber.transcribe(&buffer) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(?e, samples, audio_seconds, "STT falló");
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                return;
            }
        }
    };
    let stt_latency = started.elapsed();

    // Anti-alucinación (exact match contra blacklist ES+EN).
    if phrase_filter::filter(&text).is_none() {
        tracing::info!(
            ?text,
            stt_latency_ms = stt_latency.as_millis() as u64,
            audio_seconds,
            "frase descartada por filtro"
        );
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        return;
    }

    // Inyección. Bloquea (clipboard + paste).
    let inject_started = Instant::now();
    if let Err(e) = injector.inject(&text) {
        tracing::error!(?e, "injector falló");
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        return;
    }
    let inject_latency = inject_started.elapsed();

    let report = LatencyReport {
        audio_secs: audio_seconds,
        stt_ms: stt_latency.as_millis() as u64,
        inject_ms: inject_latency.as_millis() as u64,
        total_ms: stt_latency.as_millis() as u64 + inject_latency.as_millis() as u64,
    };
    tracing::info!(
        text = %text,
        samples,
        latency_report = ?report,
        "dictado inyectado"
    );
    let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
}
