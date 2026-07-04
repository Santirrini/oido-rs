//! Pipeline MVP `dicta-y-pega`. Hold F8 → hablas → release → texto en
//! cursor.
//!
//!
//! Regla R1 (canales-only): el búfer de muestras vive en un
//! `Arc<parking_lot::Mutex<_>>` compartido por (a) el consumer que
//! appende frames mientras graba y (b) los callbacks de hotkey que
//! snapshot+borran al release. La violación está acotada a UN campo
//! (`samples`/`recording`) y es inevitable dado que cpal requiere que
//! el closure de capture viva sobre el `Stream` desde el `open()`.
//!
//! Regla R3: el único mutex del workspace *fuera* de `oido-config` es
//! este. Justificación: ningún estado del programa cabe en un canal
//! puro sin serializar el hold-mode. Si esto se propaga a más crates,
//! refactorizar a un bus-suscripción por canal (Fase 8 si surge).
//!
//! Regla R2: trivial, no hay `unsafe` aquí; el FFI vive sólo en
//! `oido-stt/src/whisper_cpp.rs`.

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;

use oido_platform::{
    AudioFrame, AudioRx, AudioTx,
    CaptureSource, Hotkey, Injector,
};
use oido_stt::Transcriber;

use crate::dedup::Dedup;
use crate::phrase_filter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    Idle,
    Recording,
    Processing,
}

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    State(PipelineState),
}

pub struct PipelineConfig {
    pub capture: Box<dyn CaptureSource>,
    pub transcriber: Arc<dyn Transcriber>,
    pub injector: Arc<dyn Injector>,
    pub hotkey: Box<dyn Hotkey>,
}

#[derive(Default)]
struct BufferState {
    samples: Vec<f32>,
    recording: bool,
    dedup: Dedup,
}

pub struct Pipeline {
    cfg: PipelineConfig,
    recording: Arc<Mutex<BufferState>>,
    audio_tx: AudioTx,
    audio_rx: AudioRx,
    event_tx: Sender<PipelineEvent>,
    event_rx: Receiver<PipelineEvent>,
    consumer: Option<JoinHandle<()>>,
}

impl Pipeline {
    /// Crear y construir el objeto Pipeline (no arranca todavía).
    pub fn new(cfg: PipelineConfig) -> Self {
        let (audio_tx, audio_rx) = crossbeam_channel::bounded(1024);
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        Self {
            cfg,
            recording: Arc::new(Mutex::new(BufferState::default())),
            audio_tx,
            audio_rx,
            event_tx,
            event_rx,
            consumer: None,
        }
    }

    /// Receiver de eventos para los observadores (bin → tray).
    #[must_use]
    pub fn events(&self) -> Receiver<PipelineEvent> {
        self.event_rx.clone()
    }

    /// Abre captura + arranca el consumer thread + registra el hotkey.
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

        // 2) consumer thread: drena audio_rx → buffer si está grabando.
        let recording = Arc::clone(&self.recording);
        let audio_rx = self.audio_rx.clone();
        let handle = thread::Builder::new()
            .name("oido-audio".into())
            .spawn(move || {
                while let Ok(frame) = audio_rx.recv() {
                    let mut s = recording.lock();
                    if s.recording {
                        s.samples.extend(frame.samples);
                    }
                }
            })?;
        self.consumer = Some(handle);

        // 3) hotkey: cierra el ciclo press → record, release → process.
        let event_tx = self.event_tx.clone();
        let recording_p = Arc::clone(&self.recording);
        let on_press = move || {
            let mut s = recording_p.lock();
            if !s.recording {
                s.recording = true;
                s.samples.clear();
                s.dedup = Dedup::new();
                drop(s);
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Recording));
            }
        };

        let event_tx = self.event_tx.clone();
        let recording_r = Arc::clone(&self.recording);
        let transcriber = Arc::clone(&self.cfg.transcriber);
        let injector = Arc::clone(&self.cfg.injector);
        let on_release = move || {
            // 3a) snapshot del buffer + cambio de estado.
            let buffer = {
                let mut s = recording_r.lock();
                s.recording = false;
                std::mem::take(&mut s.samples)
            };
            if buffer.is_empty() {
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                return;
            }
            let _ = event_tx.send(PipelineEvent::State(PipelineState::Processing));

            // 3b) STT. Bloquea (whisper.cpp es CPU-bound).
            let text = match transcriber.transcribe(&buffer) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(?e, samples = buffer.len(), "STT falló");
                    let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                    return;
                }
            };

            // 3c) Anti-alucinación (exact match contra blacklist ES+EN).
            if phrase_filter::filter(&text).is_none() {
                tracing::info!(?text, "frase descartada por filtro");
                let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                return;
            }

            // 3d) Inyección. Bloquea (clipboard + paste).
            if let Err(e) = injector.inject(&text) {
                tracing::error!(?e, "injector falló");
            }
            let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        };

        self.cfg
            .hotkey
            .register(on_press, on_release)
            .map_err(|e| anyhow::anyhow!("hotkey.register: {e}"))?;
        Ok(())
    }

    /// Para el pipeline: para la captura, desregistra el hotkey.
    /// El consumer thread abandona al final del programa, no lo
    /// bloqueamos aquí (el recv termina cuando cpal libera su clon
    /// del Sender al destruir su `Stream`).
    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.cfg.capture.stop();
        let _ = self.cfg.hotkey.unregister();
        Ok(())
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
