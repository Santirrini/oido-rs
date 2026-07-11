//! Pipeline para Streaming STT usando el algoritmo LocalAgreement-2.
//!
//! Regla R1: Todo paso de mensajes principal es vía canales.
//! Regla R3: Mutex acotado (sólo BufferState compartido), el estado del streamer (LA-2)
//! es pertenencia exclusiva del thread worker STT.

use crate::pipeline::{PipelineEvent, PipelineState};
use crossbeam_channel::{Receiver, Sender};
use oido_platform::{AudioRx, AudioTx, CaptureSource, Hotkey, Injector, Resampler};
use oido_stt::Streamer;
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Configuración para arrancar el pipeline de streaming.
#[derive(Debug)]
pub struct StreamingPipelineConfig {
    pub capture: Box<dyn CaptureSource>,
    pub streamer: Box<dyn Streamer>,
    pub injector: Arc<dyn Injector>,
    pub hotkey: Box<dyn Hotkey>,
    pub hotkey_binding: String,
}

#[derive(Default, Debug)]
struct BufferState {
    samples: Vec<f32>,
    recording: bool,
}

/// Pipeline que maneja la captura continua y la inyección incremental por teclado.
#[derive(Debug)]
pub struct StreamingPipeline {
    cfg_capture: Box<dyn CaptureSource>,
    cfg_injector: Arc<dyn Injector>,
    cfg_hotkey: Box<dyn Hotkey>,
    cfg_hotkey_binding: String,

    recording: Arc<Mutex<BufferState>>,
    audio_tx: AudioTx,
    audio_rx: AudioRx,

    start_tx: Sender<()>,
    start_rx: Receiver<()>,
    release_tx: Sender<()>,
    release_rx: Receiver<()>,

    event_tx: Sender<PipelineEvent>,
    event_rx: Receiver<PipelineEvent>,

    audio_consumer: Option<JoinHandle<()>>,
    worker: Option<JoinHandle<()>>,
    streamer: Option<Box<dyn Streamer>>,
}

impl StreamingPipeline {
    /// Crea una instancia del pipeline de streaming sin arrancarlo.
    pub fn new(cfg: StreamingPipelineConfig) -> Self {
        let (audio_tx, audio_rx) = crossbeam_channel::bounded(1024);
        let (start_tx, start_rx) = crossbeam_channel::bounded(1);
        let (release_tx, release_rx) = crossbeam_channel::bounded(1);
        let (event_tx, event_rx) = crossbeam_channel::unbounded();

        Self {
            cfg_capture: cfg.capture,
            cfg_injector: cfg.injector,
            cfg_hotkey: cfg.hotkey,
            cfg_hotkey_binding: cfg.hotkey_binding,

            recording: Arc::new(Mutex::new(BufferState::default())),
            audio_tx,
            audio_rx,
            start_tx,
            start_rx,
            release_tx,
            release_rx,
            event_tx,
            event_rx,
            audio_consumer: None,
            worker: None,
            streamer: Some(cfg.streamer),
        }
    }

    /// Obtiene el receptor de eventos del pipeline para propagar estados.
    pub fn events(&self) -> Receiver<PipelineEvent> {
        self.event_rx.clone()
    }

    /// Arranca la captura y el worker que procesa el dictado incremental.
    pub fn start(&mut self) -> anyhow::Result<()> {
        self.cfg_capture
            .open(self.audio_tx.clone())
            .map_err(|e| anyhow::anyhow!("capture.open: {e}"))?;
        self.cfg_capture
            .start()
            .map_err(|e| anyhow::anyhow!("capture.start: {e}"))?;

        let input_rate = self.cfg_capture.sample_rate_hz();
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

        // 1) Audio consumer thread: mueve muestras del hardware al BufferState
        let recording_c = Arc::clone(&self.recording);
        let audio_rx = self.audio_rx.clone();
        let audio_consumer = thread::Builder::new()
            .name("oido-audio-stream".into())
            .spawn(move || {
                let mut resampler = resampler;
                while let Ok(frame) = audio_rx.recv() {
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
                }
            })?;
        self.audio_consumer = Some(audio_consumer);

        // 2) Worker thread: Ejecuta inferencia periódica usando LocalAgreement-2.
        // Mueve la propiedad exclusiva del streamer al thread para no usar lock.
        let mut streamer = self
            .streamer
            .take()
            .expect("streamer already moved or not loaded");
        let start_rx = self.start_rx.clone();
        let release_rx = self.release_rx.clone();
        let recording_w = Arc::clone(&self.recording);
        let injector = Arc::clone(&self.cfg_injector);
        let event_tx = self.event_tx.clone();

        let worker = thread::Builder::new()
            .name("oido-stt-stream".into())
            .spawn(move || {
                while let Ok(()) = start_rx.recv() {
                    let ticker = crossbeam_channel::tick(std::time::Duration::from_millis(1000));
                    loop {
                        crossbeam_channel::select! {
                            recv(release_rx) -> _ => {
                                // Final release: vacía el buffer de audio acumulado.
                                // Nota (Bug 3): NO llamamos a process() aquí — flush_final()
                                // ya confirma todo lo pendiente en prev_tokens, así que una
                                // pasada de inferencia extra sería redundante y duplicaría
                                // la transcripción.
                                {
                                    let mut s = recording_w.lock();
                                    s.samples.clear();
                                }
                                match streamer.flush_final() {
                                    Ok(transcript) => {
                                        if !transcript.confirmed.is_empty() {
                                            if let Err(e) = injector.type_text(&transcript.confirmed) {
                                                tracing::error!(?e, "injector.type_text falló");
                                                let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(?e, "flush_final falló");
                                        let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
                                    }
                                }
                                let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
                                break;
                            }
                            recv(ticker) -> _ => {
                                let audio = {
                                    let s = recording_w.lock();
                                    let max_samples = 25 * 16_000; // 25s cap
                                    if s.samples.len() > max_samples {
                                        s.samples[s.samples.len() - max_samples..].to_vec()
                                    } else {
                                        s.samples.clone()
                                    }
                                };
                                match streamer.process(&audio) {
                                    Ok(transcript) => {
                                        if !transcript.confirmed.is_empty() {
                                            if let Err(e) = injector.type_text(&transcript.confirmed) {
                                                tracing::error!(?e, "injector.type_text falló");
                                                let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(?e, "streamer.process falló");
                                        let _ = event_tx.send(PipelineEvent::State(PipelineState::Error));
                                    }
                                }
                            }
                        }
                    }
                }
            })?;
        self.worker = Some(worker);

        // 3) Callbacks de hotkey con resguardo anti-reentrada
        let recording_press = Arc::clone(&self.recording);
        let start_tx = self.start_tx.clone();
        let event_tx_press = self.event_tx.clone();
        let on_press = Box::new(move || {
            let mut s = recording_press.lock();
            if !s.recording {
                s.recording = true;
                s.samples.clear();
                drop(s);
                let _ = start_tx.send(());
                let _ = event_tx_press.send(PipelineEvent::State(PipelineState::Recording));
            }
        });

        let recording_release = Arc::clone(&self.recording);
        let release_tx = self.release_tx.clone();
        let event_tx_release = self.event_tx.clone();
        let on_release = Box::new(move || {
            let mut s = recording_release.lock();
            if !s.recording {
                return;
            }
            s.recording = false;
            drop(s);
            let _ = release_tx.send(());
            let _ = event_tx_release.send(PipelineEvent::State(PipelineState::Processing));
        });

        self.cfg_hotkey
            .register(&self.cfg_hotkey_binding, on_press, on_release)
            .map_err(|e| anyhow::anyhow!("hotkey.register: {e}"))?;

        Ok(())
    }

    /// Detiene y desconecta los canales de ejecución.
    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.cfg_capture.stop();
        let _ = self.cfg_hotkey.unregister();
        let _ = self.event_tx.send(PipelineEvent::Shutdown);
        Ok(())
    }
}

impl Drop for StreamingPipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
