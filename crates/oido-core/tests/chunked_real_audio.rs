//! Test E2E con audio real para diagnosticar el modo "Chunked".
//!
//! Carga un audio PCM 16kHz mono float32 desde `audio_16k.bin` (~1.1 MB,
//! ~18s), lo empuja por el ChunkedPipeline con un WhisperCpp real
//! (modelo ggml-small.bin) y captura cada inyección con timestamp
//! relativo. Misma transcripción contra el método `transcribe` batch
//! del mismo backend para comparar.
//!
//! Para regenerar `audio_16k.bin` desde `Grabación_prueba.mp3`:
//!
//! ```bash
//! python -c "
//! import soundfile as sf, numpy as np
//! from scipy.signal import resample_poly
//! data, sr = sf.read('Grabación_prueba.mp3')
//! data = resample_poly(data, up=1, down=3).astype('<f4')
//! data.tofile('audio_16k.bin')
//! "
//! ```
//!
//! Modelo requerido: `%APPDATA%/oido/models/ggml-small.bin` (Windows).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use oido_audio::{AudioError, AudioFrame, CaptureSource};
use oido_core::{ChunkedPipeline, ChunkedPipelineConfig, PipelineEvent, PipelineState};
use oido_hotkey::{Hotkey, HotkeyError};
use oido_input::{InjectError, Injector};
use oido_stt::{SharedTranscriber, Transcriber, WhisperCpp};

// ----- Mock capture que auto-reproduce un audio pre-cargado en start() -----

#[derive(Debug)]
struct FilePlaybackCapture {
    sink: parking_lot::Mutex<Option<Sender<AudioFrame>>>,
    audio: Vec<f32>,
    playing: Arc<AtomicBool>,
    play_thread: parking_lot::Mutex<Option<thread::JoinHandle<()>>>,
}

impl FilePlaybackCapture {
    fn new(audio: Vec<f32>) -> Self {
        Self {
            sink: parking_lot::Mutex::new(None),
            audio,
            playing: Arc::new(AtomicBool::new(false)),
            play_thread: parking_lot::Mutex::new(None),
        }
    }
    fn playing_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.playing)
    }
}

impl CaptureSource for FilePlaybackCapture {
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), AudioError> {
        *self.sink.lock() = Some(sink);
        Ok(())
    }
    fn start(&mut self) -> Result<(), AudioError> {
        // Lanza un thread que reproduce el audio en frames de 50ms a
        // velocidad real. El thread termina cuando se agota el audio o
        // cuando `playing` pasa a false (vía stop()).
        let sink = self.sink.lock().clone();
        let Some(tx) = sink else {
            return Err(AudioError::Capture("sink no registrado".into()));
        };
        let audio = self.audio.clone();
        let playing = Arc::clone(&self.playing);
        playing.store(true, Ordering::SeqCst);
        let handle = thread::Builder::new()
            .name("mock-playback".into())
            .spawn(move || {
                let frame_size = 16_000 / 20; // 50ms
                for chunk in audio.chunks(frame_size) {
                    if !playing.load(Ordering::SeqCst) {
                        break;
                    }
                    let frame = AudioFrame {
                        samples: chunk.to_vec(),
                        sample_rate_hz: 16_000,
                    };
                    if tx.send(frame).is_err() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                playing.store(false, Ordering::SeqCst);
            })
            .map_err(|e| AudioError::Capture(format!("spawn: {e}")))?;
        *self.play_thread.lock() = Some(handle);
        Ok(())
    }
    fn stop(&mut self) -> Result<(), AudioError> {
        self.playing.store(false, Ordering::SeqCst);
        if let Some(h) = self.play_thread.lock().take() {
            let _ = h.join();
        }
        *self.sink.lock() = None;
        Ok(())
    }
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }
}

// ----- Mock hotkey -----

struct MockHotkeyInner {
    on_press: parking_lot::Mutex<Option<Box<dyn Fn() + Send + 'static>>>,
    on_release: parking_lot::Mutex<Option<Box<dyn Fn() + Send + 'static>>>,
}

#[derive(Clone)]
struct MockHotkeyHandle {
    inner: Arc<MockHotkeyInner>,
}

impl std::fmt::Debug for MockHotkeyHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockHotkeyHandle").finish()
    }
}

struct MockHotkey {
    inner: Arc<MockHotkeyInner>,
}

impl std::fmt::Debug for MockHotkey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockHotkey").finish_non_exhaustive()
    }
}

impl MockHotkey {
    fn new() -> Self {
        Self {
            inner: Arc::new(MockHotkeyInner {
                on_press: parking_lot::Mutex::new(None),
                on_release: parking_lot::Mutex::new(None),
            }),
        }
    }
    fn handle(&self) -> MockHotkeyHandle {
        MockHotkeyHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockHotkeyHandle {
    fn press(&self) {
        if let Some(cb) = self.inner.on_press.lock().as_ref() {
            cb();
        }
    }
    fn release(&self) {
        if let Some(cb) = self.inner.on_release.lock().as_ref() {
            cb();
        }
    }
}

impl Hotkey for MockHotkey {
    fn register(
        &mut self,
        _binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), HotkeyError> {
        *self.inner.on_press.lock() = Some(on_press);
        *self.inner.on_release.lock() = Some(on_release);
        Ok(())
    }
    fn unregister(&mut self) -> Result<(), HotkeyError> {
        *self.inner.on_press.lock() = None;
        *self.inner.on_release.lock() = None;
        Ok(())
    }
}

// ----- Mock injector: registra todo con timestamp relativo -----

#[derive(Debug)]
struct RecordingInjector {
    log: parking_lot::Mutex<Vec<(Duration, String)>>,
    started_at: parking_lot::Mutex<Option<Instant>>,
}

impl RecordingInjector {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            log: parking_lot::Mutex::new(Vec::new()),
            started_at: parking_lot::Mutex::new(None),
        })
    }
    fn log(&self) -> Vec<(Duration, String)> {
        self.log.lock().clone()
    }
    fn mark_start(&self) {
        *self.log.lock() = Vec::new();
        *self.started_at.lock() = Some(Instant::now());
    }
}

impl Injector for RecordingInjector {
    fn inject(&self, text: &str) -> Result<(), InjectError> {
        let started = *self.started_at.lock();
        let elapsed = started.map(|t| t.elapsed()).unwrap_or_default();
        self.log.lock().push((elapsed, text.to_owned()));
        Ok(())
    }
}

// ----- Cargar audio -----

fn load_audio_bin(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("no se pudo leer audio_16k.bin");
    assert!(
        bytes.len() % 4 == 0,
        "audio_16k.bin debe tener tamaño múltiplo de 4"
    );
    let count = bytes.len() / 4;
    let mut audio = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        audio.push(f32::from_le_bytes(arr));
    }
    audio
}

fn load_whisper(model_path: &PathBuf) -> Arc<SharedTranscriber> {
    let mut stt = WhisperCpp::with_language("es");
    stt.load_model(model_path).expect("cargar modelo whisper");
    let _ = stt.warm_up();
    Arc::new(SharedTranscriber::new(stt))
}

// ----- Test -----

#[test]
#[ignore = "requiere audio_16k.bin y ggml-small.bin en disco"]
fn diagnose_chunked_with_real_audio() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();
    let audio_bin = workspace.join("audio_16k.bin");
    let model_path = if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
            .join("oido")
            .join("models")
            .join("ggml-small.bin")
    } else {
        PathBuf::from(std::env::var("HOME").expect("HOME"))
            .join(".local")
            .join("share")
            .join("oido")
            .join("models")
            .join("ggml-small.bin")
    };
    assert!(
        audio_bin.exists(),
        "audio_16k.bin no existe en {audio_bin:?}; regenerar desde Grabación_prueba.mp3"
    );
    assert!(
        model_path.exists(),
        "modelo whisper no existe en {model_path:?}"
    );

    println!("\n========================================");
    println!("DIAGNOSTICO: ChunkedPipeline con audio real");
    println!("========================================");
    println!("Audio: {}", audio_bin.display());
    println!("Modelo: {}", model_path.display());

    let audio = load_audio_bin(&audio_bin);
    let duration_secs = audio.len() as f64 / 16_000.0;
    println!(
        "Audio cargado: {} samples ({:.2}s @ 16kHz, peak={:.4})",
        audio.len(),
        duration_secs,
        audio.iter().fold(0.0f32, |a, b| a.max(b.abs()))
    );

    let transcriber = load_whisper(&model_path);

    // === FASE 1: baseline batch ===
    println!("\n--- FASE 1: transcripción batch directa ---");
    let batch_started = Instant::now();
    let batch_text = match transcriber.transcribe(&audio) {
        Ok(t) => t,
        Err(e) => panic!("batch transcribe falló: {e:?}"),
    };
    let batch_elapsed = batch_started.elapsed();
    println!("Batch: {:.2}s", batch_elapsed.as_secs_f64());
    println!("Texto batch: {:?}", batch_text);

    // === FASE 2: también transcribir con timestamps para ver palabras ===
    println!("\n--- FASE 1.5: batch con transcribe_timed (todo el audio) ---");
    let timed_started = Instant::now();
    let timed_result = transcriber
        .transcribe_timed(&audio, audio.len())
        .expect("transcribe_timed");
    let timed_elapsed = timed_started.elapsed();
    println!("Timed batch: {:.2}s", timed_elapsed.as_secs_f64());
    println!("Texto timed: {:?}", timed_result.text);
    println!(
        "last_word_end_sample: {} / {}",
        timed_result.last_word_end_sample,
        audio.len()
    );

    // === FASE 3: ChunkedPipeline real ===
    println!("\n--- FASE 2: ChunkedPipeline (chunk_secs=5.0) ---");
    let chunk_secs = 5.0_f32;
    let capture = FilePlaybackCapture::new(audio.clone());
    let _playing = capture.playing_handle();
    let hotkey = MockHotkey::new();
    let hotkey_h = hotkey.handle();
    let injector = RecordingInjector::new();
    injector.mark_start();
    let injector_handle = injector.clone();

    let cfg = ChunkedPipelineConfig {
        capture: Box::new(capture),
        transcriber: transcriber.clone(),
        injector: injector as Arc<dyn Injector>,
        hotkey: Box::new(hotkey),
        hotkey_binding: "F8".into(),
        chunk_duration_secs: chunk_secs,
    };
    let mut pipeline = ChunkedPipeline::new(cfg);
    pipeline.start().expect("start chunked");
    let events = pipeline.events();

    let t0 = Instant::now();
    hotkey_h.press();
    let _ = wait_for_state(&events, PipelineState::Recording, Duration::from_secs(2));
    println!("[t={:.2}s] Recording started", t0.elapsed().as_secs_f64());

    // Esperar a que el audio se reproduzca + worker procese + último
    // carryover. La duración del audio es ~18s + latencia del worker.
    // Esperamos hasta que el injector no reciba nada nuevo por 3 segundos.
    let mut last_log_len = 0;
    let mut stable_since = Instant::now();
    let mut all_log: Vec<(Duration, String)> = Vec::new();
    loop {
        thread::sleep(Duration::from_millis(500));
        let cur = injector_handle.log();
        if cur.len() != last_log_len {
            // Hay actividad nueva: capturar y resetear el reloj.
            for entry in &cur[last_log_len..] {
                println!(
                    "  [#{:02} t={:.2}s] {:?}",
                    all_log.len(),
                    entry.0.as_secs_f64(),
                    entry.1
                );
                all_log.push(entry.clone());
            }
            last_log_len = cur.len();
            stable_since = Instant::now();
        } else if t0.elapsed() > Duration::from_secs(duration_secs as u64 + 5)
            && stable_since.elapsed() > Duration::from_secs(3)
        {
            // Ya se reprodujo todo el audio y llevamos 3s sin actividad.
            break;
        }
        if t0.elapsed() > Duration::from_secs(60) {
            println!("  [TIMEOUT] cortando tras 60s");
            break;
        }
    }

    hotkey_h.release();
    let _ = wait_for_state(&events, PipelineState::Idle, Duration::from_secs(15));

    let chunked_elapsed = t0.elapsed();
    println!(
        "\n[t={:.2}s] Pipeline cerrado (total {:.2}s de wall time)",
        chunked_elapsed.as_secs_f64(),
        chunked_elapsed.as_secs_f64()
    );

    // === FASE 4: comparación ===
    println!("\n--- FASE 3: Comparación ---");
    let chunked_text: String = all_log
        .iter()
        .map(|(_, t)| t.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    println!(
        "\nTexto chunked concatenado ({} inyecciones):",
        all_log.len()
    );
    println!("  {:?}", chunked_text);

    println!("\nTexto batch (baseline):");
    println!("  {:?}", batch_text);

    let batch_n = batch_text.chars().count();
    let chunked_n = chunked_text.chars().count();
    println!(
        "\nDiferencia de longitud: {} chars (batch) vs {} chars (chunked) → ratio {:.2}",
        batch_n,
        chunked_n,
        chunked_n as f64 / batch_n.max(1) as f64
    );

    // Análisis palabra a palabra: ¿faltan o sobran?
    let batch_words: Vec<&str> = batch_text.split_whitespace().collect();
    let chunked_words: Vec<&str> = chunked_text.split_whitespace().collect();
    println!(
        "\nPalabras: {} (batch) vs {} (chunked)",
        batch_words.len(),
        chunked_words.len()
    );
    let batch_set: std::collections::HashSet<&str> = batch_words.iter().copied().collect();
    let chunked_set: std::collections::HashSet<&str> = chunked_words.iter().copied().collect();
    let only_batch: Vec<&&str> = batch_set.difference(&chunked_set).collect();
    let only_chunked: Vec<&&str> = chunked_set.difference(&batch_set).collect();
    if !only_batch.is_empty() {
        println!(
            "\nPalabras SOLO en batch (no aparecen en chunked): {:?}",
            only_batch
        );
    }
    if !only_chunked.is_empty() {
        println!(
            "Palabras SOLO en chunked (no aparecen en batch): {:?}",
            only_chunked
        );
    }
    if only_batch.is_empty() && only_chunked.is_empty() {
        println!("\n✓ Conjuntos de palabras coinciden (mismas palabras únicas).");
    }

    pipeline.shutdown().ok();
}

fn wait_for_state(
    rx: &crossbeam_channel::Receiver<PipelineEvent>,
    target: PipelineState,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(PipelineEvent::State(s)) if s == target => return true,
            Ok(_) => continue,
            Err(_) => return false,
        }
    }
    false
}
