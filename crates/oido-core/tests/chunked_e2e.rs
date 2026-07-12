//! Integration tests para el pipeline "Chunked".
//!
//! Reusa los mocks de `pipeline_e2e` (capture, hotkey, injector) y añade
//! un `MockTimedTranscriber` que implementa `transcribe_timed` con
//! comportamiento predecible: devuelve un texto fijo y un corte en
//! `max_samples` (simulando que toda palabra cabe en el rango).

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::Sender;
use oido_audio::{AudioError, AudioFrame, CaptureSource};
use oido_core::{ChunkedPipeline, ChunkedPipelineConfig, PipelineEvent, PipelineState};
use oido_hotkey::{Hotkey, HotkeyError};
use oido_input::{InjectError, Injector};
use oido_stt::{SttError, Transcriber, WordTimings};

// ----- Mock capture (idéntico al de pipeline_e2e) ---------------------------

#[derive(Debug)]
struct MockCaptureInner {
    sink: parking_lot::Mutex<Option<Sender<AudioFrame>>>,
    sample_rate: u32,
}

#[derive(Clone)]
struct MockCaptureHandle {
    inner: Arc<MockCaptureInner>,
}

impl std::fmt::Debug for MockCaptureHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockCaptureHandle")
            .field("sample_rate", &self.inner.sample_rate)
            .finish()
    }
}

#[derive(Debug)]
struct MockCapture {
    inner: Arc<MockCaptureInner>,
}

impl MockCapture {
    fn new(sample_rate: u32) -> Self {
        Self {
            inner: Arc::new(MockCaptureInner {
                sink: parking_lot::Mutex::new(None),
                sample_rate,
            }),
        }
    }

    fn handle(&self) -> MockCaptureHandle {
        MockCaptureHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockCaptureHandle {
    fn send(&self, frame: AudioFrame) {
        let guard = self.inner.sink.lock();
        if let Some(tx) = guard.as_ref() {
            let _ = tx.send(frame);
        }
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }
}

impl CaptureSource for MockCapture {
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), AudioError> {
        *self.inner.sink.lock() = Some(sink);
        Ok(())
    }
    fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
    fn stop(&mut self) -> Result<(), AudioError> {
        *self.inner.sink.lock() = None;
        Ok(())
    }
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate
    }
}

// ----- Mock hotkey (idéntico al de pipeline_e2e) ----------------------------

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
        let guard = self.inner.on_press.lock();
        if let Some(cb) = guard.as_ref() {
            cb();
        }
    }
    fn release(&self) {
        let guard = self.inner.on_release.lock();
        if let Some(cb) = guard.as_ref() {
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

// ----- Mock transcriber con transcribe_timed --------------------------------

/// Transcriber mock que cuenta cuántas veces se llama a `transcribe_timed`
/// y devuelve un texto numerado para distinguir cada llamada. El corte
/// siempre es `max_samples` (simula que toda palabra cabe en el rango,
/// sin carryover).
#[derive(Debug)]
struct MockTimedTranscriberInner {
    call_count: AtomicUsize,
    fail: AtomicBool,
}

#[derive(Clone)]
struct MockTimedTranscriberHandle {
    inner: Arc<MockTimedTranscriberInner>,
}

impl std::fmt::Debug for MockTimedTranscriberHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockTimedTranscriberHandle").finish()
    }
}

#[derive(Debug)]
struct MockTimedTranscriber {
    inner: Arc<MockTimedTranscriberInner>,
}

impl MockTimedTranscriber {
    fn new() -> Self {
        Self {
            inner: Arc::new(MockTimedTranscriberInner {
                call_count: AtomicUsize::new(0),
                fail: AtomicBool::new(false),
            }),
        }
    }

    fn handle(&self) -> MockTimedTranscriberHandle {
        MockTimedTranscriberHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockTimedTranscriberHandle {
    fn call_count(&self) -> usize {
        self.inner.call_count.load(Ordering::SeqCst)
    }
    fn set_fail(&self, fail: bool) {
        self.inner.fail.store(fail, Ordering::SeqCst);
    }
}

impl Transcriber for MockTimedTranscriber {
    fn transcribe(&self, _audio: &[f32]) -> Result<String, SttError> {
        Ok("texto".to_string())
    }

    fn transcribe_timed(&self, audio: &[f32], max_samples: usize) -> Result<WordTimings, SttError> {
        if self.inner.fail.load(Ordering::SeqCst) {
            return Err(SttError::Backend("mock failure".into()));
        }
        let n = self.inner.call_count.fetch_add(1, Ordering::SeqCst) + 1;
        // Texto numerado para distinguir cada bloque en el injector.
        let text = format!("bloque_{n}");
        // Corte a max_samples: simula que toda palabra cabe. Si el chunk
        // es más corto que max_samples (resto final), todo cabe.
        let cut = max_samples.min(audio.len());
        Ok(WordTimings {
            text,
            last_word_end_sample: cut,
        })
    }

    fn load_model(&mut self, _path: &Path) -> Result<(), SttError> {
        Ok(())
    }
}

// ----- Mock injector (idéntico al de pipeline_e2e) --------------------------

#[derive(Debug)]
struct MockInjectorInner {
    texts: parking_lot::Mutex<Vec<String>>,
}

#[derive(Clone)]
struct MockInjectorHandle {
    inner: Arc<MockInjectorInner>,
}

impl std::fmt::Debug for MockInjectorHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockInjectorHandle").finish()
    }
}

#[derive(Debug)]
struct MockInjector {
    inner: Arc<MockInjectorInner>,
}

impl MockInjector {
    fn new() -> Self {
        Self {
            inner: Arc::new(MockInjectorInner {
                texts: parking_lot::Mutex::new(Vec::new()),
            }),
        }
    }

    fn handle(&self) -> MockInjectorHandle {
        MockInjectorHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockInjectorHandle {
    fn texts(&self) -> Vec<String> {
        self.inner.texts.lock().clone()
    }
}

impl Injector for MockInjector {
    fn inject(&self, text: &str) -> Result<(), InjectError> {
        self.inner.texts.lock().push(text.to_owned());
        Ok(())
    }
}

// ----- Harness --------------------------------------------------------------

struct ChunkedRig {
    pipeline: ChunkedPipeline,
    hotkey: MockHotkeyHandle,
    capture: MockCaptureHandle,
    injector: MockInjectorHandle,
    transcriber: MockTimedTranscriberHandle,
    events: crossbeam_channel::Receiver<PipelineEvent>,
}

/// Crea un rig con `chunk_secs=1.0` (16.000 muestras) para que los tests
/// sean rápidos y no necesiten enviar 5s de audio.
fn make_chunked_rig(chunk_secs: f32) -> ChunkedRig {
    let capture = MockCapture::new(16_000);
    let hotkey = MockHotkey::new();
    let transcriber = MockTimedTranscriber::new();
    let injector = MockInjector::new();

    let capture_h = capture.handle();
    let hotkey_h = hotkey.handle();
    let transcriber_h = transcriber.handle();
    let injector_h = injector.handle();

    let cfg = ChunkedPipelineConfig {
        capture: Box::new(capture),
        transcriber: Arc::new(transcriber) as Arc<dyn Transcriber>,
        injector: Arc::new(injector) as Arc<dyn Injector>,
        hotkey: Box::new(hotkey),
        hotkey_binding: "F8".into(),
        chunk_duration_secs: chunk_secs,
    };
    let mut pipeline = ChunkedPipeline::new(cfg);
    pipeline.start().expect("start del chunked pipeline");
    let events = pipeline.events();

    ChunkedRig {
        pipeline,
        hotkey: hotkey_h,
        capture: capture_h,
        injector: injector_h,
        transcriber: transcriber_h,
        events,
    }
}

// ----- Helpers --------------------------------------------------------------

fn wait_for_state(
    rx: &crossbeam_channel::Receiver<PipelineEvent>,
    target: PipelineState,
    timeout: Duration,
) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(20)) {
            Ok(PipelineEvent::State(s)) if s == target => return true,
            Ok(_) => continue,
            Err(_) => return false,
        }
    }
    false
}

fn drain_events(rx: &crossbeam_channel::Receiver<PipelineEvent>, duration: Duration) {
    let deadline = std::time::Instant::now() + duration;
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(20)) {
            Ok(_) => continue,
            Err(_) => break,
        }
    }
}

fn send_silence_and_drain(rig: &ChunkedRig, duration_ms: u32) {
    rig.capture
        .send(AudioFrame::silence(duration_ms, rig.capture.sample_rate()));
    // Respiro para que el consumer thread procese el frame.
    thread::sleep(Duration::from_millis(80));
}

// ----- Tests ----------------------------------------------------------------

/// Hold + release sin audio: buffer vacío → no se encola nada → no se
/// inyecta nada → estado final Idle.
#[test]
fn chunked_hold_release_empty_does_not_inject() {
    let mut rig = make_chunked_rig(1.0);

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(200));

    assert!(
        rig.injector.texts().is_empty(),
        "no se debe inyectar nada sin audio; obtuve {:?}",
        rig.injector.texts()
    );
    assert_eq!(
        rig.transcriber.call_count(),
        0,
        "transcribe_timed no debe llamarse sin audio"
    );

    rig.pipeline.shutdown().ok();
}

/// Un dictado más largo que el chunk size produce múltiples inyecciones
/// incrementales (una por bloque). Verifica el pipelining: el consumer
/// corta bloques mientras graba, sin esperar al release.
#[test]
fn chunked_long_recording_produces_multiple_injections() {
    // chunk_secs=1.0 → chunk_size=16.000 muestras (1s de audio).
    let mut rig = make_chunked_rig(1.0);

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );

    // Enviamos 3s de audio en bloques de 1s. Cada segundo debe disparar
    // un corte + transcripción + inyección.
    for _ in 0..3 {
        send_silence_and_drain(&rig, 1000);
    }

    // Damos tiempo a que los workers terminen.
    drain_events(&rig.events, Duration::from_millis(800));

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    let texts = rig.injector.texts();
    // Esperamos al menos 2 inyecciones (3s de audio con chunks de 1s
    // deberían dar 3 bloques, pero el carryover y el timing del
    // consumer pueden variar). Aceptamos >= 2 para no flakar.
    assert!(
        texts.len() >= 2,
        "un dictado de 3s con chunks de 1s debe producir >= 2 inyecciones; obtuve {:?}",
        texts
    );

    rig.pipeline.shutdown().ok();
}

/// El resto final (audio que no alcanzó a llenar un chunk) se transcribe
/// al soltar la tecla. Verifica que `on_release` encola el remainder.
#[test]
fn chunked_release_flushes_remainder() {
    // chunk_secs=2.0 → chunk_size=32.000 (2s). Enviamos solo 1s: no
    // alcanza para un chunk completo, así que el remainder debe
    // transcribirse al release.
    let mut rig = make_chunked_rig(2.0);

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 1000);

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(800));

    let texts = rig.injector.texts();
    assert_eq!(
        texts.len(),
        1,
        "el remainder debe transcribirse una vez; obtuve {:?}",
        texts
    );

    rig.pipeline.shutdown().ok();
}

/// Si el STT falla en un bloque, el pipeline emite Error + Idle pero no
/// cae: el siguiente bloque puede procesarse normalmente.
#[test]
fn chunked_stt_failure_emits_error_and_continues() {
    let mut rig = make_chunked_rig(1.0);

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );

    rig.transcriber.set_fail(true);
    send_silence_and_drain(&rig, 1000);
    // Drenar el evento de error esperado.
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Error,
        Duration::from_millis(500),
    );
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    // Recuperar y enviar otro bloque.
    rig.transcriber.set_fail(false);
    send_silence_and_drain(&rig, 1000);
    drain_events(&rig.events, Duration::from_millis(500));

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    // El segundo bloque debe haberse inyectado tras la recuperación.
    assert!(
        !rig.injector.texts().is_empty(),
        "tras recuperar de un fallo, el siguiente bloque debe inyectarse; obtuve {:?}",
        rig.injector.texts()
    );

    rig.pipeline.shutdown().ok();
}
