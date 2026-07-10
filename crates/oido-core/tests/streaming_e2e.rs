//! Integration tests para el pipeline de dictado interactivo en Streaming.
//!
//! Verifica el flujo completo usando mocks para capturar audio, hotkey global
//! e inyección de texto por teclado individual, controlando de manera precisa
//! las pasadas temporales y el algoritmo de LocalAgreement-2.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use parking_lot::Mutex;

use crossbeam_channel::Sender;
use oido_core::{StreamingPipeline, StreamingPipelineConfig, PipelineEvent, PipelineState};
use oido_platform::{AudioFrame, CaptureSource, Hotkey, Injector, PlatformError};
use oido_stt::{SttError, PartialTranscript, Streamer};

// ----- Mock capture ---------------------------------------------------------
#[derive(Debug)]
struct MockCaptureInner {
    sink: Mutex<Option<Sender<AudioFrame>>>,
    sample_rate: u32,
}

#[derive(Clone, Debug)]
struct MockCaptureHandle {
    inner: Arc<MockCaptureInner>,
}

#[derive(Debug)]
struct MockCapture {
    inner: Arc<MockCaptureInner>,
}

impl MockCapture {
    fn new(sample_rate: u32) -> Self {
        Self {
            inner: Arc::new(MockCaptureInner {
                sink: Mutex::new(None),
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
}

impl CaptureSource for MockCapture {
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), PlatformError> {
        *self.inner.sink.lock() = Some(sink);
        Ok(())
    }
    fn start(&mut self) -> Result<(), PlatformError> {
        Ok(())
    }
    fn stop(&mut self) -> Result<(), PlatformError> {
        *self.inner.sink.lock() = None;
        Ok(())
    }
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate
    }
}

// ----- Mock hotkey ----------------------------------------------------------
struct MockHotkeyInner {
    on_press: Mutex<Option<Box<dyn Fn() + Send + 'static>>>,
    on_release: Mutex<Option<Box<dyn Fn() + Send + 'static>>>,
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
                on_press: Mutex::new(None),
                on_release: Mutex::new(None),
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
    ) -> Result<(), PlatformError> {
        *self.inner.on_press.lock() = Some(on_press);
        *self.inner.on_release.lock() = Some(on_release);
        Ok(())
    }
    fn unregister(&mut self) -> Result<(), PlatformError> {
        *self.inner.on_press.lock() = None;
        *self.inner.on_release.lock() = None;
        Ok(())
    }
}

// ----- Mock injector --------------------------------------------------------
#[derive(Debug, Default)]
struct MockInjectorInner {
    texts: Mutex<Vec<String>>,
}

#[derive(Clone, Debug)]
struct MockInjectorHandle {
    inner: Arc<MockInjectorInner>,
}

#[derive(Debug)]
struct MockInjector {
    inner: Arc<MockInjectorInner>,
}

impl MockInjector {
    fn new() -> Self {
        Self {
            inner: Arc::new(MockInjectorInner::default()),
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
    fn inject(&self, text: &str) -> Result<(), PlatformError> {
        self.inner.texts.lock().push(text.to_owned());
        Ok(())
    }
    fn type_text(&self, text: &str) -> Result<(), PlatformError> {
        self.inner.texts.lock().push(text.to_owned());
        Ok(())
    }
}

// ----- Mock streamer --------------------------------------------------------
#[derive(Debug, Default)]
struct MockStreamerInner {
    process_responses: Mutex<Vec<PartialTranscript>>,
    flush_response: Mutex<PartialTranscript>,
    reset_called: AtomicBool,
}

#[derive(Clone, Debug)]
struct MockStreamerHandle {
    inner: Arc<MockStreamerInner>,
}

#[derive(Debug)]
struct MockStreamer {
    inner: Arc<MockStreamerInner>,
}

impl MockStreamer {
    fn new() -> Self {
        Self {
            inner: Arc::new(MockStreamerInner::default()),
        }
    }
    fn handle(&self) -> MockStreamerHandle {
        MockStreamerHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockStreamerHandle {
    fn set_process_responses(&self, resps: Vec<PartialTranscript>) {
        *self.inner.process_responses.lock() = resps;
    }
    fn set_flush_response(&self, resp: PartialTranscript) {
        *self.inner.flush_response.lock() = resp;
    }
    fn was_reset_called(&self) -> bool {
        self.inner.reset_called.load(Ordering::SeqCst)
    }
    fn clear_reset(&self) {
        self.inner.reset_called.store(false, Ordering::SeqCst);
    }
}

impl Streamer for MockStreamer {
    fn process(&mut self, _audio: &[f32]) -> Result<PartialTranscript, SttError> {
        let mut guard = self.inner.process_responses.lock();
        if guard.is_empty() {
            Ok(PartialTranscript::default())
        } else {
            Ok(guard.remove(0))
        }
    }
    fn flush_final(&mut self) -> Result<PartialTranscript, SttError> {
        self.reset();
        Ok(self.inner.flush_response.lock().clone())
    }
    fn reset(&mut self) {
        self.inner.reset_called.store(true, Ordering::SeqCst);
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

// ----- Test Suite -----------------------------------------------------------

#[test]
fn test_streaming_e2e_normal_dictation() {
    let capture = MockCapture::new(16000);
    let capture_handle = capture.handle();
    let hotkey = MockHotkey::new();
    let hotkey_handle = hotkey.handle();
    let injector = MockInjector::new();
    let injector_handle = injector.handle();
    let streamer = MockStreamer::new();
    let streamer_handle = streamer.handle();

    // Programar respuestas del transcriptor
    streamer_handle.set_process_responses(vec![
        PartialTranscript {
            confirmed: "Hello ".to_owned(),
            unconfirmed: "worl".to_owned(),
        },
        PartialTranscript {
            confirmed: "world".to_owned(),
            unconfirmed: "!".to_owned(),
        },
    ]);
    streamer_handle.set_flush_response(PartialTranscript {
        confirmed: "!".to_owned(),
        unconfirmed: String::new(),
    });

    let config = StreamingPipelineConfig {
        capture: Box::new(capture),
        streamer: Box::new(streamer),
        injector: Arc::new(injector),
        hotkey: Box::new(hotkey),
        hotkey_binding: "F8".to_string(),
    };

    let mut pipeline = StreamingPipeline::new(config);
    pipeline.start().unwrap();

    let events = pipeline.events();

    // 1) Presionamos tecla -> Estado cambia a Recording
    hotkey_handle.press();
    assert!(wait_for_state(&events, PipelineState::Recording, Duration::from_millis(200)));

    // Mandamos algo de audio
    capture_handle.send(AudioFrame {
        samples: vec![0.1; 4800],
        sample_rate_hz: 16000,
    });

    // Esperar al primer tick (~1.1s)
    thread::sleep(Duration::from_millis(1100));
    assert_eq!(injector_handle.texts(), vec!["Hello "]);

    // Mandamos más audio
    capture_handle.send(AudioFrame {
        samples: vec![0.2; 4800],
        sample_rate_hz: 16000,
    });

    // Esperar al segundo tick (~1.1s)
    thread::sleep(Duration::from_millis(1100));
    assert_eq!(injector_handle.texts(), vec!["Hello ", "world"]);

    // 2) Soltamos tecla -> triggers final flush
    hotkey_handle.release();
    assert!(wait_for_state(&events, PipelineState::Processing, Duration::from_millis(200)));
    assert!(wait_for_state(&events, PipelineState::Idle, Duration::from_millis(200)));

    // Confirmar que el flush final inyectó la última parte "!" y se llamó al reset del streamer
    assert_eq!(injector_handle.texts(), vec!["Hello ", "world", "!"]);
    assert!(streamer_handle.was_reset_called());
}

#[test]
fn test_streaming_e2e_multiple_cycles_resets() {
    let capture = MockCapture::new(16000);
    let hotkey = MockHotkey::new();
    let hotkey_handle = hotkey.handle();
    let injector = MockInjector::new();
    let injector_handle = injector.handle();
    let streamer = MockStreamer::new();
    let streamer_handle = streamer.handle();

    let config = StreamingPipelineConfig {
        capture: Box::new(capture),
        streamer: Box::new(streamer),
        injector: Arc::new(injector),
        hotkey: Box::new(hotkey),
        hotkey_binding: "F8".to_string(),
    };

    let mut pipeline = StreamingPipeline::new(config);
    pipeline.start().unwrap();

    // --- Ciclo 1 ---
    streamer_handle.set_process_responses(vec![
        PartialTranscript {
            confirmed: "One ".to_owned(),
            unconfirmed: String::new(),
        }
    ]);
    streamer_handle.set_flush_response(PartialTranscript {
        confirmed: "Two".to_owned(),
        unconfirmed: String::new(),
    });
    streamer_handle.clear_reset();

    hotkey_handle.press();
    thread::sleep(Duration::from_millis(1100));
    hotkey_handle.release();
    thread::sleep(Duration::from_millis(50));

    assert_eq!(injector_handle.texts(), vec!["One ", "Two"]);
    assert!(streamer_handle.was_reset_called());

    // --- Ciclo 2 ---
    streamer_handle.set_process_responses(vec![
        PartialTranscript {
            confirmed: "Three ".to_owned(),
            unconfirmed: String::new(),
        }
    ]);
    streamer_handle.set_flush_response(PartialTranscript {
        confirmed: "Four".to_owned(),
        unconfirmed: String::new(),
    });
    streamer_handle.clear_reset();

    hotkey_handle.press();
    thread::sleep(Duration::from_millis(1100));
    hotkey_handle.release();
    thread::sleep(Duration::from_millis(50));

    assert_eq!(injector_handle.texts(), vec!["One ", "Two", "Three ", "Four"]);
    assert!(streamer_handle.was_reset_called());
}

#[test]
fn test_streaming_e2e_no_reentry_in_idle() {
    let capture = MockCapture::new(16000);
    let hotkey = MockHotkey::new();
    let hotkey_handle = hotkey.handle();
    let injector = MockInjector::new();
    let injector_handle = injector.handle();
    let streamer = MockStreamer::new();
    let streamer_handle = streamer.handle();

    let config = StreamingPipelineConfig {
        capture: Box::new(capture),
        streamer: Box::new(streamer),
        injector: Arc::new(injector),
        hotkey: Box::new(hotkey),
        hotkey_binding: "F8".to_string(),
    };

    let mut pipeline = StreamingPipeline::new(config);
    pipeline.start().unwrap();

    let events = pipeline.events();

    // Release sin haber presionado en Idle -> Debe ser un NO-OP absoluto.
    hotkey_handle.release();

    // Esperamos un momento para ver si sucede algo
    thread::sleep(Duration::from_millis(100));

    // No se inyecta texto, no se llama a flush, y el canal de eventos permanece vacío.
    assert!(injector_handle.texts().is_empty());
    assert!(!streamer_handle.was_reset_called());
    assert!(events.try_recv().is_err());
}
