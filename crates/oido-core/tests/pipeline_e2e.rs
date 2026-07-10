//! Integration tests para el pipeline MVP `dicta-y-pega`.
//!
//! NO toca OS (sin cpal, global-hotkey, arboard, whisper.cpp). Inyecta
//! mocks vía los traits de `oido-platform` y `oido-stt` y verifica el
//! flujo completo press → STT → filtro → inyección.
//!
//! Reglas del proyecto que respetamos:
//! - `parking_lot::Mutex` para estado mutable compartido.
//! - `Box<dyn Trait>` + `Arc<dyn Trait>` en los seams del pipeline.
//! - sin `unsafe` (Regla R2).
//!
//! Patrón: cada mock tiene un `Handle` (clonable, barato) que comparte
//! el estado con el mock movido al pipeline. Así el test puede
//! configurar y observar sin tener acceso al objeto dentro del
//! `Box<dyn _>`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::Sender;
use oido_core::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
use oido_platform::{AudioFrame, CaptureSource, Hotkey, Injector, PlatformError};
use oido_stt::{SttError, Transcriber};

// ----- Mock capture ---------------------------------------------------------

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
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), PlatformError> {
        *self.inner.sink.lock() = Some(sink);
        Ok(())
    }
    fn start(&mut self) -> Result<(), PlatformError> {
        Ok(())
    }
    fn stop(&mut self) -> Result<(), PlatformError> {
        // Importante para que el consumer thread termine su `recv()`
        // con `Err` y salga del loop.
        *self.inner.sink.lock() = None;
        Ok(())
    }
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate
    }
}

// ----- Mock hotkey ----------------------------------------------------------

// `Box<dyn Fn()>` no es Debug, así que el inner no deriva Debug.
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
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), PlatformError> {
        // El mock ignora el binding (no tiene OS-level semantics); los
        // tests del parser en `oido-platform::hotkey` ejercitan la
        // conversión `&str → (Modifiers, Code)` independientemente.
        let _ = binding;
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

// ----- Mock transcriber -----------------------------------------------------

#[derive(Debug)]
struct MockTranscriberInner {
    response: parking_lot::Mutex<String>,
    fail: AtomicBool,
}

#[derive(Clone)]
struct MockTranscriberHandle {
    inner: Arc<MockTranscriberInner>,
}

impl std::fmt::Debug for MockTranscriberHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockTranscriberHandle").finish()
    }
}

#[derive(Debug)]
struct MockTranscriber {
    inner: Arc<MockTranscriberInner>,
}

impl MockTranscriber {
    fn new(initial: &str) -> Self {
        Self {
            inner: Arc::new(MockTranscriberInner {
                response: parking_lot::Mutex::new(initial.to_owned()),
                fail: AtomicBool::new(false),
            }),
        }
    }

    fn handle(&self) -> MockTranscriberHandle {
        MockTranscriberHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockTranscriberHandle {
    fn set_response(&self, s: &str) {
        *self.inner.response.lock() = s.to_owned();
    }
    fn set_fail(&self, fail: bool) {
        self.inner.fail.store(fail, Ordering::SeqCst);
    }
}

impl Transcriber for MockTranscriber {
    fn transcribe(&self, _audio: &[f32]) -> Result<String, SttError> {
        if self.inner.fail.load(Ordering::SeqCst) {
            return Err(SttError::Backend("mock failure".into()));
        }
        Ok(self.inner.response.lock().clone())
    }
    fn load_model(&mut self, _path: &Path) -> Result<(), SttError> {
        Ok(())
    }
}

// ----- Mock injector --------------------------------------------------------

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
    fn inject(&self, text: &str) -> Result<(), PlatformError> {
        self.inner.texts.lock().push(text.to_owned());
        Ok(())
    }
}

// ----- Harness --------------------------------------------------------------

/// Lo que el test retiene: el pipeline (para shutdown) + handles para
/// disparar hotkey, mandar audio, observar el injector y reconfigurar
/// el transcriber.
struct Rig {
    pipeline: Pipeline,
    hotkey: MockHotkeyHandle,
    capture: MockCaptureHandle,
    injector: MockInjectorHandle,
    transcriber: MockTranscriberHandle,
    events: crossbeam_channel::Receiver<PipelineEvent>,
}

fn make_rig(initial_response: &str) -> Rig {
    let capture = MockCapture::new(16_000);
    let hotkey = MockHotkey::new();
    let transcriber = MockTranscriber::new(initial_response);
    let injector = MockInjector::new();

    // Handles para el test.
    let capture_h = capture.handle();
    let hotkey_h = hotkey.handle();
    let transcriber_h = transcriber.handle();
    let injector_h = injector.handle();

    let cfg = PipelineConfig {
        capture: Box::new(capture),
        transcriber: Arc::new(transcriber) as Arc<dyn Transcriber>,
        injector: Arc::new(injector) as Arc<dyn Injector>,
        hotkey: Box::new(hotkey),
        hotkey_binding: "F8".into(),
    };
    let mut pipeline = Pipeline::new(cfg);
    pipeline.start().expect("start del pipeline con mocks");
    let events = pipeline.events();

    Rig {
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

fn send_silence_and_drain(rig: &Rig, duration_ms: u32) {
    rig.capture
        .send(AudioFrame::silence(duration_ms, rig.capture.sample_rate()));
    // El consumer thread debe drenar el canal antes de que `release()`
    // haga snapshot del buffer. Sin este respiro hay una race real:
    // `on_release` puede ver el buffer vacío si el consumer no alcanzó
    // a procesar el frame todavía.
    thread::sleep(Duration::from_millis(50));
}

// ----- Tests ----------------------------------------------------------------

/// Hold+release sin audio: buffer vacío → STT no se llama → no se
/// inyecta nada → estado final Idle.
#[test]
fn hold_release_empty_does_not_inject() {
    let mut rig = make_rig("hola mundo");

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
        "no se debe inyectar nada si no hubo audio; obtuve {:?}",
        rig.injector.texts()
    );

    rig.pipeline.shutdown().ok();
}

/// Hold → audio → release con transcripción normal: el injector recibe
/// exactamente el texto del transcriber mock.
#[test]
fn hold_release_injects_transcribed_text() {
    let mut rig = make_rig("hola mundo");

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 500);

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    assert_eq!(rig.injector.texts(), vec!["hola mundo".to_string()]);

    rig.pipeline.shutdown().ok();
}

/// Si el STT devuelve una frase de la blacklist de `phrase_filter`, el
/// pipeline la descarta y no inyecta nada.
#[test]
fn hold_release_filters_hallucinated_phrase() {
    let mut rig = make_rig("thank you for watching");

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 500);

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    assert!(
        rig.injector.texts().is_empty(),
        "frase alucinada debe descartarse; injector recibió {:?}",
        rig.injector.texts()
    );

    rig.pipeline.shutdown().ok();
}

/// Si el STT devuelve error, el pipeline loggea y vuelve a Idle sin
/// inyectar.
#[test]
fn stt_error_does_not_inject() {
    let mut rig = make_rig("hola");
    rig.transcriber.set_fail(true);

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 500);

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    assert!(
        rig.injector.texts().is_empty(),
        "STT error no debe llegar al injector; injector recibió {:?}",
        rig.injector.texts()
    );

    rig.pipeline.shutdown().ok();
}

/// Tres ciclos hold-release consecutivos: cada uno inyecta su texto
/// (el pipeline no acumula dedup entre ciclos: cada press resetea el
/// estado del buffer).
#[test]
fn multiple_cycles_each_inject() {
    let mut rig = make_rig("hola");

    for _ in 0..3 {
        rig.hotkey.press();
        let _ = wait_for_state(
            &rig.events,
            PipelineState::Recording,
            Duration::from_millis(200),
        );
        send_silence_and_drain(&rig, 300);
        rig.hotkey.release();
        let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));
    }

    assert_eq!(
        rig.injector.texts(),
        vec!["hola".to_string(), "hola".to_string(), "hola".to_string()],
        "cada ciclo inyecta su transcripción independientemente"
    );

    rig.pipeline.shutdown().ok();
}

/// Secuencia de estados observados durante un ciclo completo:
/// Recording (al press) → Processing (entre STT e inject) → Idle.
#[test]
fn observed_states_during_cycle() {
    let mut rig = make_rig("hola");

    rig.hotkey.press();
    send_silence_and_drain(&rig, 500);
    rig.hotkey.release();

    let mut states = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_millis(500);
    while std::time::Instant::now() < deadline && states.len() < 3 {
        if let Ok(PipelineEvent::State(s)) = rig.events.recv_timeout(Duration::from_millis(20)) {
            states.push(s);
        }
    }

    assert_eq!(
        states,
        vec![
            PipelineState::Recording,
            PipelineState::Processing,
            PipelineState::Idle,
        ],
        "secuencia esperada Recording → Processing → Idle; obtuve {:?}",
        states
    );

    rig.pipeline.shutdown().ok();
}

/// El filtro de frase distingue mayúsculas y sólo matchea la frase
/// completa (no sub-string). Complementario a los tests unitarios de
/// `phrase_filter` pero ejercitado vía el pipeline completo.
#[test]
fn phrase_filter_case_insensitive_full_match() {
    let mut rig = make_rig("Thank You For Watching");

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 500);

    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    assert!(
        rig.injector.texts().is_empty(),
        "frase en mayúsculas también debe descartarse (case-insensitive)"
    );

    rig.pipeline.shutdown().ok();
}

/// Verifica que podemos reconfigurar la respuesta del transcriber
/// entre ciclos para simular cambios de modelo o contexto.
#[test]
fn transcriber_response_can_be_reconfigured() {
    let mut rig = make_rig("primero");

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 300);
    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    rig.transcriber.set_response("segundo");

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 300);
    rig.hotkey.release();
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));

    assert_eq!(
        rig.injector.texts(),
        vec!["primero".to_string(), "segundo".to_string()],
        "el segundo ciclo debe usar la respuesta reconfigurada"
    );

    rig.pipeline.shutdown().ok();
}

/// Hotkey `on_release` debe ser no-bloqueante: vuelve al callback en
/// microsegundos (no espera la inferencia STT). Esto se verifica
/// cronometrando el `release()` y observando que el estado vuelve a
/// `Idle` (transición completa) después.
#[test]
fn on_release_is_non_blocking() {
    let mut rig = make_rig("hola");

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 500);

    // Cronometramos release(): el callback debería encolar el buffer y
    // volver en microsegundos (no esperar al STT).
    let started = std::time::Instant::now();
    rig.hotkey.release();
    let elapsed = started.elapsed();

    // release() retorna inmediatamente. Le damos un margen amplio
    // porque en Windows el parking_lot::Mutex::lock puede tardar
    // algo en condiciones de contención.
    assert!(
        elapsed < Duration::from_millis(50),
        "release() debe ser no-bloqueante, tardó {elapsed:?}"
    );

    // La transición completa sí ocurre asincrónicamente.
    let _ = wait_for_state(&rig.events, PipelineState::Idle, Duration::from_millis(500));
    assert_eq!(rig.injector.texts(), vec!["hola".to_string()]);

    rig.pipeline.shutdown().ok();
}

/// Cuando el STT falla, el pipeline emite `Error` antes de volver a
/// `Idle` para que la UI pueda mostrar feedback.
#[test]
fn stt_error_emits_error_state() {
    let mut rig = make_rig("hola");
    rig.transcriber.set_fail(true);

    rig.hotkey.press();
    let _ = wait_for_state(
        &rig.events,
        PipelineState::Recording,
        Duration::from_millis(200),
    );
    send_silence_and_drain(&rig, 500);
    rig.hotkey.release();

    // Recolectamos todos los estados durante el ciclo (con margen
    // amplio porque el recv_timeout del canal puede tardar).
    let mut states = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_millis(2000);
    while std::time::Instant::now() < deadline {
        match rig.events.recv_timeout(Duration::from_millis(50)) {
            Ok(PipelineEvent::State(s)) => {
                if s == PipelineState::Error || s == PipelineState::Idle {
                    states.push(s);
                }
                if states.contains(&PipelineState::Error) && states.contains(&PipelineState::Idle) {
                    break;
                }
            }
            Ok(PipelineEvent::Shutdown) => break,
            Err(_) => break,
        }
    }

    assert!(
        states.contains(&PipelineState::Error),
        "STT error debe emitir estado Error; secuencia: {states:?}"
    );
    assert!(
        states.contains(&PipelineState::Idle),
        "tras Error debe volver a Idle; secuencia: {states:?}"
    );

    rig.pipeline.shutdown().ok();
}

/// `warm_up` (default del trait) debe ser un no-op seguro para que el
/// bin lo llame sin chequeos. Mock lo hereda del default.
#[test]
fn warm_up_is_safe_noop_for_mock() {
    let t = MockTranscriber::new("hola");
    let t: Arc<dyn Transcriber> = Arc::new(t);
    // No falla aunque el mock no sobreescriba warm_up.
    t.warm_up().expect("warm_up default debe ser Ok");
}
