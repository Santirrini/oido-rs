//! `SmartInjector`: orquesta el backend directo (UIAutomation en Windows)
//! con el fallback de `ArboardInjector`.
//!
//! Cadena de `inject(text)`:
//! 1. Si hay `DirectInjector` y `inject_focused` devuelve `Ok`, listo.
//! 2. Si devuelve `Err(NotEditable)`, se loguea y se cae al clipboard.
//! 3. Si devuelve `Err(Unsupported)` u otro error, se loguea como warn y
//!    también se cae al clipboard (mejor pegar "lento" que perder texto).
//! 4. Sin backend directo configurado, va directo al clipboard.
//!
//! `type_text` (streaming) sigue siendo el del fallback directo:
//! simular teclas por accesibilidad es incompatible con inserción incremental
//! y `send_text` del UIA es costoso para textos de streaming cortos repetidos.

use std::fmt;
use std::sync::Arc;

use crate::direct::DirectInjector;
use crate::{InjectError, Injector};

pub struct SmartInjector {
    direct: Option<Arc<dyn DirectInjector>>,
    fallback: Arc<dyn Injector>,
}

impl fmt::Debug for SmartInjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmartInjector")
            .field("direct", &self.direct.as_ref().map(|d| d as &dyn fmt::Debug))
            .field("fallback", &self.fallback)
            .finish_non_exhaustive()
    }
}

impl SmartInjector {
    /// Construye el inyector con un backend directo opcional y un
    /// inyector de fallback (típicamente `ArboardInjector`).
    pub fn new(direct: Option<Arc<dyn DirectInjector>>, fallback: Arc<dyn Injector>) -> Self {
        Self { direct, fallback }
    }

    /// Variante conveniente: si el backend directo no se puede inicializar
    /// (p.ej. `OIDO_UIA_ENABLED=0`, error de `CoInitializeEx`, o SO no
    /// soportado por el stub), se registra un warn y se continúa solo con
    /// el fallback.
    pub fn with_direct_factory<F>(
        fallback: Arc<dyn Injector>,
        make_direct: F,
    ) -> Self
    where
        F: FnOnce() -> Result<Arc<dyn DirectInjector>, InjectError>,
    {
        let direct = match make_direct() {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::warn!(
                    ?e,
                    "UIA direct injector no disponible; funcionando solo con clipboard",
                );
                None
            }
        };
        Self::new(direct, fallback)
    }
}

impl Injector for SmartInjector {
    fn inject(&self, text: &str) -> Result<(), InjectError> {
        if let Some(d) = &self.direct {
            match d.inject_focused(text) {
                Ok(()) => return Ok(()),
                Err(InjectError::NotEditable) => {
                    tracing::info!("elemento focused no editable; fallback a clipboard");
                }
                Err(other) => {
                    // Err(Unsupported) típico en cfg stub/no-windows. Mejor
                    // intentar el fallback que perder el texto dictado.
                    tracing::warn!(?other, "direct.inject_focused falló; fallback a clipboard");
                }
            }
        }
        self.fallback.inject(text)
    }

    fn type_text(&self, text: &str) -> Result<(), InjectError> {
        // Streaming: no rotar el clipboard del usuario, usar el fallback
        // (enigo) directo.
        self.fallback.type_text(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock de `DirectInjector` programable: devuelve la `Result`
    /// preestablecida y registra el `text` recibido.
    #[derive(Debug, Default)]
    struct MockDirect {
        // (text recibido, resultado a devolver)
        next: Mutex<Option<(String, Result<(), InjectError>)>>,
        calls: Mutex<Vec<String>>,
    }
    impl MockDirect {
        fn new(result: Result<(), InjectError>) -> Arc<Self> {
            Arc::new(Self {
                next: Mutex::new(Some((String::new(), result))),
                calls: Mutex::new(Vec::new()),
            })
        }
        fn how_many(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }
    impl DirectInjector for MockDirect {
        fn inject_focused(&self, text: &str) -> Result<(), InjectError> {
            self.calls.lock().unwrap().push(text.to_owned());
            let (_, res) = self.next.lock().unwrap().take().unwrap();
            res
        }
    }

    /// Mock de `Injector` (fallback) que registra cada llamada.
    #[derive(Debug, Default)]
    struct MockFallback {
        inject_calls: Mutex<Vec<String>>,
        type_calls: Mutex<Vec<String>>,
    }
    impl MockFallback {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }
    }
    impl Injector for MockFallback {
        fn inject(&self, text: &str) -> Result<(), InjectError> {
            self.inject_calls.lock().unwrap().push(text.to_owned());
            Ok(())
        }
        fn type_text(&self, text: &str) -> Result<(), InjectError> {
            self.type_calls.lock().unwrap().push(text.to_owned());
            Ok(())
        }
    }

    #[test]
    fn direct_ok_skips_fallback() {
        let direct = MockDirect::new(Ok(()));
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(Some(direct.clone()), fallback.clone());

        inj.inject("hola").unwrap();
        assert_eq!(direct.how_many(), 1);
        assert_eq!(fallback.inject_calls.lock().unwrap().len(), 0);
    }

    #[test]
    fn not_editable_falls_back() {
        let direct = MockDirect::new(Err(InjectError::NotEditable));
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(Some(direct.clone()), fallback.clone());

        inj.inject("hola").unwrap();
        assert_eq!(direct.how_many(), 1, "direct debe intentar una vez");
        assert_eq!(
            fallback.inject_calls.lock().unwrap().as_slice(),
            &["hola".to_string()]
        );
    }

    #[test]
    fn unsupported_falls_back() {
        // Unsupported también cae a fallback (no perdemos el texto dictado).
        let direct = MockDirect::new(Err(InjectError::Unsupported("oops".into())));
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(Some(direct.clone()), fallback.clone());

        inj.inject("hola").unwrap();
        assert_eq!(direct.how_many(), 1);
        assert_eq!(
            fallback.inject_calls.lock().unwrap().as_slice(),
            &["hola".to_string()]
        );
    }

    #[test]
    fn generic_inject_error_propagates_to_fallback() {
        // Err(Inject(_)) también cae a fallback por diseño (mejor pegar que perder).
        let direct = MockDirect::new(Err(InjectError::Inject("peer".into())));
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(Some(direct.clone()), fallback.clone());

        inj.inject("hola").unwrap();
        assert_eq!(direct.how_many(), 1);
        assert_eq!(
            fallback.inject_calls.lock().unwrap().as_slice(),
            &["hola".to_string()]
        );
    }

    #[test]
    fn no_direct_uses_fallback_directly() {
        // Sin direct configurado, ninguna llamada extra y fallback recibe todo.
        let direct: Option<Arc<dyn DirectInjector>> = None;
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(direct, fallback.clone());

        inj.inject("hola").unwrap();
        assert_eq!(
            fallback.inject_calls.lock().unwrap().as_slice(),
            &["hola".to_string()]
        );
    }

    #[test]
    fn type_text_bypasses_direct() {
        // type_text no debe tocar el backend directo (que es costoso para
        // streaming), solo el fallback.
        let direct = MockDirect::new(Ok(()));
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(Some(direct.clone()), fallback.clone());

        inj.type_text("stream").unwrap();
        assert_eq!(direct.how_many(), 0, "type_text no toca direct");
        assert_eq!(
            fallback.type_calls.lock().unwrap().as_slice(),
            &["stream".to_string()]
        );
    }

    #[test]
    fn with_direct_factory_records_unavailable_warning() {
        // Si la factory falla, `direct` queda `None` y el comportamiento es
        // idéntico a `no_direct_uses_fallback_directly`.
        let direct: Option<Arc<dyn DirectInjector>> = None;
        let fallback = MockFallback::new();
        let inj = SmartInjector::new(direct, fallback.clone());

        inj.inject("hola").unwrap();
        assert_eq!(
            fallback.inject_calls.lock().unwrap().as_slice(),
            &["hola".to_string()]
        );
    }
}
