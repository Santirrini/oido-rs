//! `UpdateScheduler`: thread en background que chequea updates
//! periódicamente siguiendo el patrón canónico del codebase
//! (`oido-downloader` / `oido-mic-probe`).
//!
//! Contrato:
//! - **Cumple R1 (AGENTS.md)**: comunicación con el resto del bin
//!   vía `crossbeam_channel` (bounded). Nunca expone `Arc<Mutex<...>>`.
//! - **Cumple R3**: lee/escribe la `UpdateConfig` a través del
//!   `ConfigStore` existente (`parking_lot::Mutex` global). No crea
//!   su propio Mutex.
//! - **No es async**: el bin es 100% síncrono (`std::thread` +
//!   `crossbeam_channel`). El control loop del bin hace `try_recv` y
//!   debe drenar también los `UpdateEvent`.
//!
//! Política de checks:
//! - Al `spawn`, hace un check inmediato si `auto_update` y
//!   `last_check` es `None` o `> now - interval_hours`.
//! - Cada `tick_interval` (1h) revisa si toca check.
//! - Si la última versión == `skipped_version`, se reporta
//!   `UpdateEvent::Skipped` (no se molesta al usuario) pero NO se
//!   actualiza `last_check`.
//! - Cualquier otro resultado (UpToDate, UpdateAvailable, Failed)
//!   persiste `last_check = now`.

use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::Sender;

use crate::backend::InstallerBackend;
use crate::error::UpdateError;
use crate::status::{CheckResult, UpdateEvent};

/// Snapshot mutable de la `UpdateConfig` que el scheduler lee/escribe.
///
/// Lo definimos como trait aquí (no como struct concreto) para que
/// `oido-config::ConfigStore` lo implemente sin acoplamiento
/// inverso (`oido-updater` no debe depender de `oido-config`).
pub trait UpdateSettings: Send + Sync + 'static {
    /// ¿Auto-update habilitado?
    fn auto_update(&self) -> bool;
    /// Intervalo entre checks en horas. Default: 24.
    fn check_interval_hours(&self) -> u32;
    /// Versión que el usuario marcó "saltar". `None` = no saltar nada.
    fn skipped_version(&self) -> Option<String>;
    /// Último check (epoch segundos). `None` = nunca.
    fn last_check(&self) -> Option<i64>;
    /// Persistir `last_check = now`. Llamado por el scheduler tras un
    /// check no-skip.
    fn set_last_check(&self, epoch_secs: i64);
}

/// Sender de eventos hacia el bin. El bin hace `match` sobre los
/// `UpdateEvent` en el control loop. Tipo público para que el bin
/// pueda declarar el canal sin importar internals del scheduler.
pub type EventSender = Sender<UpdateEvent>;

/// Builder del scheduler. Usar `UpdateScheduler::builder()` para
/// configurar y `.spawn()` para arrancar el thread.
pub struct UpdateSchedulerBuilder {
    settings: Option<Box<dyn UpdateSettings>>,
    tx: Option<EventSender>,
    backend: Option<Box<dyn InstallerBackend>>,
    tick_interval: Duration,
}

impl std::fmt::Debug for UpdateSchedulerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateSchedulerBuilder")
            .field("settings", &self.settings.is_some())
            .field("tx", &self.tx.is_some())
            .field("backend", &self.backend.is_some())
            .field("tick_interval", &self.tick_interval)
            .finish()
    }
}

impl UpdateSchedulerBuilder {
    fn new() -> Self {
        Self {
            settings: None,
            tx: None,
            backend: None,
            tick_interval: Duration::from_secs(60 * 60), // 1h
        }
    }

    pub fn settings(mut self, s: Box<dyn UpdateSettings>) -> Self {
        self.settings = Some(s);
        self
    }

    pub fn event_sender(mut self, tx: EventSender) -> Self {
        self.tx = Some(tx);
        self
    }

    pub fn backend(mut self, b: Box<dyn InstallerBackend>) -> Self {
        self.backend = Some(b);
        self
    }

    /// Override del tick (default 1h). Útil para tests.
    pub fn tick_interval(mut self, d: Duration) -> Self {
        self.tick_interval = d;
        self
    }

    /// Lanza el thread. Devuelve el `JoinHandle` (el bin no lo usa
    /// normalmente; el thread es fire-and-forget como el
    /// `oido-downloader`).
    pub fn spawn(self) -> std::io::Result<JoinHandle<()>> {
        let settings = self
            .settings
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "settings"))?;
        let tx = self
            .tx
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "event_sender"))?;
        let backend = self
            .backend
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "backend"))?;
        let tick = self.tick_interval;

        thread::Builder::new()
            .name("oido-update-scheduler".into())
            .spawn(move || {
                run(settings, tx, backend, tick);
            })
    }
}

/// Handle liviano al scheduler ya spawneado. Permite `.stop()` en el
/// futuro (hoy: el thread termina cuando el bin cierra).
pub struct UpdateScheduler {
    _handle: JoinHandle<()>,
}

impl std::fmt::Debug for UpdateScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateScheduler").finish_non_exhaustive()
    }
}

impl UpdateScheduler {
    pub fn builder() -> UpdateSchedulerBuilder {
        UpdateSchedulerBuilder::new()
    }
}

fn run(
    settings: Box<dyn UpdateSettings>,
    tx: EventSender,
    backend: Box<dyn InstallerBackend>,
    tick: Duration,
) {
    tracing::info!("update scheduler iniciado");

    let span = tracing::info_span!("scheduler_tick");
    let _enter = span.enter();
    maybe_check(&*settings, &tx, &*backend);

    loop {
        thread::sleep(tick);
        maybe_check(&*settings, &tx, &*backend);
    }
}

/// ¿Toca chequear? Si sí, ejecuta y emite el evento correspondiente.
fn maybe_check(settings: &dyn UpdateSettings, tx: &EventSender, backend: &dyn InstallerBackend) {
    if !settings.auto_update() {
        tracing::debug!("auto_update=false, skip");
        return;
    }

    let now = now_epoch_secs();
    let interval = (settings.check_interval_hours() as i64).saturating_mul(60 * 60);
    match settings.last_check() {
        Some(prev) if now - prev < interval => {
            tracing::debug!(
                elapsed = now - prev,
                interval,
                "scheduler: todavía no toca check"
            );
            return;
        }
        _ => {}
    }

    let current = env!("CARGO_PKG_VERSION");
    let event = match crate::check_for_update(current) {
        Ok(check) => check_to_event(check, settings),
        Err(e) => UpdateEvent::Failed {
            reason: e.user_message(),
        },
    };

    persist_last_check(settings, now);

    if tx.send(event).is_err() {
        tracing::warn!("receiver colgado, scheduler saliendo");
    }

    let _ = backend;
}

fn check_to_event(check: CheckResult, settings: &dyn UpdateSettings) -> UpdateEvent {
    if !check.has_update {
        return UpdateEvent::UpToDate;
    }
    if settings.skipped_version().as_deref() == Some(check.latest.as_str()) {
        return UpdateEvent::Skipped {
            version: check.latest,
        };
    }
    UpdateEvent::UpdateAvailable {
        version: check.latest,
        current: check.current,
    }
}

fn persist_last_check(settings: &dyn UpdateSettings, now: i64) {
    settings.set_last_check(now);
}

fn now_epoch_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Helper público: ejecutar un check one-shot y devolver el evento
/// correspondiente. Usado por el handler del menú "Buscar
/// actualizaciones" para no duplicar lógica. El caller encola el
/// evento en su canal de control.
///
/// `install` = `true` descarga+instala si hay update; `false` sólo
/// notifica (para flujo manual: "veo si hay algo y aviso").
#[tracing::instrument(skip(backend))]
pub fn emit_one_shot_check(backend: &dyn InstallerBackend, install: bool) -> UpdateEvent {
    let current = env!("CARGO_PKG_VERSION");
    let check = match crate::check_for_update(current) {
        Ok(c) => c,
        Err(e) => {
            return UpdateEvent::Failed {
                reason: e.user_message(),
            }
        }
    };
    if !check.has_update {
        return UpdateEvent::UpToDate;
    }
    if !install {
        return UpdateEvent::UpdateAvailable {
            version: check.latest,
            current: check.current,
        };
    }
    match crate::check_and_apply(backend) {
        Ok(crate::Status::UpToDate) => UpdateEvent::UpToDate,
        Ok(crate::Status::DownloadedAndInstalling { version }) => UpdateEvent::Updated { version },
        Err(e) => UpdateEvent::Failed {
            reason: e.user_message(),
        },
    }
}

/// Helper testeable: dado `now`, `last_check` y `interval_hours`,
/// ¿toca check?
pub fn should_check_now(now: i64, last_check: Option<i64>, interval_hours: u32) -> bool {
    let interval = (interval_hours as i64).saturating_mul(60 * 60);
    match last_check {
        None => true,
        Some(prev) => now.saturating_sub(prev) >= interval,
    }
}

/// Helper testeable: convertir `CheckResult` + skipped_version en `UpdateEvent`.
#[allow(dead_code)] // expuesto para tests futuros / otros consumidores
pub fn check_result_to_event(check: &CheckResult, skipped_version: Option<&str>) -> UpdateEvent {
    if !check.has_update {
        return UpdateEvent::UpToDate;
    }
    if skipped_version == Some(check.latest.as_str()) {
        return UpdateEvent::Skipped {
            version: check.latest.clone(),
        };
    }
    UpdateEvent::UpdateAvailable {
        version: check.latest.clone(),
        current: check.current.clone(),
    }
}

/// ¿Esta variante de `UpdateError` debe reportarse al usuario vía
/// `UpdateEvent::Failed`? Las benignas (`UnsupportedPlatform` cuando
/// no hay backend para el OS actual, por ejemplo) NO.
pub fn should_emit_error(err: &UpdateError) -> bool {
    err.is_user_visible()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_check_now_first_run() {
        assert!(should_check_now(1_000_000, None, 24));
    }

    #[test]
    fn should_check_now_within_interval() {
        assert!(!should_check_now(1_000_000 + 3600, Some(1_000_000), 24));
    }

    #[test]
    fn should_check_now_past_interval() {
        assert!(should_check_now(1_000_000 + 25 * 3600, Some(1_000_000), 24));
    }

    #[test]
    fn check_to_event_up_to_date() {
        let check = CheckResult {
            current: "0.1.0".into(),
            latest: "0.1.0".into(),
            has_update: false,
        };
        assert_eq!(check_result_to_event(&check, None), UpdateEvent::UpToDate);
    }

    #[test]
    fn check_to_event_skipped() {
        let check = CheckResult {
            current: "0.1.0".into(),
            latest: "0.2.0".into(),
            has_update: true,
        };
        let ev = check_result_to_event(&check, Some("0.2.0"));
        assert!(matches!(ev, UpdateEvent::Skipped { .. }));
    }

    #[test]
    fn check_to_event_available() {
        let check = CheckResult {
            current: "0.1.0".into(),
            latest: "0.2.0".into(),
            has_update: true,
        };
        let ev = check_result_to_event(&check, None);
        assert!(matches!(ev, UpdateEvent::UpdateAvailable { .. }));
    }
}
