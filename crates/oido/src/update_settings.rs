//! Adaptador: `Arc<ConfigStore>` → `dyn UpdateSettings`.
//!
//! Vive en el bin (no en `oido-config` ni en `oido-updater`) para que
//! la dirección de dependencias quede:
//!
//! ```text
//! oido-updater ──(trait UpdateSettings)── oido (bin)
//!                       ▲
//!                       │ impl
//!              oido_config::ConfigStore
//! ```
//!
//! Si el adapter viviera en `oido-config`, entonces `oido-config`
//! tendría que depender de `oido-updater` (acoplamiento inverso
//! innecesario). Aquí el bin es el punto de encuentro natural.
//!
//! **Regla R3 (AGENTS.md)**: el adapter NO introduce un `Mutex` propio.
//! Toda la sincronización va por el `parking_lot::Mutex` interno del
//! `ConfigStore`. `Arc<ConfigStore>` es lo único que se cruza entre
//! threads.

#[cfg(feature = "updater")]
use std::sync::Arc;

#[cfg(feature = "updater")]
use oido_config::ConfigStore;
#[cfg(feature = "updater")]
use oido_updater::UpdateSettings;

/// Newtype que implementa `UpdateSettings` delegando al `ConfigStore`.
///
/// `set_last_check` persiste inmediatamente vía `save()` — coherente
/// con cómo el resto del bin muta `Config` (snapshot → replace → save).
#[cfg(feature = "updater")]
pub struct ConfigStoreUpdateSettings(pub Arc<ConfigStore>);

#[cfg(feature = "updater")]
impl UpdateSettings for ConfigStoreUpdateSettings {
    fn auto_update(&self) -> bool {
        self.0.snapshot().update.auto_update
    }

    fn check_interval_hours(&self) -> u32 {
        self.0.snapshot().update.check_interval_hours
    }

    fn skipped_version(&self) -> Option<String> {
        self.0.snapshot().update.skipped_version
    }

    fn last_check(&self) -> Option<i64> {
        self.0.snapshot().update.last_check
    }

    fn set_last_check(&self, epoch_secs: i64) {
        let mut cfg = self.0.snapshot();
        cfg.update.last_check = Some(epoch_secs);
        self.0.replace(cfg);
        if let Err(e) = self.0.save() {
            tracing::warn!(?e, "no se pudo persistir last_check del updater");
        }
    }
}
