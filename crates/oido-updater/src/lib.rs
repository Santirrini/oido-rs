//! Auto-updater MSI del bin `oido`.
//!
//! Este crate expone su API sólo cuando se compila con la feature
//! `updater` (activa). Sin esa feature, el crate compila como un
//! módulo raíz vacío con este docstring — un `use oido_updater::*`
//! en ese modo da un error claro ("no items") en vez del críptico
//! "crate sin items" que se obtiene con `#![cfg]` a nivel de crate.

#[cfg(feature = "updater")]
mod inner;

#[cfg(feature = "updater")]
pub use inner::{check_and_apply, download_file, verify_sha256, Status, BIN_NAME, REPO};
