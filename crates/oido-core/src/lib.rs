//! Orquestación del pipeline de dictado: conecta
//! `CaptureSource → Transcriber → Filtro (dedup + frase) → Injector`
//! vía `crossbeam::channel`. Sin estado mutable compartido entre etapas.
//!
//! La implementación llega en Fase 1. Aquí solo se declara la forma del
//! pipeline para que el resto del workspace pueda compilar contra ella.

#![doc = include_str!("../../../ARCHITECTURE.md")]

pub mod dedup;
pub mod phrase_filter;

// Re-exporta los tipos de dato cuyo origen es el SO (`oido-platform`).
// El pipeline trabaja con estos tipos sin redefinirlos.
pub use oido_platform::{AudioFrame, AudioRx, AudioTx, InjectedText, TextRx, TextTx};
