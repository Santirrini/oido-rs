//! Implementación `whisper.cpp` del trait `Transcriber`.
//!
//! TODO Fase 1:
//! - Enlazar `whisper-rs` con build estático.
//! - `WhisperImpl { ctx: NonNull<whisper_context> }`.
//! - `unsafe impl Send` (justificado en el doc-comment superior por el
//!   hecho de que whisper.cpp es thread-confined y usamos serialización
//!   en `oido-core`).
//! - Implementar `Transcriber` envolviendo las llamadas C.
//!
//! Mientras tanto, este archivo existe para mantener la regla R2
//! (**FFI aislado en un único archivo**) y servir de anclaje.

#![allow(unsafe_code)]

// Aquí llegan las declaraciones.
