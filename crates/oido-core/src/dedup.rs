//! Anti-alucinación: descarta segmentos repetidos consecutivamente.
//!
//! Port directo de la `SegmentDeduplicator` de Oido. La idea: whisper.cpp
//! tiende a repetir el último fragmento como "alucinación de relleno"
//! cuando el usuario deja de hablar. Basta guardar el último texto y
//! descartar la próxima lectura si es idéntica (después de trim).

/// Estructura con el último segmento aceptado. No thread-safe: un
/// `Dedup` por worker del filtro. Regla R1: si dos workers necesitasen
/// el mismo estado, replantear (en este caso, no lo necesitan).
#[derive(Debug, Default, Clone)]
pub struct Dedup {
    last: Option<String>,
}

impl Dedup {
    #[must_use]
    pub const fn new() -> Self {
        Self { last: None }
    }

    /// Procesa un nuevo segmento. Devuelve `Some(texto)` si debe
    /// propagarse, `None` si es duplicado del inmediato anterior.
    #[must_use]
    pub fn process(&mut self, next: &str) -> Option<&str> {
        let trimmed = next.trim();
        if trimmed.is_empty() {
            return None;
        }
        match &self.last {
            Some(prev) if prev == trimmed => None,
            _ => {
                self.last = Some(trimmed.to_owned());
                self.last.as_deref()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_discarded() {
        let mut d = Dedup::new();
        assert_eq!(d.process(""), None);
        assert_eq!(d.process("   "), None);
    }

    #[test]
    fn consecutive_dup_discarded() {
        let mut d = Dedup::new();
        assert_eq!(d.process("hola"), Some("hola"));
        assert_eq!(d.process("hola"), None);
        assert_eq!(d.process("hola"), None);
    }

    #[test]
    fn trim_whitespace_then_dedup() {
        let mut d = Dedup::new();
        assert_eq!(d.process("  hola  "), Some("hola"));
        assert_eq!(d.process("hola"), None);
        // case-sensitive: "Hola" NO es dup de "hola".
        assert_eq!(d.process(" Hola "), Some("Hola"));
        assert_eq!(d.process("Hola"), None);
    }

    #[test]
    fn distinct_kept() {
        let mut d = Dedup::new();
        assert_eq!(d.process("hola"), Some("hola"));
        assert_eq!(d.process("mundo"), Some("mundo"));
        assert_eq!(d.process("hola"), Some("hola"));
    }
}
