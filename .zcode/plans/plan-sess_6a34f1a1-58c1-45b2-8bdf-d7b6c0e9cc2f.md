# Plan: Lógica de Transcripción Optimizada para Oido 2.0

## Objetivo

Optimizar la función principal (cargar modelo → click → grabar → soltar → procesar → pegar) para máxima eficiencia y mínima latencia, manteniendo las 3 reglas inviolables de Rust (R1/R2/R3) y el cronograma de Fase 1 del proyecto.

---

## Cuellos de botella identificados (estado actual)

| # | Problema | Ubicación | Impacto |
|---|----------|-----------|---------|
| C1 | STT corre **inline en el callback `on_release`** del hotkey | `pipeline.rs:137-172` | El thread del hotkey se bloquea durante toda la transcripción. No se puede grabar de nuevo hasta que termine. |
| C2 | **GPU desactivada**: `WhisperContextParameters::default()` → `n_gpu_layers=0` | `whisper_cpp.rs:108` | Mayor salto de rendimiento disponible no aprovechado. |
| C3 | **Sin `set_n_threads`**: no se controlan los hilos de cómputo | `whisper_cpp.rs:58-72` | whisper.cpp usa su default; no se adapta al hardware. |
| C4 | **Sin warm-up**: primer dictado sufre cold-start (carga lazy de pesos en la primera inferencia) | `main.rs:86-95` | Latencia pico en el primer uso. |
| C5 | **Bug de sample rate**: si el dispositivo no soporta 16kHz, whisper recibe audio corrupto | `capture.rs:56-71` | Salida basura/error en hardware común (ej. muchos USB mics son 44.1/48kHz). |
| C6 | **Sin VAD de silencio**: se transcribe silencio inicial/final innecesariamente | `whisper_cpp.rs:77-79` | Tiempo de inferencia perdido + riesgo de alucinación. |
| C7 | Parámetros subóptimos: falta `single_segment`, `set_no_context`, `set_temperature` | `whisper_cpp.rs:58-72` | Overhead de multi-segment cuando el audio de dictado es <30s. |

---

## Cambios por crate

### 1. `oido-stt` — Backend optimizado

#### 1a. Tuning de parámetros `whisper_cpp.rs`

Modificar `Transcriber::transcribe` para aplicar parámetros optimizados para dictado corto:

```rust
let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
// Configurar threads según hardware disponible
let n_threads = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(4)
    .min(8); // whisper.cpp no escala bien más allá de ~8 threads
params.set_n_threads(n_threads as i32);

// Idioma + supresión de output (existente)
params.set_language(self.language.as_deref());
params.set_print_realtime(false);
params.set_print_progress(false);
params.set_print_timestamps(false);
params.set_print_special(false);
params.set_suppress_blank(true);
params.set_suppress_nst(true);

// NUEVO: optimizaciones para dictado corto (<30s)
params.set_translate(false);
params.set_no_context(true);          // no usar contexto de llamadas previas
params.set_single_segment(true);      // audio de dictado cabe en 1 ventana de 30s
params.set_max_len(60);               // limitar tokens de output (anti-loop)
params.set_temperature(0.0);          // greedy determinista, menos alucinación
params.set_temperature_inc(0.0);      // sin fallback de temperatura
params.set_token_timestamps(false);   // no necesitamos timestamps
```

**Justificación por parámetro:**
- `set_n_threads`: control explícito del paralelismo de cómputo, ajustado al hardware.
- `set_single_segment`: evita que whisper divida el audio en múltiples ventanas de 30s. El audio de dictado siempre es <30s, así que forzar 1 segmento elimina overhead de segmentación y mejora latencia.
- `set_no_context`: whisper.cpp por default pasa el resultado anterior como contexto. En dictado puntual, esto causa que frases se repitan entre dictados. `true` lo desactiva.
- `set_max_len(60)`: previene loops de repetición ("gracias gracias gracias...") que a veces ocurren con silencio.
- `set_temperature(0.0)`: greedy puro, determinista, más rápido.
- `set_token_timestamps(false)`: cálculo de timestamps tiene overhead y no lo usamos.

#### 1b. GPU acceleration en `whisper_cpp.rs`

Añadir campo `gpu_config` a `WhisperCpp` y usarlo en `load_model`:

```rust
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub use_gpu: bool,
    pub n_gpu_layers: i32,   // 99 = offload total
    pub flash_attn: bool,    // requiere build con GPU
}

impl Default for GpuConfig {
    fn default() -> Self {
        // Detección automática: si la feature de GPU está compilada, activar.
        #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
        {
            Self { use_gpu: true, n_gpu_layers: 99, flash_attn: true }
        }
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
        {
            Self { use_gpu: false, n_gpu_layers: 0, flash_attn: false }
        }
    }
}
```

En `load_model`:
```rust
let ctx_params = WhisperContextParameters {
    use_gpu: self.gpu_config.use_gpu,
    n_gpu_layers: self.gpu_config.n_gpu_layers,
    flash_attn: self.gpu_config.flash_attn,
    ..Default::default()
};
```

#### 1c. Feature flags en `Cargo.toml`

```toml
[features]
cuda = ["whisper-rs/cuda"]
metal = ["whisper-rs/metal"]
vulkan = ["whisper-rs/vulkan"]
```

Propagar features desde el bin `oido` → `oido-core` → `oido-stt` con `optional-dependencies` + `feature` forwarding para que `cargo build --features cuda` funcione desde la raíz del workspace.

#### 1d. Método `warm_up` en el trait `Transcriber`

Añadir al trait `Transcriber` (en `lib.rs`):
```rust
fn warm_up(&self) -> Result<(), SttError>;
```

Implementación en `whisper_cpp.rs`: transcribe 1 segundo de silencio sintético (16000 muestras de 0.0f32) para forzar la carga de pesos a memoria/GPU en el arranque, no en el primer dictado del usuario.

#### 1e. Tests actualizados

- Actualizar `tests/pipeline_e2e.rs`: los mocks deben implementar el nuevo método `warm_up` del trait.
- Actualizar test `short_audio_returns_audio_too_short`: ahora el threshold puede ser 1600 (con VAD interno) — ajustar para reflejar el comportamiento con `set_single_segment`.

---

### 2. `oido-core` — Pipeline con thread worker dedicado

#### 2a. Refactor: mover STT fuera del callback `on_release`

**Cambios en `pipeline.rs`:**

1. **Nuevo canal de transcripción** (bounded 1, según PLAN.md "bounded 8 for text" pero como es hold-to-talk serializado, 1 es óptimo):
```rust
let (stt_tx, stt_rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
```

2. **Nuevo thread worker `"oido-stt"`**: drena `stt_rx`, transcribe, aplica `phrase_filter`, inyecta.
   - Posee: `Arc<dyn Transcriber>`, `Arc<dyn Injector>`, clon del `event_tx`.
   - Bucle: `while let Ok(buffer) = stt_rx.recv() { ... transcribe → filter → inject ... }`
   - Registra spans de `tracing` por etapa (`#[tracing::instrument]` o spans explícitos).

3. **`on_release` simplificado**: snapshot del buffer → envía por `stt_tx` → vuelve inmediatamente:
```rust
let on_release = Box::new(move || {
    let buffer = {
        let mut s = recording_r.lock();
        s.recording = false;
        std::mem::take(&mut s.samples)
    };
    if buffer.is_empty() {
        let _ = event_tx.send(PipelineEvent::State(PipelineState::Idle));
        return;
    }
    let _ = event_tx.send(PipelineEvent::State(PipelineState::Processing));
    let _ = stt_tx.send(buffer);  // no bloquea si el worker está libre
});
```

4. **`Pipeline::start()` actualizado**: spawn del worker thread `"oido-stt"` además del consumer thread `"oido-audio"`.

5. **`Pipeline::shutdown()` actualizado**: añadir `self.stt_tx` a un `Option` para poder dropearlo y causar que el worker salga del `recv()`. El `capture.stop()` ya dropea el `audio_tx` de cpal.

**Beneficio**: el callback `on_release` ahora termina en microsegundos. El hotkey queda libre inmediatamente para aceptar un nuevo dictado (se encola en `stt_rx` bounded 1).

**Conformidad R1**: Esto **alinea** el pipeline con la regla R1 (channels-only). El estado actual con STT inline en el callback era la excepción documentada; este refactor la elimina.

#### 2b. Nuevos estados del pipeline (opcional, para feedback UI)

Añadir `PipelineState::Error` para distinguir fallos de STT/injection del Idle normal:
```rust
pub enum PipelineState {
    Idle,
    Recording,
    Processing,
    Error,  // NUEVO: STT o injection falló, vuelve a Idle tras feedback
}
```

El worker emite `Error` antes de volver a `Idle` cuando `transcribe` o `inject` fallan, para que el tray/UI pueda mostrar feedback.

---

### 3. `oido-config` — Schema extendido

#### 3a. Campos nuevos en `Config`

```rust
pub struct Config {
    pub hotkey: String,        // existente
    pub model: String,         // existente
    pub language_ui: String,   // existente
    // NUEVOS:
    pub use_gpu: bool,         // default: true si build con GPU feature
    pub n_threads: Option<u32>, // None = autodetectar (min(cores, 8))
}
```

#### 3b. Proptest `Arbitrary` actualizado para incluir los nuevos campos.

#### 3c. Default actualizado: `use_gpu` depende de features compiladas.

---

### 4. `oido-platform` — Fix de sample rate

#### 4a. Resampler en `capture.rs`

Añadir dependencia `rubato` (mencionado en el TODO existente en `capture.rs:5`):

```rust
// En CpalCapture, si sample_rate != 16000:
// Configurar un resampler que convierte al vuelo cada AudioFrame.
```

Alternativa más simple y liviana (sin dependencia nueva): **decimación lineal por polinomio**. Para 48kHz → 16kHz es factor 3x (decimación limpia), para 44.1kHz → 16kHz es ratio no-enteoro (requiere `rubato`).

**Decisión**: usar `rubato` (es el plan documentado en Fase 2, pero el bug es crítico en Fase 1 — transcribir a sample rate incorrecto produce output basura). Se trae a Fase 1.

El resampler vive dentro de `CpalCapture` (antes de enviar el `AudioFrame` al canal, ya viene a 16kHz). Esto mantiene el contrato: todo lo que llega por el canal es 16kHz mono F32.

#### 4b. Cambio en el branch de captura

```rust
cpal::SampleFormat::F32 => self.device.build_input_stream(
    self.stream_config,
    move |data: &[f32], _cb| {
        let resampled = resampler.process(data); // identity si ya es 16kHz
        let _ = sink.send(AudioFrame {
            samples: resampled,
            sample_rate_hz: 16_000,  // SIEMPRE 16kHz al salir
        });
    },
    ...
)
```

---

### 5. Bin `oido` — Orquestación con warm-up

#### 5a. Warm-up al arranque

Después de `stt.load_model()`, llamar `stt.warm_up()`:
```rust
let mut stt = WhisperCpp::with_language(&snap.language_ui, gpu_config);
stt.load_model(&model_path)?;
stt.warm_up()?;  // NUEVO: 1 inferencia de silencio para cargar pesos
```

Esto elimina el cold-start del primer dictado del usuario.

#### 5b. Logging de performance

Añadir spans de `tracing` con medición de tiempo en el worker de STT:
```rust
let span = tracing::info_span!("stt_inference", samples = buffer.len());
let _enter = span.enter();
let start = std::time::Instant::now();
let result = transcriber.transcribe(&buffer);
tracing::info!(
    latency_ms = start.elapsed().as_millis(),
    samples = buffer.len(),
    duration_s = buffer.len() / 16_000,
    "stt completado"
);
```

Esto permite medir si se cumple el DoD de Fase 1 (latencia <3s para frase corta) y detectar regresiones.

---

## Flujo de datos optimizado (resultado final)

```
[global-hotkey F8]
    │
    ├─ on_press ──► lock buffer ──► recording=true, clear samples
    │                                      │
    │                                      ▼
    │                              [cpal CaptureSource]
    │                                      │ (16kHz mono F32, resampled si hace falta)
    │                          crossbeam bounded(1024)
    │                                      │
    │                                      ▼
    │                              [thread "oido-audio"]
    │                               append a buffer si recording
    │
    └─ on_release ──► snapshot+clear buffer ──► stt_tx.send(buffer) ──► return inmediato
                                                            │
                                                crossbeam bounded(1)
                                                            │
                                                            ▼
                                                    [thread "oido-stt"]
                                                     transcribe (GPU/CPU tuned)
                                                     phrase_filter
                                                     injector.inject (Ctrl/Cmd+V)
                                                            │
                                                            ▼
                                                    event_tx: Processing → Idle
```

**Tiempos estimados** (modelo base, frase de 3 palabras ~1s de audio):
- CPU sin optimizar: ~2-4s
- CPU optimizado (threads + single_segment + no_context): ~1-2s
- GPU (CUDA/Metal): ~0.3-0.8s

---

## Orden de implementación (prioridad por impacto)

1. **C1 + 2a**: Thread worker dedicado (elimina bloqueo del hotkey). Mayor cambio arquitectónico, toca 7 tests.
2. **C3 + 1a**: Tuning de parámetros de whisper.cpp (threads, single_segment, etc.). Sin tocar arquitectura, gran impacto en latencia.
3. **C6**: VAD de silencio (trim inicial/final del buffer antes de transcribir). Reduce samples innecesarios.
4. **C5 + 4a**: Resampler (corrige bug crítico en hardware común).
5. **C4 + 5a**: Warm-up (elimina cold-start del primer uso).
6. **C7 + 2b**: Nuevos estados + logging de performance.
7. **C2 + 1b/1c + 3a**: GPU acceleration (feature flags + config). Infraestructura para el mayor salto de rendimiento.

---

## Tests

- **Actualización de `tests/pipeline_e2e.rs`**: los 7 tests existentes se adaptan al nuevo flujo con canal STT. El mock `MockTranscriber` implementa `warm_up`. Se añade un test que verifica que `on_release` no bloquea (el worker procesa asíncronamente).
- **Tests unitarios en `whisper_cpp.rs`**: nuevo test `gpu_config_default_reflects_features` (verifica que el default cambia según features compiladas).
- **Tests en `capture.rs`**: test de resampling (48kHz → 16kHz produce output de longitud correcta).
- **Smoke test manual**: `cargo run` con `--features cuda` o `--features metal` en hardware con GPU, verificar latencia.

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| Feature flags de GPU rompen build CPU | Features son opt-in (`#[cfg(feature=...)]`); build sin features = comportamiento actual. CI corre build sin GPU. |
| `rubato` añade dependencia | Es estándar en ecosistema Rust audio, sin advisories, licencia MIT. Ya estaba planificado. |
| Refactor del pipeline rompe tests | Se actualizan los 7 tests E2E en el mismo cambio. Los mocks ya están diseñados para esto. |
| `set_single_segment` con audio >30s | Audio de dictado hold-to-talk rara vez supera 30s. Si pasa, whisper trunca — aceptable. Documentar límite. |
| GPU no disponible en runtime | `GpuConfig::default()` detecta features compiladas. Si `use_gpu=true` pero no hay GPU, whisper.cpp cae a CPU automáticamente con warning. |

---

## Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `crates/oido-stt/src/lib.rs` | Trait `Transcriber` + método `warm_up` |
| `crates/oido-stt/src/whisper_cpp.rs` | Parámetros optimizados, `GpuConfig`, `warm_up`, warm-up |
| `crates/oido-stt/Cargo.toml` | Feature flags cuda/metal/vulkan |
| `crates/oido-core/src/pipeline.rs` | Thread worker STT, canal `stt_tx/rx`, `on_release` simplificado |
| `crates/oido-core/tests/pipeline_e2e.rs` | Actualizar 7 tests + nuevo test de no-bloqueo |
| `crates/oido-config/src/lib.rs` | Campos `use_gpu`, `n_threads` en `Config` |
| `crates/oido-platform/src/capture.rs` | Resampler `rubato` |
| `crates/oido-platform/Cargo.toml` | Dep `workspace) `rubato` |
| `crates/oido/src/main.rs` | `warm_up()` al arranque, logging de perf |
| `Cargo.toml` (workspace) | `rubato` en `[workspace.dependencies]` |
| `Cargo.toml` (workspace) | Features forwarding `cuda`/`metal`/`vulkan` |