//! Bench aislado de throughput STT: mide `transcribe` sobre audio
//! sintético con el modelo y preset reales del usuario (small, CPU,
//! Balanced). Sirve para separar "la app se puso lenta" de "la máquina
//! está lenta ahora" (térmico, contención, power plan).
//!
//! Uso:
//!   cargo run --release -p oido-stt --example stt_bench -- [model_path]
//!
//! Default model: %APPDATA%\oido\models\ggml-small.bin

use std::time::Instant;

use oido_stt::{GpuConfig, Transcriber, WhisperCpp};

fn synth(secs: usize) -> Vec<f32> {
    // Ruido modulado tipo "habla": envolvente de ráfagas sobre una mezcla
    // de senos. El contenido no importa para medir LATENCIA (el encoder
    // cuesta fijo por duración); importa que el decoder tenga trabajo.
    let n = secs * 16_000;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let env = ((t * 2.5).sin().max(0.0)) * 0.3; // ráfagas ~2.5 Hz
            env * (0.5 * (t * 220.0 * std::f32::consts::TAU).sin()
                + 0.3 * (t * 440.0 * std::f32::consts::TAU).sin()
                + 0.2 * (t * 880.0 * std::f32::consts::TAU).sin())
        })
        .collect()
}

fn main() {
    let model = std::env::args().nth(1).unwrap_or_else(|| {
        let appdata = std::env::var("APPDATA").expect("APPDATA");
        format!(r"{appdata}\oido\models\ggml-small.bin")
    });
    // argv[2] opcional: n_threads (default = detect, cap 8). La app del
    // usuario corre con 4; probar 4 vs 8 discrimina si los E-cores del
    // CPU híbrido castigan el caso multi-thread.
    let n_threads: u16 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    println!("modelo: {model}");

    let mut stt = WhisperCpp::with_language("es");
    if n_threads > 0 {
        stt = stt.with_runtime(GpuConfig::default(), n_threads);
    }
    let t0 = Instant::now();
    stt.load_model(std::path::Path::new(&model)).expect("load_model");
    println!("load_model: {:?}", t0.elapsed());

    let t0 = Instant::now();
    stt.warm_up().expect("warm_up");
    println!("warm_up:    {:?}", t0.elapsed());

    for secs in [5usize, 10] {
        let audio = synth(secs);
        for run in 1..=3 {
            let t0 = Instant::now();
            let text = stt.transcribe(&audio).expect("transcribe");
            let ms = t0.elapsed().as_millis();
            println!(
                "transcribe {secs}s run{run}: {ms} ms  (rtf={:.2}) text={:?}",
                ms as f64 / (secs as f64 * 1000.0),
                &text.chars().take(60).collect::<String>()
            );
        }
    }
}
