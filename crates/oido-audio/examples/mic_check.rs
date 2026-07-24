//! Diagnóstico de micrófono: lista dispositivos de entrada y captura
//! 3 segundos del dispositivo default, reportando nivel RMS en dBFS.
//!
//! Interpretación:
//! - rms = -inf / near_zero = 1.00 → silencio digital exacto: mic
//!   muteado, permiso denegado, o dispositivo equivocado (ej. Stereo
//!   Mix sin señal).
//! - rms < -60 dBFS → señal extremadamente débil (mic lejos o gain 0).
//! - rms -50..-35 dBFS en ambiente → mic OK (ruido de sala presente).
//!
//! Uso: cargo run --release -p oido-audio --example mic_check

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn rms_dbfs(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (f32::NEG_INFINITY, 1.0);
    }
    let mut sum = 0.0f64;
    let mut zeros = 0usize;
    for &s in samples {
        sum += f64::from(s) * f64::from(s);
        if s.abs() < 1e-4 {
            zeros += 1;
        }
    }
    let rms = (sum / samples.len() as f64).sqrt() as f32;
    let db = if rms <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    };
    (db, zeros as f32 / samples.len() as f32)
}

fn device_name(d: &cpal::Device) -> String {
    d.description()
        .map(|x| x.name().to_owned())
        .unwrap_or_default()
}

fn main() {
    let host = cpal::default_host();
    let default = host.default_input_device();
    let default_name = default.as_ref().map(device_name).unwrap_or_default();

    println!("== dispositivos de entrada ==");
    for d in host.input_devices().expect("listar input devices") {
        let name = device_name(&d);
        let mark = if name == default_name {
            "  <-- DEFAULT"
        } else {
            ""
        };
        println!("  {name}{mark}");
    }

    let Some(dev) = default else {
        println!("\nSIN dispositivo de entrada default. Fin.");
        return;
    };
    let cfg = dev.default_input_config().expect("default_input_config");
    println!(
        "\ncapturando 3s desde '{}' ({:?}, {} ch, {} Hz)...",
        default_name,
        cfg.sample_format(),
        cfg.channels(),
        cfg.sample_rate()
    );

    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let channels = cfg.channels() as usize;
    let stream_config = cfg.config();

    macro_rules! build {
        ($t:ty, $conv:expr) => {{
            let buf = Arc::clone(&buf);
            dev.build_input_stream(
                stream_config.clone(),
                move |data: &[$t], _| {
                    let conv: fn($t) -> f32 = $conv;
                    let mut g = buf.lock().unwrap();
                    for frame in data.chunks(channels.max(1)) {
                        let s: f32 =
                            frame.iter().map(|&x| conv(x)).sum::<f32>() / frame.len() as f32;
                        g.push(s);
                    }
                },
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("build_input_stream")
        }};
    }

    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => build!(f32, |x| x),
        cpal::SampleFormat::I16 => build!(i16, |x| f32::from(x) / 32_768.0),
        cpal::SampleFormat::U16 => build!(u16, |x| (f32::from(x) - 32_768.0) / 32_768.0),
        f => {
            println!("formato no soportado: {f:?}");
            return;
        }
    };
    stream.play().expect("play");
    std::thread::sleep(Duration::from_secs(3));
    drop(stream);

    let samples = buf.lock().unwrap();
    println!("\ncapturados {} samples", samples.len());
    // RMS global y por bloques de 0.5s para ver variación.
    let (db, zeros) = rms_dbfs(&samples);
    println!("RMS global: {db:.1} dBFS | near_zero_frac: {zeros:.3}");
    let block = cfg.sample_rate() as usize / 2;
    for (i, chunk) in samples.chunks(block).enumerate() {
        let (b, _) = rms_dbfs(chunk);
        println!("  bloque {i}: {b:.1} dBFS");
    }
}
