use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ SampleFormat, StreamConfig};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use vosk::{ DecodingState, Model, Recognizer};

const DEFAULT_WAKE: &[&str] = &["hey iris"];

fn looks_like_vosk_model_dir(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let entries = match fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return false,
    };
    let mut has_conf = false;
    let mut has_am = false;
    let mut has_graph = false;
    for e in entries.flatten() {
        if let Ok(ft) = e.file_type() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if ft.is_dir() {
                if name == "am" {
                    has_am = true;
                }
                if name == "graph" {
                    has_graph = true;
                }
                if name == "conf" {
                    has_conf = true;
                }
            } else if ft.is_file() {
                if name == "model.conf" || name.ends_with(".conf") {
                    has_conf = true;
                }
            }
        }
    }
    // Many Vosk models have am+graph+conf; some have conf and other assets. Be permissive but not too loose.
    (has_am && has_graph) || (has_conf && (has_am || has_graph))
}

fn resolve_model_dir() -> Result<PathBuf, String> {
    // Priority: VOSK_MODEL env -> first CLI arg -> ./src/model -> ./model
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("VOSK_MODEL") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(arg1) = env::args().skip(1).next() {
        candidates.push(PathBuf::from(arg1));
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
    candidates.push(manifest_dir.join("src").join("model"));
    candidates.push(manifest_dir.join("model"));

    // Expand candidates: if a candidate is a directory that contains exactly one subdirectory and
    // itself doesn't look like a model, try that subdirectory.
    let mut expanded: Vec<PathBuf> = Vec::new();
    for cand in candidates {
        if cand.is_file() {
            // Probably a tar.gz or wrong path; skip but note later
            expanded.push(cand);
            continue;
        }
        if looks_like_vosk_model_dir(&cand) {
            return Ok(cand);
        }
        if cand.is_dir() {
            let mut subdirs: Vec<PathBuf> = Vec::new();
            if let Ok(rd) = fs::read_dir(&cand) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        subdirs.push(p);
                    }
                }
            }
            // Prefer a subdir that looks like a model
            for sd in &subdirs {
                if looks_like_vosk_model_dir(sd) {
                    return Ok(sd.clone());
                }
            }
            // If only one subdir, try it anyway
            if subdirs.len() == 1 {
                let sd = &subdirs[0];
                if looks_like_vosk_model_dir(sd) {
                    return Ok(sd.clone());
                }
            }
            expanded.push(cand);
        } else {
            expanded.push(cand);
        }
    }

    // Build an informative error message
    let mut msg = String::from("Failed to locate a valid Vosk acoustic model directory.\n");
    msg.push_str(
        "Tried the following locations (env VOSK_MODEL, CLI arg, ./src/model, ./model):\n",
    );
    for p in expanded {
        msg.push_str(&format!(" - {}\n", p.display()));
        if p.is_dir() {
            // Check if it looks like a lib folder (contains libvosk or vosk_api.h)
            if let Ok(rd) = fs::read_dir(&p) {
                let mut has_lib = false;
                let mut has_header = false;
                for e in rd.flatten() {
                    let name = e.file_name();
                    let name = name.to_string_lossy().to_string();
                    if name.contains("libvosk")
                        || name.ends_with("vosk.dll")
                        || name.ends_with("libvosk.dylib")
                        || name.ends_with("libvosk.so")
                    {
                        has_lib = true;
                    }
                    if name == "vosk_api.h" {
                        has_header = true;
                    }
                }
                if has_lib || has_header {
                    msg.push_str("   Found Vosk library/header here, but not an extracted acoustic model directory.\n");
                }
            }
        } else if p.is_file() {
            msg.push_str("   Path is a file, expected a directory (did you provide a .zip/.tar.gz archive?).\n");
        }
    }
    msg.push_str("\nPlease download and extract a Vosk model (e.g., 'vosk-model-small-en-us-0.15') so that the folder contains subfolders like 'am', 'graph', and 'conf'.\n");
    msg.push_str(
        "You can set VOSK_MODEL=/path/to/model_dir or pass it as the first CLI argument.\n",
    );
    Err(msg)
}

fn main() {
    // Allow overriding native lib path so user can point to their libvosk.so location
    if let Ok(lib_dir) = env::var("VOSK_LIB_DIR") {
        let paths = env::var_os("LD_LIBRARY_PATH")
            .map(PathBuf::from)
            .unwrap_or_default();
        let new = if paths.as_os_str().is_empty() {
            PathBuf::from(lib_dir)
        } else {
            let mut p = PathBuf::from(env::var("LD_LIBRARY_PATH").unwrap_or_default());
            // crude append with ':' for unix
            let combined = format!("{}:{}", lib_dir, p.display());
            PathBuf::from(combined)
        };
        unsafe {
            env::set_var("LD_LIBRARY_PATH", new);
        }
    }

    // Resolve model directory robustly
    let model_dir = match resolve_model_dir() {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(2);
        }
    };

    // Wake words: comma-separated in VOSK_WAKE or second arg
    let wake_words: Vec<String> = DEFAULT_WAKE.iter().map(|s| s.to_string()).collect();

    if wake_words.is_empty() {
        eprintln!("No wake words provided. Set VOSK_WAKE env var or pass as second CLI arg.");
        std::process::exit(1);
    }

    // Load Vosk model
    let model_path_str: String = model_dir.to_string_lossy().into_owned();
    let model = match Model::new(&model_path_str) {
        Some(m) => m,
        None => {
            eprintln!(
                "Failed to load Vosk model at '{}'.\n- Ensure you have extracted a Vosk acoustic model directory there (not just libvosk.so).\n- You can set env VOSK_MODEL=/path/to/model or pass it as the first CLI arg.\n- Place libvosk.so somewhere in your loader path or set VOSK_LIB_DIR.",
                model_path_str
            );
            std::process::exit(2);
        }
    };

    // Initialize audio input via cpal
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No default input device available");

    let supported_config = match device.default_input_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Failed to get default input config: {:?}", e);
            std::process::exit(3);
        }
    };

    let mut config: StreamConfig = supported_config.clone().into();

    // We prefer mono; if device supports more channels, we will downmix by averaging
    if config.channels == 0 {
        config.channels = 1;
    }

    let sample_rate_hz = config.sample_rate.0 as f32;

    // Build grammar JSON from wake words to constrain decoder to those phrases
    // Example: ["hey iris","ok iris","computer"]
    let mut grammar = String::from("[");
    for (i, w) in wake_words.iter().enumerate() {
        if i > 0 {
            grammar.push(',');
        }
        grammar.push('"');
        grammar.push_str(&w);
        grammar.push('"');
    }
    grammar.push(']');

    // Recognizer: try with grammar; if not supported, fall back to normal recognizer
    let grammar_vec: Vec<&str> = wake_words.iter().map(|s| s.as_str()).collect();
    let mut recognizer = Recognizer::new_with_grammar(&model, sample_rate_hz, &grammar_vec)
        .or_else(|| Recognizer::new(&model, sample_rate_hz))
        .expect("Failed to create recognizer");

    // Smaller max alternatives improves speed; set to 0 for best; and disable words to reduce latency
    let _ = recognizer.set_max_alternatives(0);
    let _ = recognizer.set_words(false);

    println!(
        "Listening for wake words: {} (sample rate: {} Hz, channels: {})",
        wake_words.join(", "),
        sample_rate_hz,
        config.channels
    );
    println!(
        "Tip: export VOSK_MODEL=/path/to/vosk-model and VOSK_LIB_DIR=/path/to containing libvosk.so if needed."
    );

    let triggered = Arc::new(Mutex::new(false));
    let triggered_clone = triggered.clone();
    let wake_words_clone = wake_words.clone();

    let err_flag = Arc::new(Mutex::new(None::<String>));
    let err_flag_clone = err_flag.clone();

    let mut stream = match supported_config.sample_format() {
        SampleFormat::I16 => (
            build_input_stream_i16(
                &device,
                &config,
                recognizer,
                triggered_clone,
                wake_words_clone,
                err_flag_clone,
            ),
            "I16",
        ),
        SampleFormat::U16 => (
            build_input_stream_u16(
                &device,
                &config,
                recognizer,
                triggered_clone,
                wake_words_clone,
                err_flag_clone,
            ),
            "U16",
        ),
        SampleFormat::F32 => (
            build_input_stream_f32(
                &device,
                &config,
                recognizer,
                triggered_clone,
                wake_words_clone,
                err_flag_clone,
            ),
            "I16",
        ),
        _ => panic!("Unsupported sample format"),
    };

    println!("Using {} input stream", stream.1);

    stream.0.play().expect("Failed to start input stream");

    // Wait until triggered or error
    let start = Instant::now();
    loop {
        if let Some(err) = err_flag.lock().unwrap().take() {
            eprintln!("Stream error: {}", err);
            break;
        }
        if *triggered.lock().unwrap() {
            println!("Wake word detected.\n");
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
        // Safety timeout optional
        if start.elapsed() > Duration::from_secs(24 * 60 * 60) {
            // 24h
            break;
        }
    }
}

fn build_input_stream_i16(
    device: &cpal::Device,
    config: &StreamConfig,
    mut recognizer: Recognizer,
    triggered: Arc<Mutex<bool>>,
    wake_words: Vec<String>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> cpal::Stream {
    let channels = config.channels as usize;
    let data_fn = move |data: &[i16], _: &cpal::InputCallbackInfo| {
        let mut pcm_mono: Vec<i16> = Vec::with_capacity(data.len() / channels + 1);
        if channels <= 1 {
            pcm_mono.extend_from_slice(data);
        } else {
            for frame in data.chunks_exact(channels) {
                let mut acc: i32 = 0;
                for &s in frame.iter() {
                    acc += s as i32;
                }
                let avg = (acc / channels as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                pcm_mono.push(avg);
            }
        }
        create_waveform_match(&mut recognizer, &pcm_mono, &wake_words, &triggered);
    };
    let err_fn = move |err: cpal::StreamError| {
        if let Ok(mut e) = err_flag.lock() {
            *e = Some(format!("CPAL stream error: {}", err));
        }
    };
    device
        .build_input_stream(config, data_fn, err_fn, None)
        .expect("Failed to build input stream")
}

fn build_input_stream_u16(
    device: &cpal::Device,
    config: &StreamConfig,
    mut recognizer: Recognizer,
    triggered: Arc<Mutex<bool>>,
    wake_words: Vec<String>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> cpal::Stream {
    let channels = config.channels as usize;
    let data_fn = move |data: &[u16], _: &cpal::InputCallbackInfo| {
        let mut pcm_mono: Vec<i16> = Vec::with_capacity(data.len() / channels + 1);
        if channels <= 1 {
            for &s in data.iter() {
                let v = (s as i32 - 32768) as i16;
                pcm_mono.push(v);
            }
        } else {
            for frame in data.chunks_exact(channels) {
                let mut acc: i32 = 0;
                for &s in frame.iter() {
                    acc += (s as i32 - 32768);
                }
                let avg = (acc / channels as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                pcm_mono.push(avg);
            }
        }
        create_waveform_match(&mut recognizer, &pcm_mono, &wake_words, &triggered);
    };
    let err_fn = move |err: cpal::StreamError| {
        if let Ok(mut e) = err_flag.lock() {
            *e = Some(format!("CPAL stream error: {}", err));
        }
    };
    device
        .build_input_stream(config, data_fn, err_fn, None)
        .expect("Failed to build input stream")
}

fn build_input_stream_f32(
    device: &cpal::Device,
    config: &StreamConfig,
    mut recognizer: Recognizer,
    triggered: Arc<Mutex<bool>>,
    wake_words: Vec<String>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> cpal::Stream {
    let channels = config.channels as usize;
    let data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut pcm_mono: Vec<i16> = Vec::with_capacity(data.len() / channels + 1);
        if channels <= 1 {
            for &s in data.iter() {
                let v = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                pcm_mono.push(v);
            }
        } else {
            for frame in data.chunks_exact(channels) {
                let mut acc: f32 = 0.0;
                for &s in frame.iter() {
                    acc += s;
                }
                let avgf = acc / channels as f32;
                let v = (avgf * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                pcm_mono.push(v);
            }
        }
        create_waveform_match(&mut recognizer, &pcm_mono, &wake_words, &triggered);
    };
    let err_fn = move |err: cpal::StreamError| {
        if let Ok(mut e) = err_flag.lock() {
            *e = Some(format!("CPAL stream error: {}", err));
        }
    };
    device
        .build_input_stream(config, data_fn, err_fn, None)
        .expect("Failed to build input stream")
}

fn extract_text_from_complete_json(result_json: &str) -> Option<String> {
    // complete JSON looks like: {"text": "hey iris"} or {"text":""}
    let v: serde_json::Value = serde_json::from_str(result_json).ok()?;
    v.get("text")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

fn is_wake_word(text: &str, wake_words: &[String]) -> bool {
    let t = text.trim().to_lowercase();
    if t.is_empty() {
        return false;
    }
    wake_words.iter().any(|w| t == *w)
}

fn create_waveform_match(
    recognizer: &mut Recognizer,
    pcm_mono: &[i16],
    wake_words: &[String],
    triggered: &Arc<Mutex<bool>>,
) {
    match recognizer.accept_waveform(&pcm_mono) {
        Ok(DecodingState::Running) => {
            // Ignore partial results to avoid premature triggering on single words.
        }
        Ok(_) => {
            let complete = recognizer.result();
            if let Ok(json) = serde_json::to_string(&complete) {
                if let Some(text) = extract_text_from_complete_json(&json) {
                    if is_wake_word(&text, &wake_words) {
                        if let Ok(mut t) = triggered.lock() {
                            *t = true;
                        }
                    }
                }
            }
            let _ = recognizer.reset();
        }
        Err(_) => {}
    }
}
