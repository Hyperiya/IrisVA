use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, Stream, StreamConfig};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use vosk::{DecodingState, Model, Recognizer};

const DEFAULT_WAKE: &[&str] = &["hey iris"];

#[derive(Clone)]
enum ListeningState {
    Idle,
    WakeDetected { time: Instant },
}

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
    (has_am && has_graph) || (has_conf && (has_am || has_graph))
}

fn resolve_model_dir(args: &[(String, String)]) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("VOSK_MODEL") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(arg1) = env::args().skip(1).next() {
        candidates.push(PathBuf::from(arg1));
    }
    if let Some((_, model_path)) = args.iter().find(|(key, _)| key == "--model") {
        candidates.push(PathBuf::from(model_path));
    }


    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
    candidates.push(manifest_dir.join("src").join("model"));
    candidates.push(manifest_dir.join("model"));


    let mut expanded: Vec<PathBuf> = Vec::new();
    for cand in candidates {
        if cand.is_file() {
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
            for sd in &subdirs {
                if looks_like_vosk_model_dir(sd) {
                    return Ok(sd.clone());
                }
            }
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

    let mut msg = String::from("Failed to locate a valid Vosk acoustic model directory.\n");
    msg.push_str(
        "Tried the following locations (env VOSK_MODEL, CLI arg, ./src/model, ./model, supplied path):\n",
    );
    for p in expanded {
        msg.push_str(&format!(" - {}\n", p.display()));
        if p.is_dir() {
            if let Ok(rd) = fs::read_dir(&p) {
                let mut has_lib = false;
                let mut has_header = false;
                for e in rd.flatten() {
                    let name = e.file_name();
                    let name = name.to_string_lossy().to_string();
                    if name.contains("libvosk")
                        || name.ends_with("libvosk.dll")
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

fn match_input_device(host: &Host, device_name: &str) -> Option<Device> {
    for device in host.input_devices().unwrap() {
        if device.name().unwrap() == device_name {
            return Some(device);
        }
    }
    None
}

fn collect_launch_args() -> Option<Vec<(String, String)>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut pairs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") && i + 1 < args.len() {
            let key = args[i].clone();
            let mut value = args[i + 1].clone();
            i += 2;
            // Collect remaining non-flag args as part of this value
            while i < args.len() && !args[i].starts_with("--") {
                value.push(' ');
                value.push_str(&args[i]);
                i += 1;
            }
            pairs.push((key, value));
        } else {
            i += 1;
        }
    }
    Some(pairs)
}
fn main() {
    if let Ok(lib_dir) = env::var("VOSK_LIB_DIR") {
        let paths = env::var_os("LD_LIBRARY_PATH")
            .map(PathBuf::from)
            .unwrap_or_default();
        let new = if paths.as_os_str().is_empty() {
            PathBuf::from(lib_dir)
        } else {
            let p = PathBuf::from(env::var("LD_LIBRARY_PATH").unwrap_or_default());
            let combined = format!("{}:{}", lib_dir, p.display());
            PathBuf::from(combined)
        };
        unsafe {
            env::set_var("LD_LIBRARY_PATH", new);
        }
    }

    let args = collect_launch_args().unwrap_or_default();
    let model_dir = match resolve_model_dir(&args) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("{}\n[ERR]", msg);
            std::process::exit(2);
        }
    };

    let model_path_str: String = model_dir.to_string_lossy().into_owned();
    let model = match Model::new(&model_path_str) {
        Some(m) => m,
        None => {
            eprintln!(
                "Failed to load Vosk model at '{}'.\n- Ensure you have extracted a Vosk acoustic model directory there (not just libvosk.so).\n- You can set env VOSK_MODEL=/path/to/model or pass it as the first CLI arg.\n- Place libvosk.so somewhere in your loader path or set VOSK_LIB_DIR. [ERR]",
                model_path_str
            );
            std::process::exit(2);
        }
    };


    let host = cpal::default_host();
    println!("Available input devices:");
    for device in host.input_devices().unwrap() {
        println!("Input device: {:?}", device.name());
    }
    // println!("{:?}", args);
    let selected_device = args.iter().find(|(key, _)| key == "--device");

    let device = if let Some((_, value)) = selected_device {
        match_input_device(&host, value).unwrap_or_else(|| host.default_input_device().expect("No default input device available[ERR]"))
    } else {
        host.default_input_device().expect("No default input device available")
    };

    println!("Using input device: {device:?}\n[DEVICE]({device:?})", device=device.name());

    let supported_config = match device.default_input_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Failed to get default input config: {:?}[ERR]", e);
            std::process::exit(3);
        }
    };

    let mut config: StreamConfig = supported_config.clone().into();

    if config.channels == 0 {
        config.channels = 1;
    }

    let sample_rate_hz = config.sample_rate.0 as f32;

    let mut recognizer1 =
        Recognizer::new(&model, sample_rate_hz).expect("Failed to create recognizer");
    let mut recognizer2 =
        Recognizer::new(&model, sample_rate_hz).expect("Failed to create recognizer");

    for rec in [&mut recognizer1, &mut recognizer2] {
        let _ = rec.set_max_alternatives(0);
        let _ = rec.set_words(false);
        let _ = rec.set_partial_words(false);
        let _ = rec.set_nlsml(false);
    }

    let active_recognizer = Arc::new(Mutex::new(0u8)); // 0 or 1
    let recognizers = Arc::new(Mutex::new([recognizer1, recognizer2]));

    let recognizers_clone = recognizers.clone();
    let active_clone = active_recognizer.clone();
    // Replace the swap thread with this version:
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(600)); // 10 seconds for testing, 600 seconds in prod

            let active = *active_clone.lock().unwrap();
            let inactive = 1 - active;

            // Recreate inactive recognizer using existing model
            {
                let mut recs = recognizers_clone.lock().unwrap();
                // Drop old recognizer explicitly
                drop(std::mem::replace(
                    &mut recs[inactive as usize],
                    Recognizer::new(&model, sample_rate_hz).unwrap(),
                ));

                let _ = recs[inactive as usize].set_max_alternatives(0);
                let _ = recs[inactive as usize].set_words(false);
                let _ = recs[inactive as usize].set_partial_words(false);
                let _ = recs[inactive as usize].set_nlsml(false);
            } // Release lock before swap

            *active_clone.lock().unwrap() = inactive;
            println!("Swapped to fresh recognizer\n[SWAP]");
        }
    });

    println!(
        "Listening for wake words: {} (sample rate: {} Hz, channels: {}) [LISTENING]",
        DEFAULT_WAKE.join(", "),
        sample_rate_hz,
        config.channels
    );

    let triggered = Arc::new(Mutex::new(false));
    let triggered_clone = triggered.clone();
    let state = Arc::new(Mutex::new(ListeningState::Idle));
    let state_clone = state.clone();

    let err_flag = Arc::new(Mutex::new(None::<String>));
    let err_flag_clone = err_flag.clone();

    let stream: Stream = match supported_config.sample_format() {
        SampleFormat::I16 => build_input_stream_i16(
            &device,
            &config,
            recognizers.clone(),
            active_recognizer.clone(),
            triggered_clone,
            DEFAULT_WAKE,
            state_clone,
            err_flag_clone,
        ),
        SampleFormat::U16 => build_input_stream_u16(
            &device,
            &config,
            recognizers.clone(),
            active_recognizer.clone(),
            triggered_clone,
            DEFAULT_WAKE,
            state_clone,
            err_flag_clone,
        ),
        SampleFormat::F32 => build_input_stream_f32(
            &device,
            &config,
            recognizers.clone(),
            active_recognizer.clone(),
            triggered_clone,
            DEFAULT_WAKE,
            state_clone,
            err_flag_clone,
        ),
        _ => panic!("Unsupported sample format"),
    };

    stream.play().expect("Failed to start input stream");

    let start = Instant::now();
    let mut listening_printed = false;
    loop {
        if let Some(err) = err_flag.lock().unwrap().take() {
            eprintln!("Stream error: {}\\n[ERR]", err);
            *triggered.lock().unwrap() = false;
            *state.lock().unwrap() = ListeningState::Idle;
            listening_printed = false;
            continue;
        }

        if *triggered.lock().unwrap() {
            println!("Command processed.\\n[PROCESSED]");
            *triggered.lock().unwrap() = false;
            *state.lock().unwrap() = ListeningState::Idle;
            listening_printed = false;
            std::thread::sleep(Duration::from_millis(500)); // Prevent immediate retrigger
            continue;
        }

        if let Ok(mut current_state_guard) = state.lock() {
            if let ListeningState::WakeDetected { time } = &*current_state_guard {
                let elapsed = time.elapsed();
                if elapsed > Duration::from_millis(350) {
                    if !listening_printed {
                        println!("Listening for command...\\n[WAITING]");
                        listening_printed = true;
                    }
                    if elapsed > Duration::from_secs(3) {
                        println!("No command detected. Resetting.[RESETTING]");
                        *current_state_guard = ListeningState::Idle; // Modify directly
                        listening_printed = false;

                        // Reset recognizer
                        let active_idx = *active_recognizer.lock().unwrap() as usize;
                        let mut recs = recognizers.lock().unwrap();
                        let _ = recs[active_idx].reset();

                        continue;
                    }
                }
            } else {
                listening_printed = false;
            }
        }


        std::thread::sleep(Duration::from_millis(50));
        if start.elapsed() > Duration::from_secs(24 * 60 * 60) {
            break;
        }
    }
}

fn build_input_stream_i16(
    device: &Device,
    config: &StreamConfig,
    recognizers: Arc<Mutex<[Recognizer; 2]>>,
    active_recognizer: Arc<Mutex<u8>>,
    triggered: Arc<Mutex<bool>>,
    wake_words: &'static [&'static str],
    state: Arc<Mutex<ListeningState>>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> Stream {
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

        let active_idx = *active_recognizer.lock().unwrap() as usize;
        let mut recs = recognizers.lock().unwrap();
        create_waveform_match(
            &mut recs[active_idx],
            &pcm_mono,
            &wake_words,
            &triggered,
            &state,
        );
    };

    let err_fn = move |err: cpal::StreamError| {
        if let Ok(mut e) = err_flag.lock() {
            *e = Some(format!("CPAL stream error: {}[ERR]", err));
        }
    };

    device
        .build_input_stream(config, data_fn, err_fn, None)
        .expect("Failed to build input stream[ERR]")
}

fn build_input_stream_u16(
    device: &Device,
    config: &StreamConfig,
    recognizers: Arc<Mutex<[Recognizer; 2]>>,
    active_recognizer: Arc<Mutex<u8>>,
    triggered: Arc<Mutex<bool>>,
    wake_words: &'static [&'static str],
    state: Arc<Mutex<ListeningState>>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> Stream {
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
                    acc += s as i32 - 32768;
                }
                let avg = (acc / channels as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                pcm_mono.push(avg);
            }
        }

        let active_idx = *active_recognizer.lock().unwrap() as usize;
        let mut recs = recognizers.lock().unwrap();
        create_waveform_match(
            &mut recs[active_idx],
            &pcm_mono,
            &wake_words,
            &triggered,
            &state,
        );
    };

    let err_fn = move |err: cpal::StreamError| {
        if let Ok(mut e) = err_flag.lock() {
            *e = Some(format!("CPAL stream error: {}[ERR]", err));
        }
    };

    device
        .build_input_stream(config, data_fn, err_fn, None)
        .expect("Failed to build input stream[ERR]")
}

fn build_input_stream_f32(
    device: &Device,
    config: &StreamConfig,
    recognizers: Arc<Mutex<[Recognizer; 2]>>,
    active_recognizer: Arc<Mutex<u8>>,
    triggered: Arc<Mutex<bool>>,
    wake_words: &'static [&'static str],
    state: Arc<Mutex<ListeningState>>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> Stream {
    let channels = config.channels as usize;

    let data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut pcm_mono: Vec<i16> = Vec::with_capacity(4096usize);
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

        let active_idx = *active_recognizer.lock().unwrap() as usize;
        let mut recs = recognizers.lock().unwrap();
        create_waveform_match(
            &mut recs[active_idx],
            &pcm_mono,
            &wake_words,
            &triggered,
            &state,
        );
    };

    let err_fn = move |err: cpal::StreamError| {
        if let Ok(mut e) = err_flag.lock() {
            *e = Some(format!("CPAL stream error: {}[ERR]", err));
        }
    };

    device
        .build_input_stream(config, data_fn, err_fn, None)
        .expect("Failed to build input stream[ERR]")
}

fn extract_text_from_complete_json(result_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(result_json).ok()?;
    v.get("text")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

fn contains_wake_word(text: &str, wake_words: &[&str]) -> Option<String> {
    let t = text.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }

    for wake_word in wake_words {
        if let Some(pos) = t.find(wake_word) {
            let after_wake = &t[pos + wake_word.len()..].trim();
            return if !after_wake.is_empty() {
                Some(format!("{} {}", wake_word, after_wake))
            } else {
                Some(wake_word.to_string())
            };
        }
    }
    None
}

fn is_just_wake_word(text: &str, wake_words: &[&str]) -> bool {
    let t = text.trim().to_lowercase();
    wake_words.iter().any(|w| t == *w)
}

fn create_waveform_match(
    recognizer: &mut Recognizer,
    pcm_mono: &[i16],
    wake_words: &[&str],
    triggered: &Arc<Mutex<bool>>,
    state: &Arc<Mutex<ListeningState>>,
) {
    match recognizer.accept_waveform(&pcm_mono) {
        Ok(DecodingState::Running) => {
            // Force periodic cleanup every ~1000 calls
            static mut CALL_COUNT: u32 = 0;
            unsafe {
                CALL_COUNT += 1;
                if CALL_COUNT % 1000 == 0 {
                    let _ = recognizer.reset();
                }
            }
        }
        Ok(_) => {
            let complete = recognizer.result();
            if let Ok(json) = serde_json::to_string(&complete) {
                if let Some(text) = extract_text_from_complete_json(&json) {
                    let current_state = state.lock().unwrap().clone();

                    match current_state {
                        ListeningState::Idle => {
                            if let Some(full_command) = contains_wake_word(&text, &wake_words) {
                                if is_just_wake_word(&text, &wake_words) {
                                    // Just wake word detected, start pause timer
                                    *state.lock().unwrap() = ListeningState::WakeDetected {
                                        time: Instant::now(),
                                    };
                                } else {
                                    // Full command in one go
                                    println!("Full command: {command}\n[COMMAND](hey iris {command})", command=full_command);
                                    if let Ok(mut t) = triggered.lock() {
                                        *t = true;
                                    }
                                }
                            }
                        }
                        ListeningState::WakeDetected { .. } => {
                            // Any speech after wake word is treated as command
                            if !text.trim().is_empty() {
                                println!("Full command: hey iris {command}\\n[COMMAND]({command})", command=text.trim());
                                if let Ok(mut t) = triggered.lock() {
                                    *t = true;
                                }
                                *state.lock().unwrap() = ListeningState::Idle;

                                // Reset triggered after a short delay
                                // std::thread::spawn({
                                //     let triggered = triggered.clone();
                                //     move || {
                                //         std::thread::sleep(Duration::from_millis(100));
                                //         *triggered.lock().unwrap() = false;
                                //     }
                                // });
                            }
                        }

                    }
                }
            }
            let _ = recognizer.reset();
        }
        Err(_) => {}
    }
}
