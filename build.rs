use std::env;
use std::path::PathBuf;

fn main() {
    // Determine target OS for filename and rpath behavior
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Candidate locations for the native Vosk library
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = env::var("VOSK_LIB_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    candidates.push(manifest_dir.join("src").join("model"));
    candidates.push(manifest_dir.join("model"));

    // Platform-specific library filename to probe
    let lib_filename = match target_os.as_str() {
        "windows" => "libvosk.dll",
        _ => "libvosk.so", // linux, android, etc.
    };

    // Find a directory that contains the native library
    let mut found_dir: Option<PathBuf> = None;
    // Helper: on Unix, ensure the found lib looks compatible (e.g., 64-bit when targeting x86_64)
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    for dir in candidates {
        let p = dir.join(&lib_filename);
        if p.exists() {
            let mut compatible = true;
            if target_os == "linux" || target_os == "android" {
                if let Ok(bytes) = std::fs::read(&p) {
                    // Minimal ELF check: 0..=3: 0x7F 'E' 'L' 'F', 4: class (1=32-bit, 2=64-bit)
                    if bytes.len() > 5 && &bytes[0..4] == b"\x7FELF" {
                        let ei_class = bytes[4];
                        if target_arch == "x86_64" || target_arch == "aarch64" {
                            // Require 64-bit for these targets
                            if ei_class != 2 { compatible = false; }
                        }
                    }
                }
            }
            if compatible {
                found_dir = Some(p.parent().unwrap().to_path_buf());
                break;
            } else {
                println!("cargo:warning=Ignoring incompatible lib at {} (architecture/class mismatch)", p.display());
            }
        }
    }

    // Always link against the dynamic lib name "vosk"
    println!("cargo:rustc-link-lib=dylib=vosk");

    if let Some(dir) = &found_dir {
        // Help the linker find the library at build/link time
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // Add robust rpaths so the runtime loader can find the library without env vars
    match target_os.as_str() {
        "linux" | "android" => {
            // Prefer relative rpaths so placing libvosk next to the binary works
            // Note: $ORIGIN is interpreted by the dynamic linker at runtime
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
            if let Some(dir) = &found_dir {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            }
        }
        "macos" => {
            // On macOS, use @loader_path instead of $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib");
            if let Some(dir) = &found_dir {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            }
        }
        _ => { /* Windows uses DLL search rules; skip rpath additions */ }
    }

    if found_dir.is_none() {
        // If not found in known locations, we still emit the link-lib. Users can set system path
        // or provide VOSK_LIB_DIR env var for Cargo so the build script can locate it next time.
        // For clarity during builds, print an informative message.
        println!("cargo:warning=libvosk not found in VOSK_LIB_DIR, ./src/model, or ./model. The system linker paths will be used.");
        println!("cargo:warning=If linking fails with 'cannot find -lvosk', set VOSK_LIB_DIR to the folder containing libvosk.");
        println!("cargo:warning=At runtime, you can also place libvosk next to the binary (target/<profile>/) thanks to embedded rpaths.");
    }
}
