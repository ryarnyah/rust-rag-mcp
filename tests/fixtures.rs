//! Helpers shared by more than one test binary.
//!
//! Not every test binary uses every helper here.
#![allow(dead_code)]

use std::path::PathBuf;

/// The suite's shared embedding-model cache (the repo's `.fastembed_cache`).
///
/// Tests must *not* point the model cache at their own temp dir: every test
/// would re-download the ~130MB ONNX model (30+ seconds each, and flaky when
/// the hub throttles repeated downloads). One shared, gitignored dir is
/// downloaded once and then serves every test offline.
pub fn model_cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".fastembed_cache")
}

pub fn sample_text() -> String {
    "Rust is a systems programming language focused on safety, speed, and concurrency. \
     It achieves memory safety without garbage collection through its ownership system. \
     The borrow checker enforces strict rules about how references can be used. \
     Rust is used in web browsers, operating systems, and game engines. \
     The crate ecosystem provides libraries for almost any task. \
     Async Rust enables efficient concurrent programming with the tokio runtime. \
     The Rust compiler provides detailed error messages to help developers fix issues. \
     Cargo is the build system and package manager for Rust projects."
        .to_string()
}
