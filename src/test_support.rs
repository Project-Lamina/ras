use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub(crate) fn new(label: &str) -> Self {
        let base = option_env!("CARGO_TARGET_TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/test-tmp"))
            });
        std::fs::create_dir_all(&base).unwrap();
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = base.join(format!("ras-{label}-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Kept on failure so the output files of a failing test can be inspected.
        if std::thread::panicking() {
            return;
        }
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The `ras` executable this crate builds. `cargo test` builds it as part of the bin
/// target; `cargo test --lib` does not, so say which command is missing rather than
/// failing later on an absent program. Cargo places the test binary at a depth that has
/// changed between releases, so every directory up to `target` is searched.
#[allow(dead_code)]
pub(crate) fn ras_binary() -> PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    let name = format!("ras{}", std::env::consts::EXE_SUFFIX);
    for dir in exe.ancestors().skip(1) {
        let candidate = dir.join(&name);
        if candidate.is_file() {
            return candidate;
        }
        if dir.file_name().is_some_and(|last| last == "target") {
            break;
        }
    }
    panic!("{name} is not built: run `cargo build --bin ras`");
}
