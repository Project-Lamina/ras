#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> Self {
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

    pub fn path(&self) -> &Path {
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
