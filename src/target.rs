//! Target support: which (arch, OS) combinations are supported.

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

pub fn is_assembly_target_supported(
    arch: TargetArchitecture,
    os: TargetOperatingSystem,
) -> bool {
    if !is_arch_encoded(arch) {
        return false;
    }
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::MacOS
            | TargetOperatingSystem::Windows
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
    )
}

pub fn is_object_file_supported(arch: TargetArchitecture, os: TargetOperatingSystem) -> bool {
    if !is_arch_encoded(arch) {
        return false;
    }
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
    )
}

pub fn is_jit_arch_supported(arch: TargetArchitecture) -> bool {
    matches!(
        arch,
        TargetArchitecture::X86_64 | TargetArchitecture::Aarch64
    )
}

fn is_arch_encoded(arch: TargetArchitecture) -> bool {
    matches!(
        arch,
        TargetArchitecture::X86_64 | TargetArchitecture::Aarch64
    )
}

pub fn unsupported_target_hint(arch: TargetArchitecture, os: TargetOperatingSystem) -> String {
    if !is_arch_encoded(arch) {
        format!(
            "Architecture {:?} has no binary encoder. For RISC-V/WASM use lamina pipeline.",
            arch
        )
    } else if !is_assembly_target_supported(arch, os) {
        format!(
            "OS {:?} not supported. Supported: Linux, macOS, Windows, FreeBSD, OpenBSD, NetBSD, DragonFly, Redox.",
            os
        )
    } else {
        "Target should be supported; check ras::target module.".to_string()
    }
}

pub fn supported_jit_targets_hint() -> &'static str {
    "JIT supports x86_64 and AArch64 only."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_arch_supported() {
        assert!(is_jit_arch_supported(TargetArchitecture::X86_64));
        assert!(is_jit_arch_supported(TargetArchitecture::Aarch64));
        assert!(!is_jit_arch_supported(TargetArchitecture::Riscv64));
    }

    #[test]
    fn test_object_file_supported() {
        assert!(is_object_file_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
        ));
        assert!(!is_object_file_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::MacOS
        ));
    }
}
