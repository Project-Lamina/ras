//! Target support: which (arch, OS) combinations are supported.

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

pub fn is_assembly_target_supported(arch: TargetArchitecture, os: TargetOperatingSystem) -> bool {
    if !is_arch_encoded(arch) {
        return false;
    }
    if arch == TargetArchitecture::Arx64 {
        return matches!(
            os,
            TargetOperatingSystem::Linux
                | TargetOperatingSystem::FreeBSD
                | TargetOperatingSystem::OpenBSD
                | TargetOperatingSystem::NetBSD
                | TargetOperatingSystem::DragonFly
                | TargetOperatingSystem::Redox
        );
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
            | TargetOperatingSystem::Orbis
            | TargetOperatingSystem::Prospero
    )
}

pub fn is_object_file_supported(arch: TargetArchitecture, os: TargetOperatingSystem) -> bool {
    if !is_arch_encoded(arch) {
        return false;
    }
    match object_format_for_os(os) {
        // ELF is the complete writer and carries every encoded architecture.
        Some(ObjectFormat::Elf) => !matches!(
            (arch, os),
            // ARX64 and RISC-V exist on Linux and the BSDs only.
            (
                TargetArchitecture::Arx64
                    | TargetArchitecture::Riscv32
                    | TargetArchitecture::Riscv64,
                TargetOperatingSystem::Orbis | TargetOperatingSystem::Prospero
            )
        ),
        // The COFF writer rejects anything but x86_64, see object.rs.
        Some(ObjectFormat::Coff) => arch == TargetArchitecture::X86_64,
        Some(ObjectFormat::MachO) => arch == TargetArchitecture::X86_64,
        None => false,
    }
}

/// Object container for an OS, independent of whether the writer for it is finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectFormat {
    Elf,
    MachO,
    Coff,
}

pub fn object_format_for_os(os: TargetOperatingSystem) -> Option<ObjectFormat> {
    match os {
        TargetOperatingSystem::Linux
        | TargetOperatingSystem::FreeBSD
        | TargetOperatingSystem::OpenBSD
        | TargetOperatingSystem::NetBSD
        | TargetOperatingSystem::DragonFly
        | TargetOperatingSystem::Redox
        | TargetOperatingSystem::Orbis
        | TargetOperatingSystem::Prospero => Some(ObjectFormat::Elf),
        TargetOperatingSystem::MacOS => Some(ObjectFormat::MachO),
        TargetOperatingSystem::Windows => Some(ObjectFormat::Coff),
        _ => None,
    }
}

pub fn is_jit_arch_supported(arch: TargetArchitecture) -> bool {
    matches!(
        arch,
        TargetArchitecture::X86_64 | TargetArchitecture::Aarch64 | TargetArchitecture::Riscv64
    )
}

fn is_arch_encoded(arch: TargetArchitecture) -> bool {
    matches!(
        arch,
        TargetArchitecture::X86_64
            | TargetArchitecture::Aarch64
            | TargetArchitecture::Arx64
            | TargetArchitecture::Riscv32
            | TargetArchitecture::Riscv64
    )
}

pub fn unsupported_target_hint(arch: TargetArchitecture, os: TargetOperatingSystem) -> String {
    if !is_arch_encoded(arch) {
        format!(
            "Architecture {:?} has no binary encoder. For WASM use lamina pipeline.",
            arch
        )
    } else if !is_assembly_target_supported(arch, os) {
        if arch == TargetArchitecture::Arx64 {
            return format!(
                "OS {:?} not supported for ARX64 ras output. Supported: Linux, FreeBSD, OpenBSD, NetBSD, DragonFly, Redox.",
                os
            );
        }
        format!(
            "OS {:?} not supported. Supported: Linux, macOS, Windows, FreeBSD, OpenBSD, NetBSD, DragonFly, Redox, Orbis (PS4), Prospero (PS5).",
            os
        )
    } else if !is_object_file_supported(arch, os) {
        match object_format_for_os(os) {
            Some(ObjectFormat::MachO) => format!(
                "ras can encode {arch:?} but cannot yet write a Mach-O object, so {os:?} \
                 output is unavailable. Assemble with the system assembler for now."
            ),
            Some(ObjectFormat::Coff) => format!(
                "ras writes COFF objects for x86_64 only, so {arch:?} on {os:?} is \
                 unavailable."
            ),
            _ => format!("No object format is mapped for {os:?}."),
        }
    } else {
        "Target should be supported; check ras::target module.".to_string()
    }
}

pub fn supported_jit_targets_hint() -> &'static str {
    "JIT emits machine code for x86_64, AArch64, and RISC-V 64 (integer-focused MIR subset; see ras assembler JIT). ARX64 supports assembly/object emission only."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_arch_supported() {
        assert!(is_jit_arch_supported(TargetArchitecture::X86_64));
        assert!(is_jit_arch_supported(TargetArchitecture::Aarch64));
        assert!(is_jit_arch_supported(TargetArchitecture::Riscv64));
        assert!(!is_jit_arch_supported(TargetArchitecture::Riscv32));
    }

    #[test]
    fn test_object_file_supported() {
        assert!(is_object_file_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
        ));
        assert!(is_object_file_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::MacOS
        ));
        assert!(!is_object_file_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::MacOS
        ));
        assert!(is_object_file_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::MacOS
        ));
        assert!(is_object_file_supported(
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Linux
        ));
        assert!(!is_object_file_supported(
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::MacOS
        ));
        assert!(is_object_file_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows
        ));
        assert!(!is_object_file_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Windows
        ));
    }

    #[test]
    fn unsupported_object_target_says_why() {
        let macos =
            unsupported_target_hint(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS);
        assert!(macos.contains("Mach-O"), "unhelpful hint: {macos}");

        let win_arm =
            unsupported_target_hint(TargetArchitecture::Aarch64, TargetOperatingSystem::Windows);
        assert!(win_arm.contains("COFF"), "unhelpful hint: {win_arm}");
    }
}
