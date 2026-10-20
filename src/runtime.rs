//! Runtime compilation (JIT): MIR to executable memory

use crate::error::RasError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

#[cfg(windows)]
mod windows_memory {
    use std::ffi::c_void;

    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const PAGE_EXECUTE_READ: u32 = 0x20;
    pub const PAGE_READWRITE: u32 = 0x04;

    unsafe extern "system" {
        pub fn VirtualAlloc(
            address: *mut c_void,
            size: usize,
            allocation_type: u32,
            protect: u32,
        ) -> *mut c_void;
        pub fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
        pub fn VirtualProtect(
            address: *mut c_void,
            size: usize,
            new_protect: u32,
            old_protect: *mut u32,
        ) -> i32;

        #[cfg(target_arch = "aarch64")]
        pub fn FlushInstructionCache(
            process: *mut c_void,
            base_address: *const c_void,
            size: usize,
        ) -> i32;
        #[cfg(target_arch = "aarch64")]
        pub fn GetCurrentProcess() -> *mut c_void;
    }
}

#[cfg(feature = "encoder")]
use crate::assembler::RasAssembler;

#[cfg(feature = "encoder")]
use lamina_mir::Module as MirModule;

/// Executable memory for JIT-compiled code
pub struct ExecutableMemory {
    ptr: *mut u8,
    size: usize,
}

impl ExecutableMemory {
    pub fn allocate_writable(size: usize) -> Result<Self, RasError> {
        #[cfg(unix)]
        {
            use libc::{MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap};
            use std::ptr;

            let aligned_size = (size + 4095) & !4095;

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            const MAP_JIT: libc::c_int = 0x0800;

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let map_flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_JIT;

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let map_flags = MAP_ANONYMOUS | MAP_PRIVATE;

            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    aligned_size,
                    PROT_READ | PROT_WRITE,
                    map_flags,
                    -1,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(RasError::IoError(format!(
                    "mmap failed (size: {} bytes)",
                    size
                )));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size: aligned_size,
            })
        }

        #[cfg(windows)]
        {
            use crate::runtime::windows_memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };

            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            };

            if ptr.is_null() {
                return Err(RasError::IoError("VirtualAlloc failed".to_string()));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
            })
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::UnsupportedTarget(
                "Executable memory allocation not supported on this platform".to_string(),
            ))
        }
    }

    pub fn make_executable(&mut self) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use libc::{PROT_EXEC, PROT_READ, mprotect};

            let result = unsafe {
                mprotect(
                    self.ptr as *mut libc::c_void,
                    self.size,
                    PROT_READ | PROT_EXEC,
                )
            };

            if result != 0 {
                return Err(RasError::IoError("mprotect failed".to_string()));
            }

            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                    unsafe extern "C" {
                        fn __clear_cache(start: *const u8, end: *const u8);
                    }
                    __clear_cache(self.ptr, self.ptr.add(self.size));
                    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
                }
            }

            Ok(())
        }

        #[cfg(windows)]
        {
            use crate::runtime::windows_memory::{PAGE_EXECUTE_READ, VirtualProtect};

            let mut old_protect = 0;
            let result = unsafe {
                VirtualProtect(
                    self.ptr.cast(),
                    self.size,
                    PAGE_EXECUTE_READ,
                    &mut old_protect,
                )
            };

            if result == 0 {
                return Err(RasError::IoError("VirtualProtect failed".to_string()));
            }

            // On Windows AArch64, flush the instruction cache so the CPU sees
            // the newly written code after the page protection change.
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use crate::runtime::windows_memory::{FlushInstructionCache, GetCurrentProcess};
                FlushInstructionCache(GetCurrentProcess(), self.ptr.cast(), self.size);
            }

            Ok(())
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::UnsupportedTarget(
                "Making memory executable not supported".to_string(),
            ))
        }
    }

    pub fn allocate(size: usize) -> Result<Self, RasError> {
        let mut mem = Self::allocate_writable(size)?;
        mem.make_executable()?;
        Ok(mem)
    }

    pub fn write_code(&mut self, code: &[u8]) -> Result<(), RasError> {
        if code.len() > self.size {
            return Err(RasError::IoError("Code too large".to_string()));
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                unsafe extern "C" {
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(0);
            }
        }

        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *mut libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(self.ptr as *mut libc::c_void, code.len());
            }
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                unsafe extern "C" {
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(1);
            }
        }

        Ok(())
    }

    pub fn code_start(&self) -> *const u8 {
        self.ptr
    }

    /// Interprets the first byte of this region as the entry address for a function pointer.
    ///
    /// This is **not** a pointer-to-pointer: the mapping begins with machine code, not a stored
    /// address. Do not use `*const F` to the entry and then dereference; that would load
    /// instruction bytes as a pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `F` matches the ABI and signature of the generated code, and that
    /// this `ExecutableMemory` outlives every call through `F`.
    pub unsafe fn entry_fn<F: Sized>(&self) -> F {
        debug_assert_eq!(size_of::<F>(), size_of::<*mut u8>());
        unsafe { core::mem::transmute_copy(&self.ptr) }
    }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use libc::munmap;
            unsafe {
                munmap(self.ptr as *mut libc::c_void, self.size);
            }
        }

        #[cfg(windows)]
        {
            use crate::runtime::windows_memory::{MEM_RELEASE, VirtualFree};
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }
}

/// Runtime compiler: MIR to executable memory
pub struct RasRuntime {
    #[cfg(feature = "encoder")]
    target_arch: TargetArchitecture,
    #[cfg(feature = "encoder")]
    target_os: TargetOperatingSystem,
}

impl RasRuntime {
    pub fn new(
        #[cfg(feature = "encoder")] target_arch: TargetArchitecture,
        #[cfg(not(feature = "encoder"))] _target_arch: TargetArchitecture,
        #[cfg(feature = "encoder")] target_os: TargetOperatingSystem,
        #[cfg(not(feature = "encoder"))] _target_os: TargetOperatingSystem,
    ) -> Self {
        Self {
            #[cfg(feature = "encoder")]
            target_arch,
            #[cfg(feature = "encoder")]
            target_os,
        }
    }

    #[cfg(feature = "encoder")]
    pub fn compile_to_memory(&mut self, module: &MirModule) -> Result<ExecutableMemory, RasError> {
        let mut assembler = RasAssembler::new(self.target_arch, self.target_os)?;
        let (code, _) = assembler.compile_mir_to_binary_function(module, None)?;

        let mut mem = ExecutableMemory::allocate_writable(code.len())?;
        mem.write_code(&code)?;
        mem.make_executable()?;

        Ok(mem)
    }

    /// Leaks the executable mapping so the returned pointer stays valid. Prefer
    /// [`compile_to_memory`](Self::compile_to_memory) and keep the `ExecutableMemory` alive.
    #[cfg(feature = "encoder")]
    pub fn compile_function<T>(
        &mut self,
        module: &MirModule,
        _function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, RasError> {
        let mem = self.compile_to_memory(module)?;
        let f: unsafe extern "C" fn() -> T = unsafe { core::mem::transmute(mem.ptr) };
        std::mem::forget(mem);
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_memory_allocate_and_write() {
        let mut mem = ExecutableMemory::allocate_writable(4096).expect("alloc failed");
        let code = [0x90u8, 0x90, 0x90, 0xc3];
        mem.write_code(&code).expect("write failed");
        mem.make_executable().expect("make_executable failed");
        assert!(!mem.code_start().is_null());
    }

    #[test]
    fn executable_memory_too_large_returns_error() {
        // Allocation rounds up to a page (4096 bytes). Use exactly one page so
        // the next allocation produces a 4096-byte region, then try to write 8192.
        let mem = ExecutableMemory::allocate_writable(4096).expect("alloc failed");
        let big = vec![0u8; 8192];
        let result = {
            let mut m = mem;
            m.write_code(&big)
        };
        assert!(result.is_err(), "writing beyond capacity should fail");
    }

    #[test]
    fn ras_runtime_new_does_not_panic() {
        let _rt = RasRuntime::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
    }
}

/// Run JIT output on the **host** CPU (x86_64 or aarch64 only). Skips when the host OS string
/// is not mapped in `TargetOperatingSystem` (e.g. some embedded targets).
#[cfg(all(test, feature = "encoder"))]
mod jit_host_exec_tests {
    use std::str::FromStr;

    use lamina_mir::{
        block::Block,
        function::{Function, Parameter, Signature},
        instruction::{Immediate, Instruction, IntBinOp, IntCmpOp, Operand},
        module::Module,
        register::{Register, VirtualReg},
        types::{MirType, ScalarType},
    };
    use lamina_platform::{
        TargetArchitecture, TargetOperatingSystem, detect_host_architecture_only, detect_host_os,
    };

    use crate::{assembler::core::RasAssembler, error::RasError, runtime::ExecutableMemory};

    fn host_jit_pair() -> Option<(TargetArchitecture, TargetOperatingSystem)> {
        let arch = TargetArchitecture::from_str(detect_host_architecture_only()).ok()?;
        let os = TargetOperatingSystem::from_str(detect_host_os()).ok()?;
        match arch {
            TargetArchitecture::X86_64 | TargetArchitecture::Aarch64 => Some((arch, os)),
            _ => None,
        }
    }

    fn compile_to_executable(
        arch: TargetArchitecture,
        os: TargetOperatingSystem,
        module: &lamina_mir::Module,
    ) -> Result<ExecutableMemory, RasError> {
        let mut asm = RasAssembler::new(arch, os)?;
        let (code, _) = asm.compile_mir_to_binary_function(module, None)?;
        let mut mem = ExecutableMemory::allocate_writable(code.len())?;
        mem.write_code(&code)?;
        mem.make_executable()?;
        Ok(mem)
    }

    fn exec_host_i32(module: &Module) -> Option<i32> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).ok()?;
        let f = unsafe { mem.entry_fn::<extern "C" fn() -> i32>() };
        Some(f())
    }

    fn exec_host_i64(module: &Module) -> Option<i64> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).ok()?;
        let f = unsafe { mem.entry_fn::<extern "C" fn() -> i64>() };
        Some(f())
    }

    fn exec_host_i64_binary(module: &Module, a: i64, b: i64) -> Option<i64> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).ok()?;
        let f = unsafe { mem.entry_fn::<extern "C" fn(i64, i64) -> i64>() };
        Some(f(a, b))
    }

    fn ii32(v: i32) -> Operand {
        Operand::Immediate(Immediate::I32(v))
    }

    fn ii64(v: i64) -> Operand {
        Operand::Immediate(Immediate::I64(v))
    }

    fn module_scalar_binop(scalar: ScalarType, op: IntBinOp, lhs: Operand, rhs: Operand) -> Module {
        let ty = MirType::Scalar(scalar);
        let out = Register::Virtual(VirtualReg::gpr(0));
        let sig = Signature::new("f").with_return(ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty,
            dst: out.clone(),
            lhs,
            rhs,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("host_scalar_binop");
        module.add_function(f);
        module
    }

    fn module_i64_binop_params(op: IntBinOp) -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("g")
            .with_params(vec![
                Parameter::new(a.clone(), i64_ty),
                Parameter::new(b.clone(), i64_ty),
            ])
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i64_ty,
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("host_i64_params");
        module.add_function(f);
        module
    }

    fn module_i32_cmp_eq(lhs: Operand, rhs: Operand) -> Module {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let out = Register::Virtual(VirtualReg::gpr(0));
        let sig = Signature::new("cmp").with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntCmp {
            op: IntCmpOp::Eq,
            ty: i32_ty,
            dst: out.clone(),
            lhs,
            rhs,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("host_i32_cmp");
        module.add_function(f);
        module
    }

    #[test]
    fn jit_host_exec_i64_add_immediates() {
        let Some(got) = exec_host_i64(&module_scalar_binop(
            ScalarType::I64,
            IntBinOp::Add,
            ii64(10),
            ii64(32),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_sub_immediates() {
        let Some(got) = exec_host_i64(&module_scalar_binop(
            ScalarType::I64,
            IntBinOp::Sub,
            ii64(100),
            ii64(33),
        )) else {
            return;
        };
        assert_eq!(got, 67);
    }

    #[test]
    fn jit_host_exec_i64_mul_immediates() {
        let Some(got) = exec_host_i64(&module_scalar_binop(
            ScalarType::I64,
            IntBinOp::Mul,
            ii64(6),
            ii64(7),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_add_two_arguments() {
        let Some(got) = exec_host_i64_binary(&module_i64_binop_params(IntBinOp::Add), 7, 35) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_mul_two_arguments() {
        let Some(got) = exec_host_i64_binary(&module_i64_binop_params(IntBinOp::Mul), 6, 7) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i32_sub_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Sub,
            ii32(50),
            ii32(8),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i32_mul_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Mul,
            ii32(6),
            ii32(7),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i32_and_or_xor_immediates() {
        let Some(a) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::And,
            ii32(0x0F),
            ii32(0x33),
        )) else {
            return;
        };
        assert_eq!(a, 3);
        let Some(o) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Or,
            ii32(8),
            ii32(2),
        )) else {
            return;
        };
        assert_eq!(o, 10);
        let Some(x) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Xor,
            ii32(15),
            ii32(5),
        )) else {
            return;
        };
        assert_eq!(x, 10);
    }

    #[test]
    fn jit_host_exec_i32_udiv_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::UDiv,
            ii32(10),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(got, 3);
    }

    #[test]
    fn jit_host_exec_i32_sdiv_negative_dividend() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::SDiv,
            ii32(-7),
            ii32(2),
        )) else {
            return;
        };
        assert_eq!(got, -3);
    }

    #[test]
    fn jit_host_exec_i32_urem_srem_immediates() {
        let Some(u) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::URem,
            ii32(10),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(u, 1);
        let Some(s) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::SRem,
            ii32(-7),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(s, -1);
    }

    #[test]
    fn jit_host_exec_i32_shl_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Shl,
            ii32(3),
            ii32(4),
        )) else {
            return;
        };
        assert_eq!(got, 48);
    }

    #[test]
    fn jit_host_exec_i32_lshr_ashr_immediates() {
        let Some(l) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::LShr,
            ii32(128),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(l, 16);
        let Some(a) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::AShr,
            ii32(-16),
            ii32(2),
        )) else {
            return;
        };
        assert_eq!(a, -4);
    }

    #[test]
    fn jit_host_exec_i32_intcmp_eq_immediates() {
        let Some(eq) = exec_host_i32(&module_i32_cmp_eq(ii32(9), ii32(9))) else {
            return;
        };
        assert_eq!(eq, 1);
        let Some(ne) = exec_host_i32(&module_i32_cmp_eq(ii32(2), ii32(3))) else {
            return;
        };
        assert_eq!(ne, 0);
    }
}
