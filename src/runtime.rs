//! Runtime compilation (JIT): MIR to executable memory

use crate::assembler::RasAssembler;
use crate::error::RasError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

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
            use winapi::um::memoryapi::VirtualAlloc;
            use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};

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
            use winapi::um::memoryapi::VirtualProtect;
            use winapi::um::winnt::PAGE_EXECUTE_READ;

            let mut old_protect: winapi::um::winnt::DWORD = 0;
            let result = unsafe {
                VirtualProtect(
                    self.ptr as *mut winapi::um::winnt::c_void,
                    self.size,
                    PAGE_EXECUTE_READ,
                    &mut old_protect,
                )
            };

            if result == 0 {
                return Err(RasError::IoError("VirtualProtect failed".to_string()));
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
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(1);
            }
        }

        Ok(())
    }

    pub unsafe fn as_function_ptr<T>(&self) -> *const T {
        self.ptr as *const T
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
            use winapi::um::memoryapi::VirtualFree;
            use winapi::um::winnt::MEM_RELEASE;
            unsafe {
                VirtualFree(self.ptr as *mut winapi::um::winnt::c_void, 0, MEM_RELEASE);
            }
        }
    }
}

/// Runtime compiler: MIR to executable memory
pub struct RasRuntime {
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
}

impl RasRuntime {
    pub fn new(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Self {
        Self {
            target_arch,
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

    #[cfg(feature = "encoder")]
    pub fn compile_function<T>(
        &mut self,
        module: &MirModule,
        _function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, RasError> {
        let mem = self.compile_to_memory(module)?;
        unsafe {
            let ptr: *const unsafe extern "C" fn() -> T = mem.as_function_ptr();
            Ok(*ptr)
        }
    }
}
