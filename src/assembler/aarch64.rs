//! AArch64 binary code generation
//!
//! This module handles compilation of MIR to AArch64 binary machine code.

use crate::encoder::aarch64::fixed::{
    enc_ldp_post_64, enc_stp_pre_64, encode_add_rrr_aarch64, encode_and_rrr_aarch64,
    encode_and_w_rrr_aarch64, encode_msub_aarch64, encode_msub_aarch64_w,
};
use crate::{
    encoder::aarch64::ldst_imm64::{
        AArch64ScalarLdKind, AArch64ScalarStKind, encode_ldr_imm64, encode_ldr_scalar,
        encode_str_imm64, encode_str_scalar,
    },
    assembler::core::RasAssembler,
    error::RasError,
};
use lamina_codegen::aarch64::{A64RegAlloc, AArch64ABI, FrameMap};
use lamina_mir::{
    AddressMode, Immediate, IntBinOp, IntCmpOp, MirType, Operand, Register, ScalarType,
};

#[cfg(feature = "encoder")]
#[derive(Debug, Clone)]
struct BlFixup {
    /// Offset in the final code buffer where the BL instruction word begins.
    patch_location: usize,
    /// Target function name as referenced by MIR (may include or omit '@').
    target_name: String,
}

#[cfg(feature = "encoder")]
static PRINT_I64_FORMAT: [u8; 6] = *b"%lld\n\0";

/// Compile MIR to binary for AArch64
///
/// This reuses the instruction emission logic from mir_codegen/aarch64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
pub fn compile_mir_aarch64_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    _function_name: Option<&str>,
) -> Result<(Vec<u8>, std::collections::HashMap<String, usize>), RasError> {
    let _abi = AArch64ABI::new(assembler.target_os);
    let mut code = Vec::new();
    let mut function_offsets: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    // Always compile all functions (needed for internal function calls)
    // Collect all function names first to ensure deterministic order
    let mut all_function_names: Vec<String> = module.functions.keys().cloned().collect();
    all_function_names.sort(); // Sort for deterministic order

    // Pre-calculate function offsets (estimate) to handle recursive calls
    // We'll use estimated offsets initially, then update with actual offsets as we compile
    let mut estimated_sizes: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for func_name in &all_function_names {
        if let Some(func) = module.functions.get(func_name) {
            // Rough estimate: prologue + epilogue + RET + instructions
            let inst_count = func
                .blocks
                .iter()
                .map(|b| b.instructions.len())
                .sum::<usize>();
            let estimated_size = 16 + (inst_count * 4) + 12 + 4; // prologue + instructions + epilogue + ret
            estimated_sizes.insert(func_name.clone(), estimated_size);
        }
    }

    let total_est: usize = estimated_sizes.values().copied().sum();
    code.reserve(total_est.saturating_add(256));

    // Pre-populate function_offsets with estimated offsets
    let mut current_estimate = 0;
    for func_name in &all_function_names {
        function_offsets.insert(func_name.clone(), current_estimate);
        current_estimate += estimated_sizes.get(func_name).copied().unwrap_or(100);
    }

    // Track internal direct-call fixups (BL) and patch them after final function offsets are known.
    let mut bl_fixups: Vec<BlFixup> = Vec::new();
    // Tail calls use unconditional B (same patch shape as BL: 26-bit PC-relative).
    let mut tail_b_fixups: Vec<BlFixup> = Vec::new();

    // Now compile all functions, updating offsets with actual values
    for func_name in &all_function_names {
        let func = module.functions.get(func_name).ok_or_else(|| {
            RasError::EncodingError(format!("Function '{}' not found in module", func_name))
        })?;

        // Update function offset with actual value
        function_offsets.insert(func_name.clone(), code.len());
        let mut reg_alloc = A64RegAlloc::new();
        let frame = FrameMap::from_function(func);
        let mut stack_slots: std::collections::HashMap<lamina_mir::VirtualReg, i32> =
            std::collections::HashMap::new();

        // Convert FrameMap slots to HashMap for easier lookup
        for (reg, offset) in &frame.slots {
            if let Register::Virtual(vreg) = reg {
                stack_slots.insert(*vreg, *offset);
            }
        }

        let stack_size = frame.frame_size as usize;

        // Ensure stack is 16-byte aligned (AAPCS64 requirement)
        // The prologue saves x29, x30 (16 bytes), so we need to ensure
        // the total stack frame is 16-byte aligned
        let aligned_stack_size = (stack_size + 15) & !15;

        // Generate function prologue (binary encoded)
        let prologue = encode_prologue_aarch64(aligned_stack_size)?;
        code.extend_from_slice(&prologue);

        // Handle function parameters: spill ABI arg registers to the FrameMap stack slots.
        //
        // Our JIT encoder currently materializes virtual registers by loading from their
        // stack slot, so parameters must be stored to their slots up-front.
        if !func.sig.params.is_empty() {
            let arg_regs = AArch64ABI::ARG_REGISTERS;

            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        // Store x0-x7 directly to the virtual register stack slot.
                        let str_bytes =
                            encode_str_aarch64(arg_regs[index], 29 /* x29 (FP) */, *slot_off)?;
                        code.extend_from_slice(&str_bytes);
                    } else {
                        // Handle stack arguments (AAPCS64: stack args start at caller's [sp, #0])
                        // After prologue: stp x29,x30,[sp,#-16]! then mov x29,sp
                        // Caller's stack args are now at [x29, #16] (16 bytes for saved fp/lr)
                        // First stack arg (arg8) is at [x29, #16], second at [x29, #24], etc.
                        let stack_arg_index = index - arg_regs.len();
                        let caller_off = (16 + stack_arg_index * 8) as i32; // 16 for saved fp/lr
                        let ldr1 = encode_ldr_aarch64("x10", 29, caller_off)?;
                        code.extend_from_slice(&ldr1);
                        let str1 = encode_str_aarch64("x10", 29, *slot_off)?;
                        code.extend_from_slice(&str1);
                    }
                }
            }
        }

        #[derive(Debug)]
        enum BranchFixupKind {
            B { target: String },
            Cbnz { rt: u8, target: String },
            BToEpilogue,
        }

        #[derive(Debug)]
        struct BranchFixup {
            patch_location: usize,
            kind: BranchFixupKind,
        }

        fn write_u32_le(buf: &mut [u8], at: usize, word: u32) -> Result<(), RasError> {
            if at + 4 > buf.len() {
                return Err(RasError::EncodingError(format!(
                    "Patch location out of bounds: {} (len={})",
                    at,
                    buf.len()
                )));
            }
            buf[at..at + 4].copy_from_slice(&word.to_le_bytes());
            Ok(())
        }

        fn encode_b(from_pc: usize, to_pc: usize) -> Result<u32, RasError> {
            let delta = to_pc as i64 - from_pc as i64;
            if delta % 4 != 0 {
                return Err(RasError::EncodingError(format!(
                    "Unaligned B target delta {} (from={}, to={})",
                    delta, from_pc, to_pc
                )));
            }
            let imm26 = delta / 4;
            if !(-(1i64 << 25)..(1i64 << 25)).contains(&imm26) {
                return Err(RasError::EncodingError(format!(
                    "B target out of range (delta={} bytes)",
                    delta
                )));
            }
            Ok(0x1400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF))
        }

        fn encode_cbnz(rt: u8, from_pc: usize, to_pc: usize) -> Result<u32, RasError> {
            let delta = to_pc as i64 - from_pc as i64;
            if delta % 4 != 0 {
                return Err(RasError::EncodingError(format!(
                    "Unaligned CBNZ target delta {} (from={}, to={})",
                    delta, from_pc, to_pc
                )));
            }
            let imm19 = delta / 4;
            if !(-(1i64 << 18)..(1i64 << 18)).contains(&imm19) {
                return Err(RasError::EncodingError(format!(
                    "CBNZ target out of range (delta={} bytes)",
                    delta
                )));
            }
            Ok(0xB500_0000u32 | (((imm19 as u32) & 0x7_FFFF) << 5) | (rt as u32))
        }

        let mut block_offsets: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut branch_fixups: Vec<BranchFixup> = Vec::new();

        // Compile blocks. Terminators are handled here (never silently dropped).
        for block in &func.blocks {
            block_offsets.insert(block.label.clone(), code.len());

            let term = block.terminator().ok_or_else(|| {
                RasError::EncodingError(format!(
                    "Block '{}' has no terminator (invalid MIR)",
                    block.label
                ))
            })?;

            for inst in block.body() {
                if inst.is_terminator() {
                    return Err(RasError::EncodingError(format!(
                        "Terminator found in block body '{}' (invalid MIR): {:?}",
                        block.label, inst
                    )));
                }
                let current_offset = code.len();
                let inst_bytes = encode_mir_instruction_aarch64_with_context(
                    assembler,
                    inst,
                    &mut reg_alloc,
                    &stack_slots,
                    aligned_stack_size,
                    func.sig.ret_ty.as_ref(),
                    func_name,
                    &function_offsets,
                    current_offset,
                    &mut bl_fixups,
                )?;
                code.extend_from_slice(&inst_bytes);
            }

            match term {
                lamina_mir::Instruction::Ret { value } => {
                    if let Some(v) = value {
                        if let Some(rt) = &func.sig.ret_ty {
                            materialize_return_value_aarch64(
                                assembler,
                                v,
                                rt,
                                &stack_slots,
                                &mut reg_alloc,
                                &mut code,
                                aligned_stack_size,
                            )?;
                        } else {
                            materialize_operand_aarch64(
                                assembler,
                                v,
                                0,
                                &stack_slots,
                                &mut reg_alloc,
                                &mut code,
                                aligned_stack_size,
                            )?;
                        }
                    }
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes()); // B <epilogue> (patched)
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::BToEpilogue,
                    });
                }
                lamina_mir::Instruction::Jmp { target } => {
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes()); // B <target> (patched)
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::B {
                            target: target.clone(),
                        },
                    });
                }
                lamina_mir::Instruction::Br {
                    cond,
                    true_target,
                    false_target,
                } => {
                    // Load condition value and branch on non-zero.
                    let cond_reg_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                    let cond_reg = parse_register_aarch64(cond_reg_str)?;
                    materialize_operand_aarch64(
                        assembler,
                        &lamina_mir::Operand::Register(cond.clone()),
                        cond_reg,
                        &stack_slots,
                        &mut reg_alloc,
                        &mut code,
                        aligned_stack_size,
                    )?;
                    reg_alloc.free_scratch(cond_reg_str);

                    // CBNZ <cond>, <true>
                    let patch_location = code.len();
                    let placeholder = 0xB500_0000u32 | (cond_reg as u32);
                    code.extend_from_slice(&placeholder.to_le_bytes());
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::Cbnz {
                            rt: cond_reg,
                            target: true_target.clone(),
                        },
                    });

                    // B <false>
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::B {
                            target: false_target.clone(),
                        },
                    });
                }
                lamina_mir::Instruction::Switch { .. } => {
                    return Err(RasError::EncodingError(
                        "Switch terminator not yet supported by AArch64 JIT backend".to_string(),
                    ));
                }
                lamina_mir::Instruction::TailCall { name, args } => {
                    if name == "print" {
                        return Err(RasError::EncodingError(
                            "TailCall to print() is not supported by AArch64 JIT".to_string(),
                        ));
                    }

                    let is_internal = function_offsets.contains_key(name)
                        || (name.starts_with('@') && function_offsets.contains_key(&name[1..]))
                        || (!name.starts_with('@')
                            && function_offsets.contains_key(&format!("@{}", name)));
                    if !is_internal {
                        return Err(RasError::EncodingError(format!(
                            "External tail call '{}' requires runtime resolution (not implemented for AArch64 JIT)",
                            name
                        )));
                    }

                    let arg_regs = AArch64ABI::ARG_REGISTERS;
                    for (i, arg) in args.iter().enumerate().take(8) {
                        let dst = parse_register_aarch64(arg_regs[i])?;
                        materialize_operand_aarch64(
                            assembler,
                            arg,
                            dst,
                            &stack_slots,
                            &mut reg_alloc,
                            &mut code,
                            aligned_stack_size,
                        )?;
                    }
                    for (i, arg) in args.iter().enumerate().skip(8) {
                        let caller_off = 16i32 + (i as i32 - 8) * 8;
                        materialize_operand_aarch64(
                            assembler,
                            arg,
                            11,
                            &stack_slots,
                            &mut reg_alloc,
                            &mut code,
                            aligned_stack_size,
                        )?;
                        code.extend_from_slice(&encode_str_aarch64("x11", 29, caller_off)?);
                    }

                    let epilogue_tail = encode_epilogue_aarch64(aligned_stack_size)?;
                    code.extend_from_slice(&epilogue_tail);

                    let b_pc = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                    tail_b_fixups.push(BlFixup {
                        patch_location: b_pc,
                        target_name: name.clone(),
                    });
                }
                lamina_mir::Instruction::Unreachable => {
                    code.extend_from_slice(&encode_brk_aarch64(0));
                }
                other => {
                    return Err(RasError::EncodingError(format!(
                        "Unexpected terminator in block '{}': {:?}",
                        block.label, other
                    )));
                }
            }
        }

        // Patch branches that target blocks or the epilogue.
        let epilogue_offset = code.len();
        for fix in &branch_fixups {
            let from_pc = fix.patch_location;
            let to_pc = match &fix.kind {
                BranchFixupKind::BToEpilogue => epilogue_offset,
                BranchFixupKind::B { target } | BranchFixupKind::Cbnz { target, .. } => {
                    *block_offsets.get(target).ok_or_else(|| {
                        RasError::EncodingError(format!(
                            "Branch target block '{}' not found in function '{}'",
                            target, func_name
                        ))
                    })?
                }
            };
            let patched = match &fix.kind {
                BranchFixupKind::BToEpilogue | BranchFixupKind::B { .. } => {
                    encode_b(from_pc, to_pc)?
                }
                BranchFixupKind::Cbnz { rt, .. } => encode_cbnz(*rt, from_pc, to_pc)?,
            };
            write_u32_le(&mut code, fix.patch_location, patched)?;
        }

        // Generate function epilogue (must pass aligned_stack_size to restore SP)
        let epilogue = encode_epilogue_aarch64(aligned_stack_size)?;
        code.extend_from_slice(&epilogue);

        // RET instruction (x30 is LR)
        code.extend_from_slice(&encode_ret_aarch64(30)?);
    }

    // Patch BL fixups now that all functions have their final offsets.
    fn lookup_function_offset(
        function_offsets: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Option<usize> {
        function_offsets.get::<str>(name).copied().or_else(|| {
            if let Some(stripped) = name.strip_prefix('@') {
                function_offsets.get(stripped).copied()
            } else {
                function_offsets.get(&format!("@{}", name)).copied()
            }
        })
    }

    for fixup in &bl_fixups {
        let target_offset = lookup_function_offset(&function_offsets, &fixup.target_name)
            .ok_or_else(|| {
                RasError::EncodingError(format!(
                    "BL target function '{}' not found. Available: {:?}",
                    fixup.target_name,
                    function_offsets.keys().collect::<Vec<_>>()
                ))
            })?;

        let from_pc = fixup.patch_location;
        let delta = target_offset as i64 - from_pc as i64;
        if delta % 4 != 0 {
            return Err(RasError::EncodingError(format!(
                "Unaligned BL target delta {} (from={}, to={})",
                delta, from_pc, target_offset
            )));
        }
        let imm26 = delta / 4;
        if !(-(1i64 << 25)..(1i64 << 25)).contains(&imm26) {
            return Err(RasError::EncodingError(format!(
                "BL target out of range (delta={} bytes)",
                delta
            )));
        }
        let word = 0x9400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF);
        if fixup.patch_location + 4 > code.len() {
            return Err(RasError::EncodingError(format!(
                "BL patch location out of bounds: {} (len={})",
                fixup.patch_location,
                code.len()
            )));
        }
        code[fixup.patch_location..fixup.patch_location + 4].copy_from_slice(&word.to_le_bytes());
    }

    for fixup in &tail_b_fixups {
        let target_offset = lookup_function_offset(&function_offsets, &fixup.target_name)
            .ok_or_else(|| {
                RasError::EncodingError(format!(
                    "B (tail) target function '{}' not found. Available: {:?}",
                    fixup.target_name,
                    function_offsets.keys().collect::<Vec<_>>()
                ))
            })?;

        let from_pc = fixup.patch_location;
        let delta = target_offset as i64 - from_pc as i64;
        if delta % 4 != 0 {
            return Err(RasError::EncodingError(format!(
                "Unaligned B (tail) target delta {} (from={}, to={})",
                delta, from_pc, target_offset
            )));
        }
        let imm26 = delta / 4;
        if !(-(1i64 << 25)..(1i64 << 25)).contains(&imm26) {
            return Err(RasError::EncodingError(format!(
                "B (tail) target out of range (delta={} bytes)",
                delta
            )));
        }
        let word = 0x1400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF);
        if fixup.patch_location + 4 > code.len() {
            return Err(RasError::EncodingError(format!(
                "B (tail) patch location out of bounds: {} (len={})",
                fixup.patch_location,
                code.len()
            )));
        }
        code[fixup.patch_location..fixup.patch_location + 4].copy_from_slice(&word.to_le_bytes());
    }

    Ok((code, function_offsets))
}

// Encoding functions extracted from backup file

/// Encode STP 64-bit pre-index instruction
/// stp Xt, Xt2, [Xn|SP, #imm]!

/// Encode LDP 64-bit post-index instruction
/// ldp Xt, Xt2, [Xn|SP], #imm

/// Encode AArch64 prologue
fn encode_prologue_aarch64(stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    // stp x29, x30, [sp, #-16]!
    let stp = enc_stp_pre_64(29, 30, 31, -16)?;
    code.extend_from_slice(&stp.to_le_bytes());

    // add x29, sp, #0   (aka mov x29, sp)
    let mov_fp = 0x9100_03FDu32;
    code.extend_from_slice(&mov_fp.to_le_bytes());

    // SUB sp, sp, #<stack_size> (allocate stack frame)
    // Ensure stack is 16-byte aligned (AAPCS64 requirement)
    // The prologue already saved x29, x30 (16 bytes), maintaining alignment
    if stack_size > 0 {
        // Ensure stack_size is 16-byte aligned (AAPCS64 requirement)
        let aligned_size = (stack_size + 15) & !15;

        if aligned_size > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "Stack size {} (aligned: {}) too large for single SUB instruction",
                stack_size, aligned_size
            )));
        }
        // sub sp, sp, #aligned
        // Encoding matches clang/llvm-mc:
        //   sub sp, sp, #imm12  => 0xD10003FF | (imm12 << 10)
        let sub_sp = 0xD100_03FFu32 | ((aligned_size as u32) << 10);
        code.extend_from_slice(&sub_sp.to_le_bytes());
    }

    Ok(code)
}

/// Encode AArch64 epilogue
/// Must restore SP before LDP (undo the SUB sp, sp, #aligned_size from prologue)
fn encode_epilogue_aarch64(aligned_stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    if aligned_stack_size > 0 {
        if aligned_stack_size > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "stack restore too large for single ADD: {}",
                aligned_stack_size
            )));
        }
        // add sp, sp, #aligned_stack_size
        // Encoding matches clang/llvm-mc:
        //   add sp, sp, #imm12  => 0x910003FF | (imm12 << 10)
        let add_sp = 0x9100_03FFu32 | ((aligned_stack_size as u32) << 10);
        code.extend_from_slice(&add_sp.to_le_bytes());
    }

    // ldp x29, x30, [sp], #16
    let ldp = enc_ldp_post_64(29, 30, 31, 16)?;
    code.extend_from_slice(&ldp.to_le_bytes());

    Ok(code)
}

fn encode_str_aarch64(src_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let src = parse_register_aarch64(src_reg)?;
    encode_str_imm64(src, base_reg, offset)
}

fn encode_ldr_aarch64(dst_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let dst = parse_register_aarch64(dst_reg)?;
    encode_ldr_imm64(dst, base_reg, offset)
}

fn mir_scalar_ld_kind_aarch64(ty: &lamina_mir::MirType) -> Result<AArch64ScalarLdKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1) => Ok(AArch64ScalarLdKind::I8U),
        MirType::Scalar(ScalarType::I8) => Ok(AArch64ScalarLdKind::I8S),
        MirType::Scalar(ScalarType::I16) => Ok(AArch64ScalarLdKind::I16S),
        MirType::Scalar(ScalarType::I32) => Ok(AArch64ScalarLdKind::I32S),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(AArch64ScalarLdKind::I64),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "AArch64 JIT Load of floating-point type is not supported".into(),
        )),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT Load: unsupported MIR type {:?}",
            ty
        ))),
    }
}

fn mir_scalar_st_kind_aarch64(ty: &lamina_mir::MirType) -> Result<AArch64ScalarStKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Ok(AArch64ScalarStKind::I8),
        MirType::Scalar(ScalarType::I16) => Ok(AArch64ScalarStKind::I16),
        MirType::Scalar(ScalarType::I32) => Ok(AArch64ScalarStKind::I32),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(AArch64ScalarStKind::I64),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "AArch64 JIT Store of floating-point type is not supported".into(),
        )),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT Store: unsupported MIR type {:?}",
            ty
        ))),
    }
}

fn encode_ldr_typed_aarch64(
    dst_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let dst = parse_register_aarch64(dst_reg)?;
    let kind = mir_scalar_ld_kind_aarch64(ty)?;
    encode_ldr_scalar(dst, base_reg, offset, kind)
}

fn encode_str_typed_aarch64(
    src_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let src = parse_register_aarch64(src_reg)?;
    let kind = mir_scalar_st_kind_aarch64(ty)?;
    encode_str_scalar(src, base_reg, offset, kind)
}

/// Encode RET instruction (AArch64)
/// RET Xn = 0xD65F0000 | (n << 5)
fn encode_ret_aarch64(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD65F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// Encode BR instruction (AArch64)
/// BR Xn = 0xD61F0000 | (n << 5)
#[allow(dead_code)]
fn encode_br_aarch64(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD61F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// Encode BLR instruction (AArch64)
/// BLR Xn = 0xD63F0000 | (n << 5)
fn encode_blr_aarch64(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD63F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

fn encode_brk_aarch64(imm16: u16) -> Vec<u8> {
    let w = 0xD420_0000u32 | (((imm16 as u32) & 0xFFFF) << 5);
    w.to_le_bytes().to_vec()
}

fn encode_nop_aarch64() -> Vec<u8> {
    0xD503_201Fu32.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
fn aarch64_apply_i32_offset_to_reg(
    assembler: &mut RasAssembler,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    code: &mut Vec<u8>,
    base_reg: u8,
    offset: i32,
) -> Result<(), RasError> {
    if offset == 0 {
        return Ok(());
    }
    let o = i64::from(offset);
    if o > 0 && o <= 4095 {
        let inst = 0x9100_0000u32
            | (((o as u32) & 0xFFF) << 10)
            | ((base_reg as u32) << 5)
            | (base_reg as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(());
    }
    if (-4096..0).contains(&o) {
        let imm = (-o) as u32;
        let inst =
            0xD100_0000u32 | ((imm & 0xFFF) << 10) | ((base_reg as u32) << 5) | (base_reg as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(());
    }
    let imm_reg_str = reg_alloc.alloc_scratch().unwrap_or("x14");
    let imm_reg = parse_register_aarch64(imm_reg_str)?;
    materialize_operand_aarch64(
        assembler,
        &Operand::Immediate(Immediate::I64(o)),
        imm_reg,
        stack_slots,
        reg_alloc,
        code,
        stack_size,
    )?;
    code.extend_from_slice(&encode_add_rrr_aarch64(base_reg, base_reg, imm_reg));
    reg_alloc.free_scratch(imm_reg_str);
    Ok(())
}

/// MSUB Xd, Xn, Xm, Xa: Xd = Xa - (Xn * Xm)

/// ADD Xd, Xn, Xm (no shift)

/// AND Xd, Xn, Xm (no shift)

/// SBFM Xd, Xn, #immr, #imms (64-bit); used for sign-extension of low (imms+1) bits.
fn encode_sbfm64_aarch64(rd: u8, rn: u8, immr: u8, imms: u8) -> Vec<u8> {
    let w = 0x9340_0000u32
        | ((immr as u32) << 16)
        | ((imms as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

enum ShiftVKind {
    Lsl,
    Lsr,
    Asr,
}

/// LSLV / LSRV / ASRV Xd, Xn, Xm (64-bit variable shift; shift amount uses low 6 bits of Xm)
fn encode_shiftv_aarch64(which: ShiftVKind, rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let op = match which {
        ShiftVKind::Lsl => 0x9AC02000u32,
        ShiftVKind::Lsr => 0x9AC02400u32,
        ShiftVKind::Asr => 0x9AC02800u32,
    };
    let w = op | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

/// MSUB Wd, Wn, Wm, Wa

fn encode_shiftv_aarch64_w(which: ShiftVKind, rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let op = match which {
        ShiftVKind::Lsl => 0x1AC0_2000u32,
        ShiftVKind::Lsr => 0x1AC0_2400u32,
        ShiftVKind::Asr => 0x1AC0_2800u32,
    };
    let w = op | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
fn aarch64_load_base_gpr(
    base: &lamina_mir::Register,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
) -> Result<(u8, Option<&'static str>), RasError> {
    match base {
        Register::Virtual(vreg) => {
            let base_offset = stack_slots.get(vreg).ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for base register: {:?}", base))
            })?;
            let scratch = reg_alloc.alloc_scratch().unwrap_or("x11");
            code.extend_from_slice(&encode_ldr_aarch64(scratch, 29, *base_offset)?);
            let rn = parse_register_aarch64(scratch)?;
            Ok((rn, Some(scratch)))
        }
        Register::Physical(p) => {
            let rn = parse_register_aarch64(p.name)?;
            Ok((rn, None))
        }
    }
}

#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
fn aarch64_emit_indexed_address(
    _assembler: &mut RasAssembler,
    base: &lamina_mir::Register,
    index: &lamina_mir::Register,
    log2_scale: u16,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
    out_addr_scratch: &'static str,
) -> Result<(), RasError> {
    fn mov_x_dst_src(dst: u8, src: u8) -> Vec<u8> {
        let w = 0xAA00_03E0u32 | ((src as u32) << 16) | (dst as u32);
        w.to_le_bytes().to_vec()
    }

    let base_s = reg_alloc.alloc_scratch().unwrap_or("x13");
    let idx_s = reg_alloc.alloc_scratch().unwrap_or("x14");
    let base_r = parse_register_aarch64(base_s)?;
    let idx_r = parse_register_aarch64(idx_s)?;
    let out_r = parse_register_aarch64(out_addr_scratch)?;

    match base {
        Register::Virtual(v) => {
            let off = stack_slots.get(v).ok_or_else(|| {
                RasError::EncodingError("indexed address: missing base stack slot".into())
            })?;
            code.extend_from_slice(&encode_ldr_aarch64(base_s, 29, *off)?);
        }
        Register::Physical(p) => {
            let br = parse_register_aarch64(p.name)?;
            if br != base_r {
                code.extend_from_slice(&mov_x_dst_src(base_r, br));
            }
        }
    }

    match index {
        Register::Virtual(v) => {
            let off = stack_slots.get(v).ok_or_else(|| {
                RasError::EncodingError("indexed address: missing index stack slot".into())
            })?;
            code.extend_from_slice(&encode_ldr_aarch64(idx_s, 29, *off)?);
        }
        Register::Physical(p) => {
            let ir = parse_register_aarch64(p.name)?;
            if ir != idx_r {
                code.extend_from_slice(&mov_x_dst_src(idx_r, ir));
            }
        }
    }

    if log2_scale > 0 {
        let sh_s = reg_alloc.alloc_scratch().unwrap_or("x15");
        let sh_r = parse_register_aarch64(sh_s)?;
        let movz = 0xD280_0000u32 | (((log2_scale as u32) & 0xFFFF) << 5) | (sh_r as u32);
        code.extend_from_slice(&movz.to_le_bytes());
        code.extend_from_slice(&encode_shiftv_aarch64(ShiftVKind::Lsl, idx_r, idx_r, sh_r));
        reg_alloc.free_scratch(sh_s);
    }

    code.extend_from_slice(&encode_add_rrr_aarch64(out_r, base_r, idx_r));

    reg_alloc.free_scratch(base_s);
    reg_alloc.free_scratch(idx_s);

    Ok(())
}

/// Parse register name to encoding (AArch64)
fn parse_register_aarch64(reg: &str) -> Result<u8, RasError> {
    let reg = reg.trim_start_matches('%');
    match reg {
        "x0" | "w0" => Ok(0),
        "x1" | "w1" => Ok(1),
        "x2" | "w2" => Ok(2),
        "x3" | "w3" => Ok(3),
        "x4" | "w4" => Ok(4),
        "x5" | "w5" => Ok(5),
        "x6" | "w6" => Ok(6),
        "x7" | "w7" => Ok(7),
        "x8" | "w8" => Ok(8),
        "x9" | "w9" => Ok(9),
        "x10" | "w10" => Ok(10),
        "x11" | "w11" => Ok(11),
        "x12" | "w12" => Ok(12),
        "x13" | "w13" => Ok(13),
        "x14" | "w14" => Ok(14),
        "x15" | "w15" => Ok(15),
        "x16" | "w16" | "ip0" => Ok(16),
        "x17" | "w17" | "ip1" => Ok(17),
        "x18" | "w18" => Ok(18),
        "x19" | "w19" => Ok(19),
        "x20" | "w20" => Ok(20),
        "x21" | "w21" => Ok(21),
        "x22" | "w22" => Ok(22),
        "x23" | "w23" => Ok(23),
        "x24" | "w24" => Ok(24),
        "x25" | "w25" => Ok(25),
        "x26" | "w26" => Ok(26),
        "x27" | "w27" => Ok(27),
        "x28" | "w28" => Ok(28),
        "x29" | "w29" | "fp" => Ok(29),
        "x30" | "w30" | "lr" => Ok(30),
        "x31" | "w31" | "sp" | "xzr" | "wzr" => Ok(31),
        _ => Err(RasError::EncodingError(format!(
            "Unknown register: {}",
            reg
        ))),
    }
}

/// Encode MIR instruction to binary (AArch64)
///
/// This reuses the instruction emission logic from mir_codegen/aarch64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
#[allow(dead_code)]
fn encode_mir_instruction_aarch64(
    assembler: &mut RasAssembler,
    inst: &lamina_mir::Instruction,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    func_name: &str,
) -> Result<Vec<u8>, RasError> {
    let mut bl_fixups = Vec::<BlFixup>::new();
    encode_mir_instruction_aarch64_with_context(
        assembler,
        inst,
        reg_alloc,
        stack_slots,
        stack_size,
        None,
        func_name,
        &std::collections::HashMap::new(),
        0,
        &mut bl_fixups,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_mir_instruction_aarch64_with_context(
    assembler: &mut RasAssembler,
    inst: &lamina_mir::Instruction,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    fn_ret_ty: Option<&lamina_mir::MirType>,
    _func_name: &str,
    function_offsets: &std::collections::HashMap<String, usize>,
    current_offset: usize,
    bl_fixups: &mut Vec<BlFixup>,
) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    match inst {
        lamina_mir::Instruction::Ret { value } => {
            if let Some(v) = value {
                if let Some(rt) = fn_ret_ty {
                    materialize_return_value_aarch64(
                        assembler,
                        v,
                        rt,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                } else {
                    materialize_operand_aarch64(
                        assembler,
                        v,
                        0,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                }
            }
        }
        lamina_mir::Instruction::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let lhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let rhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x11");
            let dst_reg_str = reg_alloc.alloc_scratch().unwrap_or("x12");
            let lhs_reg = parse_register_aarch64(lhs_reg_str)?;
            let rhs_reg = parse_register_aarch64(rhs_reg_str)?;
            let dst_reg = parse_register_aarch64(dst_reg_str)?;

            let unsigned_atom = matches!(op, IntBinOp::UDiv | IntBinOp::URem);

            match ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    materialize_operand_aarch64(
                        assembler,
                        lhs,
                        lhs_reg,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                    materialize_operand_aarch64(
                        assembler,
                        rhs,
                        rhs_reg,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;

                    match op {
                        IntBinOp::URem | IntBinOp::SRem => {
                            let quot_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                            let quot_reg = parse_register_aarch64(quot_str)?;
                            let div_op = if matches!(op, IntBinOp::URem) {
                                0x9AC0_0800u32
                            } else {
                                0x9AC0_0C00u32
                            };
                            let div_inst = div_op
                                | ((rhs_reg as u32) << 16)
                                | ((lhs_reg as u32) << 5)
                                | (quot_reg as u32);
                            code.extend_from_slice(&div_inst.to_le_bytes());
                            code.extend_from_slice(&encode_msub_aarch64(
                                dst_reg, quot_reg, rhs_reg, lhs_reg,
                            ));
                            reg_alloc.free_scratch(quot_str);
                        }
                        IntBinOp::Shl => {
                            code.extend_from_slice(&encode_shiftv_aarch64(
                                ShiftVKind::Lsl,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::LShr => {
                            code.extend_from_slice(&encode_shiftv_aarch64(
                                ShiftVKind::Lsr,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::AShr => {
                            code.extend_from_slice(&encode_shiftv_aarch64(
                                ShiftVKind::Asr,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::Add
                        | IntBinOp::Sub
                        | IntBinOp::Mul
                        | IntBinOp::UDiv
                        | IntBinOp::SDiv
                        | IntBinOp::And
                        | IntBinOp::Or
                        | IntBinOp::Xor => {
                            let inst = match op {
                                IntBinOp::Add => 0x8B00_0000u32,
                                IntBinOp::Sub => 0xCB00_0000u32,
                                IntBinOp::Mul => 0x9B00_7C00u32,
                                IntBinOp::UDiv => 0x9AC0_0800u32,
                                IntBinOp::SDiv => 0x9AC0_0C00u32,
                                IntBinOp::And => 0x8A00_0000u32,
                                IntBinOp::Or => 0xAA00_0000u32,
                                IntBinOp::Xor => 0xCA00_0000u32,
                                _ => unreachable!(),
                            } | ((rhs_reg as u32) << 16)
                                | ((lhs_reg as u32) << 5)
                                | (dst_reg as u32);
                            code.extend_from_slice(&inst.to_le_bytes());
                        }
                    }

                    if let Register::Virtual(vreg) = dst
                        && let Some(offset) = stack_slots.get(vreg)
                    {
                        code.extend_from_slice(&encode_str_aarch64(dst_reg_str, 29, *offset)?);
                    }
                }
                MirType::Scalar(
                    ScalarType::I32 | ScalarType::I16 | ScalarType::I8 | ScalarType::I1,
                ) => {
                    materialize_scalar_operand_aarch64_int_binary(
                        assembler,
                        lhs,
                        lhs_reg,
                        ty,
                        unsigned_atom,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                    materialize_scalar_operand_aarch64_int_binary(
                        assembler,
                        rhs,
                        rhs_reg,
                        ty,
                        unsigned_atom,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;

                    match op {
                        IntBinOp::URem | IntBinOp::SRem => {
                            let quot_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                            let quot_reg = parse_register_aarch64(quot_str)?;
                            let div_op = if matches!(op, IntBinOp::URem) {
                                0x1AC0_0800u32
                            } else {
                                0x1AC0_0C00u32
                            };
                            let div_inst = div_op
                                | ((rhs_reg as u32) << 16)
                                | ((lhs_reg as u32) << 5)
                                | (quot_reg as u32);
                            code.extend_from_slice(&div_inst.to_le_bytes());
                            code.extend_from_slice(&encode_msub_aarch64_w(
                                dst_reg, quot_reg, rhs_reg, lhs_reg,
                            ));
                            reg_alloc.free_scratch(quot_str);
                        }
                        IntBinOp::Shl => {
                            code.extend_from_slice(&encode_shiftv_aarch64_w(
                                ShiftVKind::Lsl,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::LShr => {
                            code.extend_from_slice(&encode_shiftv_aarch64_w(
                                ShiftVKind::Lsr,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::AShr => {
                            code.extend_from_slice(&encode_shiftv_aarch64_w(
                                ShiftVKind::Asr,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::Add
                        | IntBinOp::Sub
                        | IntBinOp::Mul
                        | IntBinOp::UDiv
                        | IntBinOp::SDiv
                        | IntBinOp::And
                        | IntBinOp::Or
                        | IntBinOp::Xor => {
                            let inst = match op {
                                IntBinOp::Add => 0x0B00_0000u32,
                                IntBinOp::Sub => 0x4B00_0000u32,
                                IntBinOp::Mul => 0x1B00_7C00u32,
                                IntBinOp::UDiv => 0x1AC0_0800u32,
                                IntBinOp::SDiv => 0x1AC0_0C00u32,
                                IntBinOp::And => 0x0A00_0000u32,
                                IntBinOp::Or => 0x2A00_0000u32,
                                IntBinOp::Xor => 0x4A00_0000u32,
                                _ => unreachable!(),
                            } | ((rhs_reg as u32) << 16)
                                | ((lhs_reg as u32) << 5)
                                | (dst_reg as u32);
                            code.extend_from_slice(&inst.to_le_bytes());
                        }
                    }

                    if matches!(ty, MirType::Scalar(ScalarType::I1)) {
                        let mask_str = reg_alloc.alloc_scratch().unwrap_or("x13");
                        let mask_reg = parse_register_aarch64(mask_str)?;
                        code.extend_from_slice(&mov_imm_to_w_aarch64(mask_reg, 1));
                        code.extend_from_slice(&encode_and_w_rrr_aarch64(
                            dst_reg, dst_reg, mask_reg,
                        ));
                        reg_alloc.free_scratch(mask_str);
                    }

                    if let Register::Virtual(vreg) = dst
                        && let Some(offset) = stack_slots.get(vreg)
                    {
                        code.extend_from_slice(&encode_str_typed_aarch64(
                            dst_reg_str,
                            29,
                            *offset,
                            ty,
                        )?);
                    }
                }
                MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: floating-point MIR type".into(),
                    ));
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT IntBinary: unsupported type {:?}",
                        ty
                    )));
                }
            }

            reg_alloc.free_scratch(lhs_reg_str);
            reg_alloc.free_scratch(rhs_reg_str);
            reg_alloc.free_scratch(dst_reg_str);
        }
        lamina_mir::Instruction::Load { dst, addr, ty, .. } => {
            let tmp_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");

            match addr {
                AddressMode::BaseOffset { base, offset } => {
                    let (base_reg, opt_scratch) =
                        aarch64_load_base_gpr(base, stack_slots, reg_alloc, &mut code)?;
                    code.extend_from_slice(&encode_ldr_typed_aarch64(
                        tmp_reg_str,
                        base_reg,
                        i32::from(*offset),
                        ty,
                    )?);
                    if let Some(s) = opt_scratch {
                        reg_alloc.free_scratch(s);
                    }
                }
                AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                } => {
                    let log2 = match *scale {
                        1 => 0u16,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => {
                            return Err(RasError::EncodingError(format!(
                                "Unsupported address scale {} (expected 1,2,4,8)",
                                scale
                            )));
                        }
                    };
                    let addr_scratch = reg_alloc.alloc_scratch().unwrap_or("x12");
                    aarch64_emit_indexed_address(
                        assembler,
                        base,
                        index,
                        log2,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                        addr_scratch,
                    )?;
                    let addr_reg = parse_register_aarch64(addr_scratch)?;
                    code.extend_from_slice(&encode_ldr_typed_aarch64(
                        tmp_reg_str,
                        addr_reg,
                        i32::from(*offset),
                        ty,
                    )?);
                    reg_alloc.free_scratch(addr_scratch);
                }
            }

            if matches!(ty, MirType::Scalar(ScalarType::I1)) {
                let tmp_r = parse_register_aarch64(tmp_reg_str)?;
                let m_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                let m_r = parse_register_aarch64(m_str)?;
                materialize_operand_aarch64(
                    assembler,
                    &Operand::Immediate(lamina_mir::Immediate::I64(1)),
                    m_r,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_and_rrr_aarch64(tmp_r, tmp_r, m_r));
                reg_alloc.free_scratch(m_str);
            }

            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg)
            {
                code.extend_from_slice(&encode_str_aarch64(tmp_reg_str, 29, *offset)?);
            }

            reg_alloc.free_scratch(tmp_reg_str);
        }
        lamina_mir::Instruction::Store { src, addr, ty, .. } => {
            // Store from source to memory address
            let src_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let src_reg = parse_register_aarch64(src_reg_str)?;

            materialize_operand_aarch64(
                assembler,
                src,
                src_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;

            match addr {
                AddressMode::BaseOffset { base, offset } => {
                    let (base_reg, opt_scratch) =
                        aarch64_load_base_gpr(base, stack_slots, reg_alloc, &mut code)?;
                    code.extend_from_slice(&encode_str_typed_aarch64(
                        src_reg_str,
                        base_reg,
                        i32::from(*offset),
                        ty,
                    )?);
                    if let Some(s) = opt_scratch {
                        reg_alloc.free_scratch(s);
                    }
                }
                AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                } => {
                    let log2 = match *scale {
                        1 => 0u16,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => {
                            return Err(RasError::EncodingError(format!(
                                "Unsupported address scale {} (expected 1,2,4,8)",
                                scale
                            )));
                        }
                    };
                    let addr_scratch = reg_alloc.alloc_scratch().unwrap_or("x12");
                    aarch64_emit_indexed_address(
                        assembler,
                        base,
                        index,
                        log2,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                        addr_scratch,
                    )?;
                    let addr_reg = parse_register_aarch64(addr_scratch)?;
                    code.extend_from_slice(&encode_str_typed_aarch64(
                        src_reg_str,
                        addr_reg,
                        i32::from(*offset),
                        ty,
                    )?);
                    reg_alloc.free_scratch(addr_scratch);
                }
            }

            reg_alloc.free_scratch(src_reg_str);
        }
        lamina_mir::Instruction::Lea { dst, base, offset } => {
            let base_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let base_reg = parse_register_aarch64(base_reg_str)?;
            materialize_operand_aarch64(
                assembler,
                &Operand::Register(base.clone()),
                base_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            aarch64_apply_i32_offset_to_reg(
                assembler,
                reg_alloc,
                stack_slots,
                stack_size,
                &mut code,
                base_reg,
                *offset,
            )?;
            if let Register::Virtual(vreg) = dst
                && let Some(slot_off) = stack_slots.get(vreg)
            {
                code.extend_from_slice(&encode_str_aarch64(base_reg_str, 29, *slot_off)?);
            }
            reg_alloc.free_scratch(base_reg_str);
        }
        lamina_mir::Instruction::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            let t_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let f_str = reg_alloc.alloc_scratch().unwrap_or("x11");
            let c_str = reg_alloc.alloc_scratch().unwrap_or("x12");
            let d_str = reg_alloc.alloc_scratch().unwrap_or("x13");
            let t_reg = parse_register_aarch64(t_str)?;
            let f_reg = parse_register_aarch64(f_str)?;
            let c_reg = parse_register_aarch64(c_str)?;
            let d_reg = parse_register_aarch64(d_str)?;
            materialize_operand_aarch64(
                assembler,
                false_val,
                f_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand_aarch64(
                assembler,
                true_val,
                t_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand_aarch64(
                assembler,
                &Operand::Register(cond.clone()),
                c_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            let subs_xzr = 0xEB00_001Fu32 | ((31u32) << 16) | ((c_reg as u32) << 5);
            code.extend_from_slice(&subs_xzr.to_le_bytes());
            let csel = 0x9A80_0000u32
                | ((f_reg as u32) << 16)
                | (1u32 << 12)
                | ((t_reg as u32) << 5)
                | (d_reg as u32);
            code.extend_from_slice(&csel.to_le_bytes());
            if let Register::Virtual(vreg) = dst
                && let Some(slot_off) = stack_slots.get(vreg)
            {
                code.extend_from_slice(&encode_str_aarch64(d_str, 29, *slot_off)?);
            }
            reg_alloc.free_scratch(t_str);
            reg_alloc.free_scratch(f_str);
            reg_alloc.free_scratch(c_str);
            reg_alloc.free_scratch(d_str);
        }
        lamina_mir::Instruction::Unreachable => {
            code.extend_from_slice(&encode_brk_aarch64(0));
        }
        lamina_mir::Instruction::SafePoint => {
            code.extend_from_slice(&encode_nop_aarch64());
        }
        lamina_mir::Instruction::Comment { .. }
        | lamina_mir::Instruction::StackMap { .. }
        | lamina_mir::Instruction::PatchPoint { .. } => {}
        lamina_mir::Instruction::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let lhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let rhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x11");
            let dst_reg_str = reg_alloc.alloc_scratch().unwrap_or("x12");
            let lhs_reg = parse_register_aarch64(lhs_reg_str)?;
            let rhs_reg = parse_register_aarch64(rhs_reg_str)?;
            let dst_reg = parse_register_aarch64(dst_reg_str)?;

            materialize_operand_aarch64(
                assembler,
                lhs,
                lhs_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand_aarch64(
                assembler,
                rhs,
                rhs_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;

            let use_subs_w32 = matches!(ty, MirType::Scalar(ScalarType::I32));

            match ty {
                MirType::Scalar(
                    ScalarType::I64
                    | ScalarType::I32
                    | ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I1
                    | ScalarType::Ptr,
                ) => {}
                MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                    return Err(RasError::EncodingError(
                        "IntCmp with floating-point MIR type is invalid".into(),
                    ));
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT IntCmp: unsupported type {:?}",
                        ty
                    )));
                }
            }

            if matches!(
                ty,
                MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I1)
            ) {
                let bits = match ty {
                    MirType::Scalar(ScalarType::I8) => 8u8,
                    MirType::Scalar(ScalarType::I16) => 16u8,
                    MirType::Scalar(ScalarType::I1) => 1u8,
                    _ => {
                        return Err(RasError::EncodingError(
                            "AArch64 JIT IntCmp: internal narrow-type mismatch".into(),
                        ));
                    }
                };
                let imms = bits.saturating_sub(1);
                let signed_cmp = matches!(
                    op,
                    IntCmpOp::SLt | IntCmpOp::SLe | IntCmpOp::SGt | IntCmpOp::SGe
                );
                if signed_cmp {
                    code.extend_from_slice(&encode_sbfm64_aarch64(lhs_reg, lhs_reg, 0, imms));
                    code.extend_from_slice(&encode_sbfm64_aarch64(rhs_reg, rhs_reg, 0, imms));
                } else {
                    let mask_reg_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                    let mask_reg = parse_register_aarch64(mask_reg_str)?;
                    let mask_u64 = (1u64 << u32::from(bits)) - 1;
                    materialize_operand_aarch64(
                        assembler,
                        &Operand::Immediate(Immediate::I64(mask_u64 as i64)),
                        mask_reg,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                    code.extend_from_slice(&encode_and_rrr_aarch64(lhs_reg, lhs_reg, mask_reg));
                    code.extend_from_slice(&encode_and_rrr_aarch64(rhs_reg, rhs_reg, mask_reg));
                    reg_alloc.free_scratch(mask_reg_str);
                }
            }

            let cmp_inst = if use_subs_w32 {
                0x6B00_001Fu32 | ((rhs_reg as u32) << 16) | ((lhs_reg as u32) << 5)
            } else {
                0xEB00_001Fu32 | ((rhs_reg as u32) << 16) | ((lhs_reg as u32) << 5)
            };
            code.extend_from_slice(&cmp_inst.to_le_bytes());

            let cond_code = match op {
                IntCmpOp::Eq => 0b0000u32,
                IntCmpOp::Ne => 0b0001u32,
                IntCmpOp::ULt => 0b0011u32,
                IntCmpOp::ULe => 0b1001u32,
                IntCmpOp::UGt => 0b1000u32,
                IntCmpOp::UGe => 0b0010u32,
                IntCmpOp::SLt => 0b1011u32,
                IntCmpOp::SLe => 0b1101u32,
                IntCmpOp::SGt => 0b1100u32,
                IntCmpOp::SGe => 0b1010u32,
            };
            let inv_cond = cond_code ^ 1;
            let cset_base = if use_subs_w32 {
                0x1A9F_07E0u32
            } else {
                0x9A9F_07E0u32
            };
            let cset_inst = cset_base | (inv_cond << 12) | (dst_reg as u32);
            code.extend_from_slice(&cset_inst.to_le_bytes());

            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg)
            {
                code.extend_from_slice(&encode_str_typed_aarch64(dst_reg_str, 29, *offset, ty)?);
            }

            reg_alloc.free_scratch(lhs_reg_str);
            reg_alloc.free_scratch(rhs_reg_str);
            reg_alloc.free_scratch(dst_reg_str);
        }
        lamina_mir::Instruction::Call { name, args, ret } => {
            let _abi = AArch64ABI::new(assembler.target_os);

            // Materialize arguments into argument registers (x0-x7).
            let arg_regs = AArch64ABI::ARG_REGISTERS;
            for (i, arg) in args.iter().enumerate().take(8) {
                let arg_reg_str = arg_regs[i];
                let arg_reg = parse_register_aarch64(arg_reg_str)?;
                materialize_operand_aarch64(
                    assembler,
                    arg,
                    arg_reg,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
            }

            // Handle stack arguments (args beyond 8).
            let stack_args = if args.len() > 8 { &args[8..] } else { &[] };
            let stack_space = (stack_args.len() * 8 + 15) & !15;
            if stack_space > 0 {
                if stack_space > 0xFFF {
                    return Err(RasError::EncodingError(format!(
                        "Stack space {} too large for single SUB",
                        stack_space
                    )));
                }
                let sub_inst = (((0b1u32 << 31) | (0b1u32 << 30)) | (0b100010u32 << 23))
                    | ((stack_space as u32 & 0xFFF) << 10)
                    | (31u32 << 5)
                    | 31u32;
                code.extend_from_slice(&sub_inst.to_le_bytes());

                for (i, arg) in stack_args.iter().enumerate() {
                    let offset = i * 8;
                    let scratch_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                    let scratch = parse_register_aarch64(scratch_str)?;
                    materialize_operand_aarch64(
                        assembler,
                        arg,
                        scratch,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;

                    code.extend_from_slice(&encode_str_aarch64(
                        scratch_str,
                        31, // sp
                        offset as i32,
                    )?);
                    reg_alloc.free_scratch(scratch_str);
                }
            }

            // External function calls need special handling for JIT.
            // Keep `print(x)` as a special-case intrinsic.
            if name == "print" && args.len() == 1 {
                if stack_space != 0 {
                    return Err(RasError::EncodingError(
                        "print() intrinsic does not support stack-passed args".to_string(),
                    ));
                }
                // (Existing printf-based implementation below.)
                // Resolve printf - try both "printf" and "_printf" on macOS
                // The actual symbol name may vary, but dlsym usually finds "printf"
                let printf_name = "printf";
                let printf_name_alt = "_printf";

                let printf_addr = if let Some(addr) = assembler.function_pointers.get(printf_name) {
                    *addr
                } else if let Some(addr) = assembler.function_pointers.get(printf_name_alt) {
                    *addr
                } else {
                    // Try to resolve "printf" first
                    if assembler.register_function(printf_name).is_err() {
                        // Fallback to "_printf" on macOS
                        if assembler.target_os == lamina_platform::TargetOperatingSystem::MacOS {
                            if let Err(e) = assembler.register_function(printf_name_alt) {
                                return Err(RasError::EncodingError(format!(
                                    "Failed to resolve printf or _printf for print() intrinsic: {}. \
                                     Runtime function resolution may not be available on this system.",
                                    e
                                )));
                            }
                            *assembler
                                .function_pointers
                                .get(printf_name_alt)
                                .ok_or_else(|| {
                                    RasError::EncodingError(format!(
                                        "{} not resolved",
                                        printf_name_alt
                                    ))
                                })?
                        } else {
                            return Err(RasError::EncodingError(format!(
                                "Failed to resolve {} for print() intrinsic. \
                                 Runtime function resolution may not be available on this system.",
                                printf_name
                            )));
                        }
                    } else {
                        *assembler
                            .function_pointers
                            .get(printf_name)
                            .ok_or_else(|| {
                                RasError::EncodingError(format!("{} not resolved", printf_name))
                            })?
                    }
                };
                // Match clang's lowering for macOS AArch64 varargs:
                //   sub sp, sp, #32
                //   str x8, [sp]
                //   x0 = "%lld\\n"
                //   bl printf
                //   add sp, sp, #32
                let home_area_size = 32u32;

                // Allocate home area (keeps SP 16-byte aligned).
                let sub_sp = 0xD100_03FFu32 | ((home_area_size & 0xFFF) << 10);
                code.extend_from_slice(&sub_sp.to_le_bytes());

                // Spill the variadic integer argument to the home area at [sp].
                materialize_operand_aarch64(
                    assembler,
                    &args[0],
                    8, // x8
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_str_aarch64("x8", 31, 0)?);

                // Load format string pointer into x0.
                let fmt_ptr = PRINT_I64_FORMAT.as_ptr() as u64;
                materialize_operand_aarch64(
                    assembler,
                    &lamina_mir::Operand::Immediate(lamina_mir::Immediate::I64(fmt_ptr as i64)),
                    0, // x0
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;

                // Load printf address into x16 and call via BLR.
                materialize_operand_aarch64(
                    assembler,
                    &lamina_mir::Operand::Immediate(lamina_mir::Immediate::I64(printf_addr as i64)),
                    16, // x16
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_blr_aarch64(16)?);

                // Restore SP.
                let add_sp = 0x9100_03FFu32 | ((home_area_size & 0xFFF) << 10);
                code.extend_from_slice(&add_sp.to_le_bytes());
            } else {
                // Internal direct call (BL). Emit placeholder and patch later.
                let is_internal = function_offsets.contains_key(name)
                    || (name.starts_with('@') && function_offsets.contains_key(&name[1..]))
                    || (!name.starts_with('@')
                        && function_offsets.contains_key(&format!("@{}", name)));
                if !is_internal {
                    return Err(RasError::EncodingError(format!(
                        "External function call '{}' requires runtime resolution (not implemented for AArch64 JIT)",
                        name
                    )));
                }

                let bl_pc = current_offset + code.len();
                code.extend_from_slice(&0x9400_0000u32.to_le_bytes()); // BL <target> (patched)
                bl_fixups.push(BlFixup {
                    patch_location: bl_pc,
                    target_name: name.clone(),
                });

                if stack_space > 0 {
                    let add_inst = ((0b1u32 << 31) | (0b100010u32 << 23))
                        | ((stack_space as u32 & 0xFFF) << 10)
                        | (31u32 << 5)
                        | 31u32;
                    code.extend_from_slice(&add_inst.to_le_bytes());
                }

                if let Some(dst) = ret
                    && let Register::Virtual(vreg) = dst
                    && let Some(offset) = stack_slots.get(vreg)
                {
                    code.extend_from_slice(&encode_str_aarch64(
                        "x0", 29, // x29 (FP)
                        *offset,
                    )?);
                }
            }
        }
        lamina_mir::Instruction::Jmp { .. } => {
            return Err(RasError::EncodingError(
                "Jmp must be handled at block/terminator level (bug: reached instruction encoder)"
                    .to_string(),
            ));
        }
        lamina_mir::Instruction::Br { .. } => {
            return Err(RasError::EncodingError(
                "Br must be handled at block/terminator level (bug: reached instruction encoder)"
                    .to_string(),
            ));
        }
        _ => {
            return Err(RasError::EncodingError(format!(
                "MIR instruction not yet implemented: {:?}",
                inst
            )));
        }
    }

    Ok(code)
}

#[cfg(feature = "encoder")]
fn aarch64_narrow_imm_i64(bits: u8, raw: i64, signed_style: bool) -> i64 {
    if bits == 0 || bits >= 64 {
        return raw;
    }
    let bits_u = u32::from(bits);
    let mask = (1i64 << bits_u) - 1;
    let v = raw & mask;
    if signed_style {
        let sign_bit = 1i64 << (bits_u - 1);
        if (v & sign_bit) != 0 { v | !mask } else { v }
    } else {
        v
    }
}

#[cfg(feature = "encoder")]
fn mov_imm_to_w_aarch64(dst: u8, v: i32) -> Vec<u8> {
    let u = v as u32;
    let lo = u & 0xFFFF;
    let hi = (u >> 16) & 0xFFFF;
    let mut b = Vec::new();
    let movz = 0x5280_0000u32 | (lo << 5) | (dst as u32);
    b.extend_from_slice(&movz.to_le_bytes());
    if hi != 0 {
        let movk = 0x7280_0000u32 | (1u32 << 21) | (hi << 5) | (dst as u32);
        b.extend_from_slice(&movk.to_le_bytes());
    }
    b
}

/// Narrow integer scalar operands for W-width `IntBinary` (I32 / I16 / I8 / I1).
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
fn materialize_scalar_operand_aarch64_int_binary(
    _assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    dst_reg: u8,
    ty: &lamina_mir::MirType,
    unsigned_atom: bool,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
) -> Result<(), RasError> {
    let dst_w = format!("w{}", dst_reg);
    let signed_style = !unsigned_atom;

    match op {
        Operand::Immediate(imm) => {
            let raw = match imm {
                Immediate::I8(x) => *x as i64,
                Immediate::I16(x) => *x as i64,
                Immediate::I32(x) => *x as i64,
                Immediate::I64(x) => *x,
                Immediate::F32(_) | Immediate::F64(_) => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: floating-point immediate".into(),
                    ));
                }
            };
            let v: i32 = match ty {
                MirType::Scalar(ScalarType::I32) => raw as i32,
                MirType::Scalar(ScalarType::I16) => {
                    aarch64_narrow_imm_i64(16, raw, signed_style) as i32
                }
                MirType::Scalar(ScalarType::I8) => {
                    aarch64_narrow_imm_i64(8, raw, signed_style) as i32
                }
                MirType::Scalar(ScalarType::I1) => (raw & 1) as i32,
                _ => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: internal type mismatch for immediate".into(),
                    ));
                }
            };
            code.extend_from_slice(&mov_imm_to_w_aarch64(dst_reg, v));
        }
        Operand::Register(Register::Virtual(vreg)) => {
            let off = stack_slots.get(vreg).copied().ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for virtual register: {:?}", vreg))
            })?;
            match ty {
                MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: floating-point MIR type".into(),
                    ));
                }
                MirType::Scalar(_) => {
                    code.extend_from_slice(&encode_ldr_typed_aarch64(&dst_w, 29, off, ty)?);
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT IntBinary: unsupported type {:?}",
                        ty
                    )));
                }
            }
        }
        Operand::Register(Register::Physical(_)) => {
            return Err(RasError::EncodingError(
                "Physical register operands not yet fully supported".to_string(),
            ));
        }
    }
    Ok(())
}

/// Place the return value in `x0` per AAPCS64 (narrow integers are sign-extended in `x0`).
#[cfg(feature = "encoder")]
fn materialize_return_value_aarch64(
    assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    ret_ty: &lamina_mir::MirType,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<(), RasError> {
    match op {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = stack_slots.get(vreg).copied().ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for virtual register: {:?}", vreg))
            })?;
            match ret_ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    code.extend_from_slice(&encode_ldr_typed_aarch64("x0", 29, offset, ret_ty)?);
                }
                MirType::Scalar(
                    ScalarType::I32 | ScalarType::I16 | ScalarType::I8 | ScalarType::I1,
                ) => {
                    code.extend_from_slice(&encode_ldr_typed_aarch64("w0", 29, offset, ret_ty)?);
                    code.extend_from_slice(&encode_sbfm64_aarch64(0, 0, 0, 31));
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT: unsupported return type {:?}",
                        ret_ty
                    )));
                }
            }
            Ok(())
        }
        Operand::Immediate(_) | Operand::Register(Register::Physical(_)) => {
            materialize_operand_aarch64(assembler, op, 0, stack_slots, reg_alloc, code, stack_size)
        }
    }
}

/// Materialize an operand into a register (AArch64)
/// Loads from stack slot if operand is a virtual register, or moves immediate
#[cfg(feature = "encoder")]
fn materialize_operand_aarch64(
    _assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    dst_reg: u8,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
) -> Result<(), RasError> {
    match op {
        Operand::Immediate(imm) => {
            // MOVZ/MOVK sequence (64-bit).
            let imm_val: u64 = match imm {
                Immediate::I8(v) => *v as i64 as u64,
                Immediate::I16(v) => *v as i64 as u64,
                Immediate::I32(v) => *v as i64 as u64,
                Immediate::I64(v) => *v as u64,
                _ => {
                    return Err(RasError::EncodingError(
                        "Floating-point immediates not yet supported".to_string(),
                    ));
                }
            };

            // Fast path: single MOVZ.
            if imm_val <= 0xFFFF {
                let movz = 0xD280_0000u32 | ((imm_val as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movz.to_le_bytes());
                return Ok(());
            }

            // General path: MOVZ for low 16 bits + MOVK for remaining chunks.
            let chunk0 = (imm_val & 0xFFFF) as u16;
            let chunk1 = ((imm_val >> 16) & 0xFFFF) as u16;
            let chunk2 = ((imm_val >> 32) & 0xFFFF) as u16;
            let chunk3 = ((imm_val >> 48) & 0xFFFF) as u16;

            let movz = 0xD280_0000u32 | ((chunk0 as u32) << 5) | (dst_reg as u32);
            code.extend_from_slice(&movz.to_le_bytes());

            if chunk1 != 0 {
                let movk =
                    0xF280_0000u32 | (0b01u32 << 21) | ((chunk1 as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
            if chunk2 != 0 {
                let movk =
                    0xF280_0000u32 | (0b10u32 << 21) | ((chunk2 as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
            if chunk3 != 0 {
                let movk =
                    0xF280_0000u32 | (0b11u32 << 21) | ((chunk3 as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
        }
        Operand::Register(Register::Virtual(vreg)) => {
            // Load from stack slot
            // FrameMap stack slots are FP-relative offsets (negative for locals).
            if let Some(offset) = stack_slots.get(vreg) {
                let dst_reg_str = format!("x{}", dst_reg);
                code.extend_from_slice(&encode_ldr_aarch64(
                    &dst_reg_str,
                    29, // x29 (FP)
                    *offset,
                )?);
            } else {
                return Err(RasError::EncodingError(format!(
                    "No stack slot for virtual register: {:?}",
                    vreg
                )));
            }
        }
        Operand::Register(Register::Physical(_)) => {
            // Physical registers are already in place, but we need to move to dst
            // For now, assume it's already correct (this is a simplification)
            return Err(RasError::EncodingError(
                "Physical register operands not yet fully supported".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "encoder"))]
mod aarch64_jit_tail_call_tests {
    use super::*;
    use lamina_mir::{
        block::Block,
        function::{Function, Parameter, Signature},
        instruction::{
            AddressMode, Immediate, Instruction, IntBinOp, IntCmpOp, MemoryAttrs, Operand,
        },
        module::Module,
        register::{Register, VirtualReg},
        types::{MirType, ScalarType},
    };
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn subslice_present(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    fn compile_single_i32_binop_a64(op: IntBinOp) -> Vec<u8> {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("t")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty.clone()),
                Parameter::new(b.clone(), i32_ty.clone()),
            ])
            .with_return(i32_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i32_ty.clone(),
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("t");
        module.add_function(f);
        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        compile_mir_aarch64_function(&mut asm, &module, None)
            .expect("compile")
            .0
    }

    const N: usize = 15;

    fn fifteen_arg_tail_module() -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let params: Vec<Parameter> = (0..N)
            .map(|i| Parameter::new(Register::Virtual(VirtualReg::gpr(i as u32)), i64_ty.clone()))
            .collect();

        let sink_sig = Signature::new("sink")
            .with_params(params.clone())
            .with_return(i64_ty.clone());
        let mut sink_fn = Function::new(sink_sig);
        let mut sink_entry = Block::new("entry");
        sink_entry.push(Instruction::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(
                (N - 1) as u32,
            )))),
        });
        sink_fn.add_block(sink_entry);

        let tail_sig = Signature::new("forward15")
            .with_params(params)
            .with_return(i64_ty.clone());
        let mut tail_fn = Function::new(tail_sig);
        let mut tail_entry = Block::new("entry");
        let args: Vec<Operand> = (0..N)
            .map(|i| Operand::Register(Register::Virtual(VirtualReg::gpr(i as u32))))
            .collect();
        tail_entry.push(Instruction::TailCall {
            name: "sink".to_string(),
            args,
        });
        tail_fn.add_block(tail_entry);

        let mut m = Module::new("jit_tail");
        m.add_function(sink_fn);
        m.add_function(tail_fn);
        m
    }

    #[test]
    fn compile_mir_aarch64_fifteen_arg_tail_call_encodes() {
        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let module = fifteen_arg_tail_module();
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 64,
            "expected substantial machine code for 15-arg tail call, got {} bytes",
            code.len()
        );
    }

    /// Stack-passed call arguments beyond ~64 used a broken STR immediate in the JIT encoder.
    #[test]
    fn compile_mir_aarch64_eighty_arg_call_encodes() {
        const M: usize = 80;
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let params: Vec<Parameter> = (0..M)
            .map(|i| Parameter::new(Register::Virtual(VirtualReg::gpr(i as u32)), i64_ty.clone()))
            .collect();

        let sink_sig = Signature::new("sink")
            .with_params(params.clone())
            .with_return(i64_ty.clone());
        let mut sink_fn = Function::new(sink_sig);
        let mut sink_entry = Block::new("entry");
        sink_entry.push(Instruction::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(
                (M - 1) as u32,
            )))),
        });
        sink_fn.add_block(sink_entry);

        let caller_sig = Signature::new("caller").with_return(i64_ty.clone());
        let mut caller_fn = Function::new(caller_sig);
        let mut caller_entry = Block::new("entry");
        let args: Vec<Operand> = (0..M)
            .map(|i| Operand::Immediate(Immediate::I64(i as i64)))
            .collect();
        caller_entry.push(Instruction::Call {
            name: "sink".to_string(),
            args,
            ret: Some(Register::Virtual(VirtualReg::gpr(200))),
        });
        caller_entry.push(Instruction::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(200)))),
        });
        caller_fn.add_block(caller_entry);

        let mut module = Module::new("jit_many_stack_args");
        module.add_function(sink_fn);
        module.add_function(caller_fn);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 200,
            "expected non-trivial encoding for 80-arg call, got {} bytes",
            code.len()
        );
    }

    /// BaseOffset loads/stores used to mask the displacement with 0x1FF while allowing up to 0xFFF,
    /// corrupting addresses for offsets in [255, 0xFFF] (same failure mode as multi-arg call STR).
    #[test]
    fn compile_mir_aarch64_load_scaled_base_offset_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let base_v = Register::Virtual(VirtualReg::gpr(0));
        let dst_v = Register::Virtual(VirtualReg::gpr(1));
        let sig = Signature::new("load1024")
            .with_params(vec![Parameter::new(base_v.clone(), i64_ty.clone())])
            .with_return(i64_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Load {
            ty: i64_ty.clone(),
            dst: dst_v.clone(),
            addr: AddressMode::BaseOffset {
                base: base_v.clone(),
                offset: 1024,
            },
            attrs: MemoryAttrs::default(),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(dst_v.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("load_off");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 32,
            "expected prologue + scaled LDR + epilogue, got {} bytes",
            code.len()
        );
    }

    #[test]
    fn compile_mir_aarch64_lea_small_offset_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let base_v = Register::Virtual(VirtualReg::gpr(0));
        let out_v = Register::Virtual(VirtualReg::gpr(1));
        let sig = Signature::new("lea_fn")
            .with_params(vec![Parameter::new(base_v.clone(), i64_ty.clone())])
            .with_return(i64_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Lea {
            dst: out_v.clone(),
            base: base_v.clone(),
            offset: 24,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out_v.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("lea_mod");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 24,
            "expected prologue + LEA path + epilogue, got {} bytes",
            code.len()
        );
    }

    #[test]
    fn compile_mir_aarch64_select_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let cond_v = Register::Virtual(VirtualReg::gpr(0));
        let out_v = Register::Virtual(VirtualReg::gpr(1));
        let sig = Signature::new("sel_fn")
            .with_params(vec![Parameter::new(cond_v.clone(), i64_ty.clone())])
            .with_return(i64_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Select {
            ty: i64_ty.clone(),
            dst: out_v.clone(),
            cond: cond_v.clone(),
            true_val: Operand::Immediate(Immediate::I64(7)),
            false_val: Operand::Immediate(Immediate::I64(3)),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out_v.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("sel_mod");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 32,
            "expected prologue + select + epilogue, got {} bytes",
            code.len()
        );
    }

    #[test]
    fn compile_mir_aarch64_intcmp_i32_encodes() {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("cmp32")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty.clone()),
                Parameter::new(b.clone(), i32_ty.clone()),
            ])
            .with_return(i32_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntCmp {
            op: IntCmpOp::Eq,
            ty: i32_ty.clone(),
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("cmp32_mod");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 40,
            "expected prologue + 32-bit subs/cset + epilogue, got {} bytes",
            code.len()
        );
        assert!(
            subslice_present(&code, &[0x3f, 0x01, 0x0a, 0x6b]),
            "subs wzr,w9,w10"
        );
        assert!(
            subslice_present(&code, &[0xeb, 0x17, 0x9f, 0x1a]),
            "cset w11,eq (CSINC w form)"
        );
    }

    #[test]
    fn compile_mir_aarch64_int_binop_i32_add_encodes() {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("add32")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty.clone()),
                Parameter::new(b.clone(), i32_ty.clone()),
            ])
            .with_return(i32_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i32_ty.clone(),
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("add32_mod");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_aarch64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 48,
            "expected prologue + W-width add + typed STR + epilogue, got {} bytes",
            code.len()
        );
        assert!(code.ends_with(&[0xC0, 0x03, 0x5F, 0xD6]));
    }

    /// Instruction words use scratch `w9`..=`w11` (`A64RegAlloc` order), not `w10`..=`w12`.
    #[test]
    fn jit_a64_i32_udiv_w_encoding_present() {
        let code = compile_single_i32_binop_a64(IntBinOp::UDiv);
        assert!(subslice_present(&code, &[0x2B, 0x09, 0xCA, 0x1A]));
    }

    #[test]
    fn jit_a64_i32_urem_w_udiv_then_msub_encoding_present() {
        let code = compile_single_i32_binop_a64(IntBinOp::URem);
        assert!(subslice_present(&code, &[0x2c, 0x09, 0xca, 0x1a]));
        assert!(subslice_present(&code, &[0x8b, 0xa5, 0x0a, 0x1b]));
    }

    #[test]
    fn jit_a64_i32_sdiv_w_encoding_present() {
        let code = compile_single_i32_binop_a64(IntBinOp::SDiv);
        assert!(subslice_present(&code, &[0x2B, 0x0D, 0xCA, 0x1A]));
    }

    #[test]
    fn jit_a64_i32_shl_w_encoding_present() {
        let code = compile_single_i32_binop_a64(IntBinOp::Shl);
        assert!(subslice_present(&code, &[0x2B, 0x21, 0xCA, 0x1A]));
    }

    #[test]
    fn jit_a64_i32_lshr_w_encoding_present() {
        let code = compile_single_i32_binop_a64(IntBinOp::LShr);
        assert!(subslice_present(&code, &[0x2B, 0x25, 0xCA, 0x1A]));
    }

    #[test]
    fn jit_a64_i32_ashr_w_encoding_present() {
        let code = compile_single_i32_binop_a64(IntBinOp::AShr);
        assert!(subslice_present(&code, &[0x2B, 0x29, 0xCA, 0x1A]));
    }
}
