//! AArch64 binary code generation
//!
//! This module handles compilation of MIR to AArch64 binary machine code.

mod encode;
mod operand;

use encode::*;
use operand::*;

use crate::encoder::aarch64::fixed::{encode_and_rrr, encode_msub, encode_msub_w};
use crate::{assembler::core::RasAssembler, error::RasError};
use lamina_codegen::aarch64::{A64RegAlloc, AArch64ABI, FrameMap};
use lamina_mir::{
    AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, IntBinOp, IntCmpOp, MirType,
    Operand, Register, ScalarType,
};

#[cfg(feature = "encoder")]
#[derive(Debug, Clone)]
struct BlFixup {
    /// Offset in the final code buffer where the BL instruction word begins.
    patch_location: usize,
    /// Target function name as referenced by MIR (may include or omit '@').
    target_name: String,
    /// Set when the word is `ADR` of this register instead of `BL`.
    adr_register: Option<u8>,
}

#[cfg(feature = "encoder")]
static PRINT_I64_FORMAT: [u8; 6] = *b"%lld\n\0";

/// Compile MIR to binary for AArch64
#[cfg(feature = "encoder")]
pub fn compile_mir_aarch64_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    _function_name: Option<&str>,
) -> Result<(Vec<u8>, std::collections::HashMap<String, usize>), RasError> {
    let _abi = AArch64ABI::new(assembler.target_os);
    allocate_globals(assembler, module)?;
    let mut code = Vec::new();
    let mut function_offsets: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    // Always compile all functions (needed for internal function calls)
    // A function with no blocks is an extern declaration, its code lives in a loaded library.
    let mut all_function_names: Vec<String> = module
        .functions
        .iter()
        .filter(|(_, func)| !func.blocks.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    all_function_names.sort();

    // Pre-calculate function offsets (estimate) to handle recursive calls
    let mut estimated_sizes: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for func_name in &all_function_names {
        if let Some(func) = module.functions.get(func_name) {
            let inst_count = func
                .blocks
                .iter()
                .map(|b| b.instructions.len())
                .sum::<usize>();
            let estimated_size = 16 + (inst_count * 4) + 12 + 4;
            estimated_sizes.insert(func_name.clone(), estimated_size);
        }
    }

    let total_est: usize = estimated_sizes.values().copied().sum();
    code.reserve(total_est.saturating_add(256));

    let mut current_estimate = 0;
    for func_name in &all_function_names {
        function_offsets.insert(func_name.clone(), current_estimate);
        current_estimate += estimated_sizes.get(func_name).copied().unwrap_or(100);
    }

    // Track internal direct-call fixups (BL) and patch them after final function offsets are known.
    let mut bl_fixups: Vec<BlFixup> = Vec::new();
    // Tail calls use unconditional B (same patch shape as BL: 26-bit PC-relative).
    let mut tail_b_fixups: Vec<BlFixup> = Vec::new();

    for func_name in &all_function_names {
        let func = module.functions.get(func_name).ok_or_else(|| {
            RasError::EncodingError(format!("Function '{}' not found in module", func_name))
        })?;
        reject_unmodelled_abi(func)?;
        // A vector register gets a 16-byte slot, so the frame is laid out from a copy that says so.
        let vector_sized = vector_frame(func)?;
        let func = vector_sized.as_ref().unwrap_or(func);

        function_offsets.insert(func_name.clone(), code.len());
        let mut reg_alloc = A64RegAlloc::new();
        let frame = FrameMap::from_function(func);
        let mut stack_slots: std::collections::HashMap<lamina_mir::VirtualReg, i32> =
            std::collections::HashMap::new();

        for (reg, offset) in &frame.slots {
            if let Register::Virtual(vreg) = reg {
                stack_slots.insert(*vreg, *offset);
            }
        }

        let stack_size = frame.frame_size as usize;
        let homes = assign_register_homes(func, &mut stack_slots);

        // AAPCS64 wants the frame 16-byte aligned, and the prologue saves x29, x30 on top of it.
        let aligned_stack_size = (stack_size + 8 * homes.len() + 15) & !15;

        let prologue = encode_prologue(aligned_stack_size)?;
        code.extend_from_slice(&prologue);
        for (index, home) in homes.iter().enumerate() {
            code.extend_from_slice(&encode_str(
                &format!("x{home}"),
                29,
                home_save_offset(stack_size, index),
            )?);
        }

        // The encoder materializes a virtual register by loading its stack slot, so the ABI
        // argument registers have to be spilled to those slots first.
        if !func.sig.params.is_empty() {
            let arg_regs = AArch64ABI::ARG_REGISTERS;

            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        let str_bytes =
                            encode_str(arg_regs[index], 29 /* x29 (FP) */, *slot_off)?;
                        code.extend_from_slice(&str_bytes);
                    } else {
                        // Handle stack arguments (AAPCS64: stack args start at caller's [sp, #0])
                        // After prologue: stp x29,x30,[sp,#-16]! then mov x29,sp
                        // Caller's stack args are now at [x29, #16] (16 bytes for saved fp/lr)
                        // First stack arg (arg8) is at [x29, #16], second at [x29, #24], etc.
                        let stack_arg_index = index - arg_regs.len();
                        let caller_off = (16 + stack_arg_index * 8) as i32; // 16 for saved fp/lr
                        let ldr1 = encode_ldr("x10", 29, caller_off)?;
                        code.extend_from_slice(&ldr1);
                        let str1 = encode_str("x10", 29, *slot_off)?;
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
                let inst_bytes = encode_mir_instruction_with_context(
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
                            materialize_return_value(
                                assembler,
                                v,
                                rt,
                                &stack_slots,
                                &mut reg_alloc,
                                &mut code,
                                aligned_stack_size,
                            )?;
                        } else {
                            materialize_operand(
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
                    let cond_reg_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                    let cond_reg = parse_register(cond_reg_str)?;
                    materialize_operand(
                        assembler,
                        &lamina_mir::Operand::Register(cond.clone()),
                        cond_reg,
                        &stack_slots,
                        &mut reg_alloc,
                        &mut code,
                        aligned_stack_size,
                    )?;
                    reg_alloc.free_scratch(cond_reg_str);

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

                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::B {
                            target: false_target.clone(),
                        },
                    });
                }
                lamina_mir::Instruction::Switch {
                    value,
                    cases,
                    default,
                } => {
                    let value_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                    let case_str = reg_alloc.alloc_scratch().unwrap_or("x10");
                    let value_reg = parse_register(value_str)?;
                    let case_reg = parse_register(case_str)?;
                    materialize_operand(
                        assembler,
                        &lamina_mir::Operand::Register(value.clone()),
                        value_reg,
                        &stack_slots,
                        &mut reg_alloc,
                        &mut code,
                        aligned_stack_size,
                    )?;
                    for (case_value, target) in cases {
                        materialize_operand(
                            assembler,
                            &lamina_mir::Operand::Immediate(Immediate::I64(*case_value)),
                            case_reg,
                            &stack_slots,
                            &mut reg_alloc,
                            &mut code,
                            aligned_stack_size,
                        )?;
                        // cmp value, case
                        let cmp =
                            0xEB00_001Fu32 | ((case_reg as u32) << 16) | ((value_reg as u32) << 5);
                        code.extend_from_slice(&cmp.to_le_bytes());
                        // b.ne over the branch below, B reaches further than B.cond
                        code.extend_from_slice(&0x5400_0041u32.to_le_bytes());
                        let patch_location = code.len();
                        code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                        branch_fixups.push(BranchFixup {
                            patch_location,
                            kind: BranchFixupKind::B {
                                target: target.clone(),
                            },
                        });
                    }
                    reg_alloc.free_scratch(case_str);
                    reg_alloc.free_scratch(value_str);
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::B {
                            target: default.clone(),
                        },
                    });
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
                        let dst = parse_register(arg_regs[i])?;
                        materialize_operand(
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
                        materialize_operand(
                            assembler,
                            arg,
                            11,
                            &stack_slots,
                            &mut reg_alloc,
                            &mut code,
                            aligned_stack_size,
                        )?;
                        code.extend_from_slice(&encode_str("x11", 29, caller_off)?);
                    }

                    code.extend_from_slice(&encode_home_restores(&homes, stack_size)?);
                    let epilogue_tail = encode_epilogue(aligned_stack_size)?;
                    code.extend_from_slice(&epilogue_tail);

                    let b_pc = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                    tail_b_fixups.push(BlFixup {
                        patch_location: b_pc,
                        target_name: name.clone(),
                        adr_register: None,
                    });
                }
                lamina_mir::Instruction::Unreachable => {
                    code.extend_from_slice(&encode_brk(0));
                }
                other => {
                    return Err(RasError::EncodingError(format!(
                        "Unexpected terminator in block '{}': {:?}",
                        block.label, other
                    )));
                }
            }
        }

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

        code.extend_from_slice(&encode_home_restores(&homes, stack_size)?);
        let epilogue = encode_epilogue(aligned_stack_size)?;
        code.extend_from_slice(&epilogue);

        // RET instruction (x30 is LR)
        code.extend_from_slice(&encode_ret(30)?);
    }

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
        let word = match fixup.adr_register {
            Some(register) => {
                if !(-(1i64 << 20)..(1i64 << 20)).contains(&delta) {
                    return Err(RasError::EncodingError(format!(
                        "ADR target out of range (delta={} bytes)",
                        delta
                    )));
                }
                let (low, high) = ((delta & 3) as u32, ((delta >> 2) & 0x7_FFFF) as u32);
                0x1000_0000u32 | (low << 29) | (high << 5) | u32::from(register)
            }
            None => 0x9400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF),
        };
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

/// Encode MIR instruction to binary (AArch64)
#[cfg(feature = "encoder")]
#[allow(dead_code)]
fn encode_mir_instruction(
    assembler: &mut RasAssembler,
    inst: &lamina_mir::Instruction,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    func_name: &str,
) -> Result<Vec<u8>, RasError> {
    let mut bl_fixups = Vec::<BlFixup>::new();
    encode_mir_instruction_with_context(
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
fn encode_mir_instruction_with_context(
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
                    materialize_return_value(
                        assembler,
                        v,
                        rt,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                } else {
                    materialize_operand(
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
            let lhs_reg = parse_register(lhs_reg_str)?;
            let rhs_reg = parse_register(rhs_reg_str)?;
            let dst_reg = parse_register(dst_reg_str)?;

            let unsigned_atom = matches!(op, IntBinOp::UDiv | IntBinOp::URem);

            match ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    materialize_operand(
                        assembler,
                        lhs,
                        lhs_reg,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                    materialize_operand(
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
                            let quot_reg = parse_register(quot_str)?;
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
                            code.extend_from_slice(&encode_msub(
                                dst_reg, quot_reg, rhs_reg, lhs_reg,
                            ));
                            reg_alloc.free_scratch(quot_str);
                        }
                        IntBinOp::Shl => {
                            code.extend_from_slice(&encode_shiftv(
                                ShiftVKind::Lsl,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::LShr => {
                            code.extend_from_slice(&encode_shiftv(
                                ShiftVKind::Lsr,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::AShr => {
                            code.extend_from_slice(&encode_shiftv(
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
                        code.extend_from_slice(&encode_str(dst_reg_str, 29, *offset)?);
                    }
                }
                MirType::Scalar(
                    ScalarType::I32 | ScalarType::I16 | ScalarType::I8 | ScalarType::I1,
                ) => {
                    materialize_scalar_operand_int_binary(
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
                    materialize_scalar_operand_int_binary(
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

                    // A narrow value is held sign extended. What reads it as unsigned, and a
                    // shift amount, must only see the bits of the type.
                    let bits = match ty {
                        MirType::Scalar(ScalarType::I8) => Some(8u8),
                        MirType::Scalar(ScalarType::I16) => Some(16u8),
                        _ => None,
                    };
                    if let Some(bits) = bits {
                        if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
                            code.extend_from_slice(&encode_ubfm64(lhs_reg, lhs_reg, 0, bits - 1));
                        }
                        if matches!(op, IntBinOp::UDiv | IntBinOp::URem) {
                            code.extend_from_slice(&encode_ubfm64(rhs_reg, rhs_reg, 0, bits - 1));
                        }
                        if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr) {
                            let top = bits.trailing_zeros() as u8 - 1;
                            code.extend_from_slice(&encode_ubfm64(rhs_reg, rhs_reg, 0, top));
                        }
                    }

                    match op {
                        IntBinOp::URem | IntBinOp::SRem => {
                            let quot_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                            let quot_reg = parse_register(quot_str)?;
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
                            code.extend_from_slice(&encode_msub_w(
                                dst_reg, quot_reg, rhs_reg, lhs_reg,
                            ));
                            reg_alloc.free_scratch(quot_str);
                        }
                        IntBinOp::Shl => {
                            code.extend_from_slice(&encode_shiftv_w(
                                ShiftVKind::Lsl,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::LShr => {
                            code.extend_from_slice(&encode_shiftv_w(
                                ShiftVKind::Lsr,
                                dst_reg,
                                lhs_reg,
                                rhs_reg,
                            ));
                        }
                        IntBinOp::AShr => {
                            code.extend_from_slice(&encode_shiftv_w(
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

                    if let Register::Virtual(vreg) = dst
                        && let Some(offset) = stack_slots.get(vreg)
                    {
                        code.extend_from_slice(&encode_str_typed(dst_reg_str, 29, *offset, ty)?);
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
        lamina_mir::Instruction::Load { ty, dst, addr, .. } if ty.is_vector() => {
            encode_vector_load(
                assembler,
                dst,
                addr,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::Store { ty, src, addr, .. } if ty.is_vector() => {
            encode_vector_store(
                assembler,
                src,
                addr,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::VectorOp {
            op,
            ty,
            dst,
            operands,
        } => {
            encode_vector_op(
                assembler,
                *op,
                ty,
                dst,
                operands,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::Load { dst, addr, ty, .. } => {
            let tmp_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");

            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    let (base_reg, opt_scratch) =
                        load_base_gpr(base, stack_slots, reg_alloc, &mut code)?;
                    code.extend_from_slice(&encode_ldr_typed(
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
                    ..
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
                    emit_indexed_address(
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
                    let addr_reg = parse_register(addr_scratch)?;
                    code.extend_from_slice(&encode_ldr_typed(
                        tmp_reg_str,
                        addr_reg,
                        i32::from(*offset),
                        ty,
                    )?);
                    reg_alloc.free_scratch(addr_scratch);
                }
            }

            if matches!(ty, MirType::Scalar(ScalarType::I1)) {
                let tmp_r = parse_register(tmp_reg_str)?;
                let m_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                let m_r = parse_register(m_str)?;
                materialize_operand(
                    assembler,
                    &Operand::Immediate(lamina_mir::Immediate::I64(1)),
                    m_r,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_and_rrr(tmp_r, tmp_r, m_r));
                reg_alloc.free_scratch(m_str);
            }

            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg)
            {
                code.extend_from_slice(&encode_str(tmp_reg_str, 29, *offset)?);
            }

            reg_alloc.free_scratch(tmp_reg_str);
        }
        lamina_mir::Instruction::Store { src, addr, ty, .. } => {
            let src_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let src_reg = parse_register(src_reg_str)?;

            materialize_operand(
                assembler,
                src,
                src_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;

            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    let (base_reg, opt_scratch) =
                        load_base_gpr(base, stack_slots, reg_alloc, &mut code)?;
                    code.extend_from_slice(&encode_str_typed(
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
                    ..
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
                    emit_indexed_address(
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
                    let addr_reg = parse_register(addr_scratch)?;
                    code.extend_from_slice(&encode_str_typed(
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
            let base_reg = parse_register(base_reg_str)?;
            materialize_operand(
                assembler,
                &Operand::Register(base.clone()),
                base_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            apply_i32_offset_to_reg(
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
                code.extend_from_slice(&encode_str(base_reg_str, 29, *slot_off)?);
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
            let t_reg = parse_register(t_str)?;
            let f_reg = parse_register(f_str)?;
            let c_reg = parse_register(c_str)?;
            let d_reg = parse_register(d_str)?;
            materialize_operand(
                assembler,
                false_val,
                f_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand(
                assembler,
                true_val,
                t_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand(
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
                code.extend_from_slice(&encode_str(d_str, 29, *slot_off)?);
            }
            reg_alloc.free_scratch(t_str);
            reg_alloc.free_scratch(f_str);
            reg_alloc.free_scratch(c_str);
            reg_alloc.free_scratch(d_str);
        }
        lamina_mir::Instruction::Unreachable => {
            code.extend_from_slice(&encode_brk(0));
        }
        lamina_mir::Instruction::SafePoint => {
            code.extend_from_slice(&encode_nop());
        }
        lamina_mir::Instruction::Comment { .. }
        | lamina_mir::Instruction::StackMap { .. }
        | lamina_mir::Instruction::PatchPoint { .. } => {}
        lamina_mir::Instruction::IntOverflow {
            op,
            ty,
            signed,
            dst,
            lhs,
            rhs,
        } => {
            encode_int_overflow(
                assembler,
                *op,
                ty,
                *signed,
                dst,
                lhs,
                rhs,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::AtomicLoad {
            ty,
            dst,
            addr,
            ordering,
        } => {
            encode_atomic_load(
                assembler,
                ty,
                dst,
                addr,
                *ordering,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::AtomicStore {
            ty,
            src,
            addr,
            ordering,
        } => {
            encode_atomic_store(
                assembler,
                ty,
                src,
                addr,
                *ordering,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::AtomicBinary {
            op,
            ty,
            dst,
            addr,
            value,
            ..
        } => {
            encode_atomic_binary(
                assembler,
                *op,
                ty,
                dst,
                addr,
                value,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::AtomicCompareExchange {
            ty,
            dst,
            addr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
        } => {
            encode_atomic_compare_exchange(
                assembler,
                ty,
                dst,
                addr,
                expected,
                desired,
                *success_ordering,
                *failure_ordering,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
        lamina_mir::Instruction::IntUnary { op, ty, dst, src } => {
            encode_int_unary(
                assembler,
                *op,
                ty,
                dst,
                src,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
        }
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
            let lhs_reg = parse_register(lhs_reg_str)?;
            let rhs_reg = parse_register(rhs_reg_str)?;
            let dst_reg = parse_register(dst_reg_str)?;

            materialize_operand(
                assembler,
                lhs,
                lhs_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand(
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
                    code.extend_from_slice(&encode_sbfm64(lhs_reg, lhs_reg, 0, imms));
                    code.extend_from_slice(&encode_sbfm64(rhs_reg, rhs_reg, 0, imms));
                } else {
                    let mask_reg_str = reg_alloc.alloc_scratch().unwrap_or("x14");
                    let mask_reg = parse_register(mask_reg_str)?;
                    let mask_u64 = (1u64 << u32::from(bits)) - 1;
                    materialize_operand(
                        assembler,
                        &Operand::Immediate(Immediate::I64(mask_u64 as i64)),
                        mask_reg,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;
                    code.extend_from_slice(&encode_and_rrr(lhs_reg, lhs_reg, mask_reg));
                    code.extend_from_slice(&encode_and_rrr(rhs_reg, rhs_reg, mask_reg));
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
                code.extend_from_slice(&encode_str_typed(dst_reg_str, 29, *offset, ty)?);
            }

            reg_alloc.free_scratch(lhs_reg_str);
            reg_alloc.free_scratch(rhs_reg_str);
            reg_alloc.free_scratch(dst_reg_str);
        }
        lamina_mir::Instruction::Call { name, args, ret } => {
            if assembler.target_os == lamina_platform::TargetOperatingSystem::MacOS
                && encode_macos_io(
                    assembler,
                    name,
                    args,
                    ret.as_ref(),
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?
            {
                return Ok(code);
            }

            let _abi = AArch64ABI::new(assembler.target_os);

            let internal = function_offsets.contains_key(name)
                || (name.starts_with('@') && function_offsets.contains_key(&name[1..]))
                || (!name.starts_with('@') && function_offsets.contains_key(&format!("@{}", name)));
            if !internal && name != "print" {
                encode_extern_call(
                    assembler,
                    name,
                    args,
                    ret.as_ref(),
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                return Ok(code);
            }

            let arg_regs = AArch64ABI::ARG_REGISTERS;
            for (i, arg) in args.iter().enumerate().take(8) {
                let arg_reg_str = arg_regs[i];
                let arg_reg = parse_register(arg_reg_str)?;
                materialize_operand(
                    assembler,
                    arg,
                    arg_reg,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
            }

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
                    let scratch = parse_register(scratch_str)?;
                    materialize_operand(
                        assembler,
                        arg,
                        scratch,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;

                    code.extend_from_slice(&encode_str(scratch_str, 31, offset as i32)?);
                    reg_alloc.free_scratch(scratch_str);
                }
            }

            // Keep `print(x)` as a special-case intrinsic.
            if name == "print" && args.len() == 1 {
                if stack_space != 0 {
                    return Err(RasError::EncodingError(
                        "print() intrinsic does not support stack-passed args".to_string(),
                    ));
                }
                // macOS prefixes C symbols with an underscore, so "_printf" is the fallback name.
                let printf_name = "printf";
                let printf_name_alt = "_printf";

                let printf_addr = if let Some(addr) = assembler.function_pointers.get(printf_name) {
                    *addr
                } else if let Some(addr) = assembler.function_pointers.get(printf_name_alt) {
                    *addr
                } else {
                    if assembler.register_function(printf_name).is_err() {
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

                materialize_operand(
                    assembler,
                    &args[0],
                    8,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_str("x8", 31, 0)?);

                let fmt_ptr = PRINT_I64_FORMAT.as_ptr() as u64;
                materialize_operand(
                    assembler,
                    &lamina_mir::Operand::Immediate(lamina_mir::Immediate::I64(fmt_ptr as i64)),
                    0,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;

                materialize_operand(
                    assembler,
                    &lamina_mir::Operand::Immediate(lamina_mir::Immediate::I64(printf_addr as i64)),
                    16,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_blr(16)?);

                let add_sp = 0x9100_03FFu32 | ((home_area_size & 0xFFF) << 10);
                code.extend_from_slice(&add_sp.to_le_bytes());
            } else {
                let is_internal = function_offsets.contains_key(name)
                    || (name.starts_with('@') && function_offsets.contains_key(&name[1..]))
                    || (!name.starts_with('@')
                        && function_offsets.contains_key(&format!("@{}", name)));
                if !is_internal {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT: extern call to @{} reached the internal call path",
                        name
                    )));
                }

                let bl_pc = current_offset + code.len();
                code.extend_from_slice(&0x9400_0000u32.to_le_bytes()); // BL <target> (patched)
                bl_fixups.push(BlFixup {
                    patch_location: bl_pc,
                    target_name: name.clone(),
                    adr_register: None,
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
                    code.extend_from_slice(&encode_str("x0", 29, *offset)?);
                }
            }
        }
        lamina_mir::Instruction::Copy { ty, dst, src }
        | lamina_mir::Instruction::Bitcast {
            to: ty, dst, src, ..
        } => {
            if ty.is_vector() || ty.size_bytes(lamina_platform::PointerWidth::Bits64) > 8 {
                return Err(RasError::EncodingError(format!(
                    "AArch64 JIT: copy of {:?} not supported",
                    ty
                )));
            }
            // A slot holds the whole value, so moving all 8 bytes keeps every width.
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            materialize_operand(
                assembler,
                src,
                parse_register(tmp)?,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::StackAddr { dst, slot } => {
            let offset = stack_slots.get(slot).copied().ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for virtual register: {:?}", slot))
            })?;
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            let tmp_reg = parse_register(tmp)?;
            // add tmp, x29, #0
            let mov_fp = 0x9100_03A0u32 | (tmp_reg as u32);
            code.extend_from_slice(&mov_fp.to_le_bytes());
            apply_i32_offset_to_reg(
                assembler,
                reg_alloc,
                stack_slots,
                stack_size,
                &mut code,
                tmp_reg,
                offset,
            )?;
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::IntConvert {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let width = lamina_platform::PointerWidth::Bits64;
            let (from_bytes, to_bytes) = (from.size_bytes(width), to.size_bytes(width));
            if from_bytes > 8 || to_bytes > 8 {
                return Err(RasError::EncodingError(format!(
                    "AArch64 JIT: IntConvert {:?} to {:?} not supported",
                    from, to
                )));
            }
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            let tmp_reg = parse_register(tmp)?;
            materialize_operand(
                assembler,
                src,
                tmp_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            let kept = from_bytes.min(to_bytes);
            if kept < 8 {
                let top_bit = (kept * 8 - 1) as u8;
                code.extend_from_slice(&if *signed {
                    encode_sbfm64(tmp_reg, tmp_reg, 0, top_bit)
                } else {
                    encode_ubfm64(tmp_reg, tmp_reg, 0, top_bit)
                });
            }
            if to_bytes < 8 {
                let top_bit = (to_bytes * 8 - 1) as u8;
                code.extend_from_slice(&encode_sbfm64(tmp_reg, tmp_reg, 0, top_bit));
            }
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::SymbolAddress { dst, symbol } => {
            let address = match assembler.function_pointers.get(symbol.as_str()) {
                Some(address) => *address,
                None => {
                    if function_offsets.contains_key(symbol) {
                        let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
                        bl_fixups.push(BlFixup {
                            patch_location: current_offset + code.len(),
                            target_name: symbol.clone(),
                            adr_register: Some(parse_register(tmp)?),
                        });
                        code.extend_from_slice(&0x1000_0000u32.to_le_bytes());
                        store_to_slot(&mut code, tmp, dst, stack_slots)?;
                        reg_alloc.free_scratch(tmp);
                        return Ok(code);
                    }
                    resolve_symbol(assembler, symbol)?
                }
            };
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            materialize_operand(
                assembler,
                &Operand::Immediate(Immediate::I64(address as i64)),
                parse_register(tmp)?,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::FloatBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let prefix = float_register_prefix(ty)?;
            let left = reg_alloc.alloc_scratch().unwrap_or("x10");
            let right = reg_alloc.alloc_scratch().unwrap_or("x11");
            let (left_reg, right_reg) = (parse_register(left)?, parse_register(right)?);
            materialize_operand(
                assembler,
                lhs,
                left_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand(
                assembler,
                rhs,
                right_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            encode_float_in(assembler, &mut code, prefix, 0, left_reg)?;
            encode_float_in(assembler, &mut code, prefix, 1, right_reg)?;
            let mnemonic = match op {
                FloatBinOp::FAdd => "fadd",
                FloatBinOp::FSub => "fsub",
                FloatBinOp::FMul => "fmul",
                FloatBinOp::FDiv => "fdiv",
            };
            encode_text(
                assembler,
                &mut code,
                &format!("{mnemonic} {prefix}0, {prefix}0, {prefix}1"),
            )?;
            encode_float_out(assembler, &mut code, prefix, left_reg)?;
            store_to_slot(&mut code, left, dst, stack_slots)?;
            reg_alloc.free_scratch(left);
            reg_alloc.free_scratch(right);
        }
        lamina_mir::Instruction::FloatUnary { op, ty, dst, src } => {
            let prefix = float_register_prefix(ty)?;
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            let tmp_reg = parse_register(tmp)?;
            materialize_operand(
                assembler,
                src,
                tmp_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            encode_float_in(assembler, &mut code, prefix, 0, tmp_reg)?;
            let mnemonic = match op {
                FloatUnOp::FNeg => "fneg",
                FloatUnOp::FSqrt => "fsqrt",
            };
            encode_text(
                assembler,
                &mut code,
                &format!("{mnemonic} {prefix}0, {prefix}0"),
            )?;
            encode_float_out(assembler, &mut code, prefix, tmp_reg)?;
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::FloatCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let prefix = float_register_prefix(ty)?;
            let left = reg_alloc.alloc_scratch().unwrap_or("x10");
            let right = reg_alloc.alloc_scratch().unwrap_or("x11");
            let (left_reg, right_reg) = (parse_register(left)?, parse_register(right)?);
            materialize_operand(
                assembler,
                lhs,
                left_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            materialize_operand(
                assembler,
                rhs,
                right_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            encode_float_in(assembler, &mut code, prefix, 0, left_reg)?;
            encode_float_in(assembler, &mut code, prefix, 1, right_reg)?;
            encode_text(assembler, &mut code, &format!("fcmp {prefix}0, {prefix}1"))?;
            // Same conditions as the AOT backend: an unordered compare is false, ne true.
            let condition = match op {
                FloatCmpOp::Eq => "eq",
                FloatCmpOp::Ne => "ne",
                FloatCmpOp::Lt => "mi",
                FloatCmpOp::Le => "ls",
                FloatCmpOp::Gt => "gt",
                FloatCmpOp::Ge => "ge",
            };
            encode_text(assembler, &mut code, &format!("cset {left}, {condition}"))?;
            store_to_slot(&mut code, left, dst, stack_slots)?;
            reg_alloc.free_scratch(left);
            reg_alloc.free_scratch(right);
        }
        lamina_mir::Instruction::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let prefix = float_register_prefix(to)?;
            let from_bytes = from.size_bytes(lamina_platform::PointerWidth::Bits64);
            if from_bytes > 8 {
                return Err(RasError::EncodingError(format!(
                    "AArch64 JIT: IntToFloat from {:?} not supported",
                    from
                )));
            }
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            let tmp_reg = parse_register(tmp)?;
            materialize_operand(
                assembler,
                src,
                tmp_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            if from_bytes < 8 {
                let top_bit = (from_bytes * 8 - 1) as u8;
                code.extend_from_slice(&if *signed {
                    encode_sbfm64(tmp_reg, tmp_reg, 0, top_bit)
                } else {
                    encode_ubfm64(tmp_reg, tmp_reg, 0, top_bit)
                });
            }
            let mnemonic = if *signed { "scvtf" } else { "ucvtf" };
            encode_text(
                assembler,
                &mut code,
                &format!("{mnemonic} {prefix}0, {tmp}"),
            )?;
            encode_float_out(assembler, &mut code, prefix, tmp_reg)?;
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::FloatConvert { from, to, dst, src } => {
            let (from_prefix, to_prefix) =
                (float_register_prefix(from)?, float_register_prefix(to)?);
            let tmp = reg_alloc.alloc_scratch().unwrap_or("x10");
            let tmp_reg = parse_register(tmp)?;
            materialize_operand(
                assembler,
                src,
                tmp_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            if from_prefix != to_prefix {
                encode_float_in(assembler, &mut code, from_prefix, 0, tmp_reg)?;
                encode_text(
                    assembler,
                    &mut code,
                    &format!("fcvt {to_prefix}0, {from_prefix}0"),
                )?;
                encode_float_out(assembler, &mut code, to_prefix, tmp_reg)?;
            }
            store_to_slot(&mut code, tmp, dst, stack_slots)?;
            reg_alloc.free_scratch(tmp);
        }
        lamina_mir::Instruction::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let prefix = float_register_prefix(from)?;
            let bits = to.size_bytes(lamina_platform::PointerWidth::Bits64) * 8;
            if bits > 64 {
                return Err(RasError::EncodingError(format!(
                    "AArch64 JIT: FloatToInt to {:?} not supported",
                    to
                )));
            }
            let value = reg_alloc.alloc_scratch().unwrap_or("x10");
            let limit = reg_alloc.alloc_scratch().unwrap_or("x11");
            let (value_reg, limit_reg) = (parse_register(value)?, parse_register(limit)?);
            materialize_operand(
                assembler,
                src,
                value_reg,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            encode_float_in(assembler, &mut code, prefix, 0, value_reg)?;
            let mnemonic = if *signed { "fcvtzs" } else { "fcvtzu" };
            let (v32, l32) = (format!("w{value_reg}"), format!("w{limit_reg}"));
            if bits == 64 {
                encode_text(
                    assembler,
                    &mut code,
                    &format!("{mnemonic} {value}, {prefix}0"),
                )?;
            } else {
                encode_text(
                    assembler,
                    &mut code,
                    &format!("{mnemonic} {v32}, {prefix}0"),
                )?;
                if bits < 32 {
                    let (minimum, maximum) = if *signed {
                        (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
                    } else {
                        (0, (1i64 << bits) - 1)
                    };
                    code.extend_from_slice(&mov_imm_to_w(limit_reg, maximum as i32));
                    encode_text(assembler, &mut code, &format!("cmp {v32}, {l32}"))?;
                    let keep = if *signed { "le" } else { "ls" };
                    encode_text(
                        assembler,
                        &mut code,
                        &format!("csel {v32}, {v32}, {l32}, {keep}"),
                    )?;
                    if *signed {
                        code.extend_from_slice(&mov_imm_to_w(limit_reg, minimum as i32));
                        encode_text(assembler, &mut code, &format!("cmp {v32}, {l32}"))?;
                        encode_text(
                            assembler,
                            &mut code,
                            &format!("csel {v32}, {v32}, {l32}, ge"),
                        )?;
                    }
                }
                let top_bit = (bits - 1) as u8;
                code.extend_from_slice(&if *signed {
                    encode_sbfm64(value_reg, value_reg, 0, top_bit)
                } else {
                    encode_ubfm64(value_reg, value_reg, 0, top_bit)
                });
            }
            store_to_slot(&mut code, value, dst, stack_slots)?;
            reg_alloc.free_scratch(value);
            reg_alloc.free_scratch(limit);
        }
        lamina_mir::Instruction::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed,
            ret,
            ret_ty,
        } => {
            // The callee may be a JIT function, which takes every argument by position in a
            // general register, or C, which takes floats in the float registers. Both agree
            // only when the integers come before the floats, so each float is passed in both.
            let first_float = arg_types.iter().position(|ty| ty.is_float());
            let int_after_float =
                first_float.is_some_and(|first| arg_types[first..].iter().any(|ty| !ty.is_float()));
            if variadic_fixed.is_some()
                || args.len() > 8
                || arg_types.len() != args.len()
                || int_after_float
                || arg_types.iter().any(|ty| ty.is_vector())
                || ret_ty.is_some_and(|ty| ty.is_float() || ty.is_vector())
            {
                return Err(RasError::EncodingError(
                    "AArch64 JIT: indirect call with a float before an integer, vector, variadic or stack arguments not supported"
                        .to_string(),
                ));
            }
            for (index, arg) in args.iter().enumerate() {
                materialize_operand(
                    assembler,
                    arg,
                    index as u8,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                if let Some(first) = first_float
                    && index >= first
                {
                    let prefix = float_register_prefix(&arg_types[index])?;
                    encode_float_in(
                        assembler,
                        &mut code,
                        prefix,
                        (index - first) as u8,
                        index as u8,
                    )?;
                }
            }
            materialize_operand(
                assembler,
                target,
                16,
                stack_slots,
                reg_alloc,
                &mut code,
                stack_size,
            )?;
            code.extend_from_slice(&encode_blr(16)?);
            if let Some(dst) = ret {
                if let Some(ty) = ret_ty {
                    let bytes = ty.size_bytes(lamina_platform::PointerWidth::Bits64);
                    if bytes > 8 {
                        return Err(RasError::EncodingError(format!(
                            "AArch64 JIT: indirect call returning {:?} not supported",
                            ty
                        )));
                    }
                    if bytes < 8 {
                        code.extend_from_slice(&encode_sbfm64(0, 0, 0, (bytes * 8 - 1) as u8));
                    }
                }
                store_to_slot(&mut code, "x0", dst, stack_slots)?;
            }
        }
        lamina_mir::Instruction::Fence { ordering } => {
            encode_fence(*ordering, &mut code);
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
                Parameter::new(a.clone(), i32_ty),
                Parameter::new(b.clone(), i32_ty),
            ])
            .with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i32_ty,
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
            .map(|i| Parameter::new(Register::Virtual(VirtualReg::gpr(i as u32)), i64_ty))
            .collect();

        let sink_sig = Signature::new("sink")
            .with_params(params.clone())
            .with_return(i64_ty);
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
            .with_return(i64_ty);
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
        let words: Vec<u32> = code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let tail = words
            .iter()
            .position(|w| w >> 26 == 0b000101)
            .expect("tail call branch");
        assert_eq!(
            words[tail - 1],
            0xa8c1_7bfd,
            "ldp x29, x30, [sp], #16 runs before the tail branch"
        );
    }

    #[test]
    fn compile_mir_aarch64_eighty_arg_call_encodes() {
        const M: usize = 80;
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let params: Vec<Parameter> = (0..M)
            .map(|i| Parameter::new(Register::Virtual(VirtualReg::gpr(i as u32)), i64_ty))
            .collect();

        let sink_sig = Signature::new("sink")
            .with_params(params.clone())
            .with_return(i64_ty);
        let mut sink_fn = Function::new(sink_sig);
        let mut sink_entry = Block::new("entry");
        sink_entry.push(Instruction::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(
                (M - 1) as u32,
            )))),
        });
        sink_fn.add_block(sink_entry);

        let caller_sig = Signature::new("caller").with_return(i64_ty);
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
        let words: Vec<u32> = code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let bl = words
            .iter()
            .position(|w| w >> 26 == 0b100101)
            .expect("bl to the callee");
        // 80 arguments, 8 of them in registers: the other 72 take 576 bytes of stack.
        assert!(
            words[..bl].contains(&0xd109_03ff),
            "sub sp, sp, #576 before the call"
        );
        assert_eq!(
            words[bl + 1],
            0x9109_03ff,
            "add sp, sp, #576 after the call"
        );
    }

    #[test]
    fn compile_mir_aarch64_load_scaled_base_offset_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let base_v = Register::Virtual(VirtualReg::gpr(0));
        let dst_v = Register::Virtual(VirtualReg::gpr(1));
        let sig = Signature::new("load1024")
            .with_params(vec![Parameter::new(base_v.clone(), i64_ty)])
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Load {
            ty: i64_ty,
            dst: dst_v.clone(),
            addr: AddressMode::BaseOffset {
                base: base_v.clone(),
                offset: 1024,
                address_space: Default::default(),
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
        // ldr x9, [x10, #1024]: the unsigned-offset form scales imm12 by 8, so 1024 is 128.
        assert!(subslice_present(&code, &0xf942_0149u32.to_le_bytes()));
    }

    #[test]
    fn compile_mir_aarch64_lea_small_offset_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let base_v = Register::Virtual(VirtualReg::gpr(0));
        let out_v = Register::Virtual(VirtualReg::gpr(1));
        let sig = Signature::new("lea_fn")
            .with_params(vec![Parameter::new(base_v.clone(), i64_ty)])
            .with_return(i64_ty);
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
        // add x9, x9, #24
        assert!(subslice_present(&code, &0x9100_6129u32.to_le_bytes()));
    }

    #[test]
    fn compile_mir_aarch64_physical_return_operand_is_rejected() {
        use lamina_mir::{PhysicalReg, RegisterClass};

        let i64_ty = MirType::Scalar(ScalarType::I64);
        let mut f = Function::new(Signature::new("pick").with_return(i64_ty));
        let mut entry = Block::new("entry");
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(Register::Physical(PhysicalReg::new(
                "x1",
                RegisterClass::Gpr,
            )))),
        });
        f.add_block(entry);
        let mut m = Module::new("phys_ret");
        m.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let err = compile_mir_aarch64_function(&mut asm, &m, None)
            .expect_err("physical register operand must not encode");
        assert!(format!("{err:?}").contains("Physical register"));
    }

    #[test]
    fn compile_mir_aarch64_select_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let cond_v = Register::Virtual(VirtualReg::gpr(0));
        let out_v = Register::Virtual(VirtualReg::gpr(1));
        let sig = Signature::new("sel_fn")
            .with_params(vec![Parameter::new(cond_v.clone(), i64_ty)])
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Select {
            ty: i64_ty,
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
        // subs xzr, x11, xzr then csel x12, x9, x10, ne
        assert!(subslice_present(&code, &0xeb1f_017fu32.to_le_bytes()));
        assert!(subslice_present(&code, &0x9a8a_112cu32.to_le_bytes()));
    }

    #[test]
    fn compile_mir_aarch64_intcmp_i32_encodes() {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("cmp32")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty),
                Parameter::new(b.clone(), i32_ty),
            ])
            .with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntCmp {
            op: IntCmpOp::Eq,
            ty: i32_ty,
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
                Parameter::new(a.clone(), i32_ty),
                Parameter::new(b.clone(), i32_ty),
            ])
            .with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i32_ty,
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
        assert!(code.ends_with(&[0xC0, 0x03, 0x5F, 0xD6]));
    }

    /// Instruction words use scratch `w9`..=`w11` (`A64RegAlloc` order).
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
