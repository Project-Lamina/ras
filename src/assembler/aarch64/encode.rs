//! AArch64 instruction encoding helpers used when lowering MIR.

use super::operand::materialize_operand;
use crate::assembler::core::RasAssembler;
use crate::encoder::ParsedInstruction;
use crate::encoder::aarch64::fixed::{
    enc_ldp_post_64, enc_stp_pre_64, encode_add_rrr, encode_and_rrr,
};
use crate::encoder::aarch64::ldst_imm64::{
    AArch64ScalarLdKind, AArch64ScalarStKind, encode_ldr_imm64, encode_ldr_scalar,
    encode_str_imm64, encode_str_scalar,
};
use crate::error::RasError;
use lamina_mir::{Immediate, MirType, Operand, Register, ScalarType};

/// `SUB`/`ADD` sp in up to two immediates: the low 12 bits and the next 12 shifted.
fn adjust_sp(code: &mut Vec<u8>, size: usize, low: u32, high: u32) -> Result<(), RasError> {
    if size > 0xFF_FFFF {
        return Err(RasError::EncodingError(format!(
            "Stack size {} too large for AArch64 JIT frame",
            size
        )));
    }
    if size >> 12 != 0 {
        let inst = high | (((size >> 12) as u32) << 10);
        code.extend_from_slice(&inst.to_le_bytes());
    }
    if size & 0xFFF != 0 {
        let inst = low | (((size & 0xFFF) as u32) << 10);
        code.extend_from_slice(&inst.to_le_bytes());
    }
    Ok(())
}

pub(super) fn encode_prologue(stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    let stp = enc_stp_pre_64(29, 30, 31, -16)?;
    code.extend_from_slice(&stp.to_le_bytes());

    // add x29, sp, #0   (aka mov x29, sp)
    let mov_fp = 0x9100_03FDu32;
    code.extend_from_slice(&mov_fp.to_le_bytes());

    // The prologue already saved x29, x30 (16 bytes), maintaining alignment
    if stack_size > 0 {
        let aligned_size = (stack_size + 15) & !15;
        adjust_sp(&mut code, aligned_size, 0xD100_03FF, 0xD140_03FF)?;
    }

    Ok(code)
}

/// Must restore SP before LDP (undo the SUB sp, sp, #aligned_size from prologue)
pub(super) fn encode_epilogue(aligned_stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    if aligned_stack_size > 0 {
        adjust_sp(&mut code, aligned_stack_size, 0x9100_03FF, 0x9140_03FF)?;
    }

    let ldp = enc_ldp_post_64(29, 30, 31, 16)?;
    code.extend_from_slice(&ldp.to_le_bytes());

    Ok(code)
}

/// A virtual register given a register of its own has this in place of a frame offset.
/// Every slot access goes through the helpers below, so that is the one place to look.
const REGISTER_HOME: i32 = 1 << 24;

fn register_home(base_reg: u8, offset: i32) -> Option<u8> {
    (base_reg == 29 && offset >= REGISTER_HOME).then(|| (offset - REGISTER_HOME) as u8)
}

fn encode_mov_x(rd: u8, rm: u8) -> Vec<u8> {
    let word = 0xAA00_03E0u32 | ((rm as u32) << 16) | (rd as u32);
    word.to_le_bytes().to_vec()
}

/// Weight of one access: a block inside a loop runs about this many times more.
const LOOP_WEIGHT: u64 = 8;
const MAX_LOOP_WEIGHT_DEPTH: u32 = 6;

/// Give the most used virtual registers a callee-saved register for the whole function.
/// Each one keeps it alone, so no liveness is needed and calls cannot disturb it.
/// Returns the registers the prologue has to save.
#[cfg(feature = "encoder")]
pub(super) fn assign_register_homes(
    func: &lamina_mir::Function,
    stack_slots: &mut std::collections::HashMap<lamina_mir::VirtualReg, i32>,
) -> Vec<u8> {
    const HOMES: [u8; 10] = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28];
    // A save and a restore are two memory accesses, so fewer uses than that lose.
    const MIN_WEIGHT: u64 = 3;

    let position: std::collections::HashMap<&str, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.label.as_str(), index))
        .collect();
    let mut depth = vec![0u32; func.blocks.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        let targets: Vec<&str> = match block.terminator() {
            Some(lamina_mir::Instruction::Jmp { target }) => vec![target],
            Some(lamina_mir::Instruction::Br {
                true_target,
                false_target,
                ..
            }) => vec![true_target, false_target],
            Some(lamina_mir::Instruction::Switch { cases, default, .. }) => cases
                .iter()
                .map(|(_, target)| target.as_str())
                .chain([default.as_str()])
                .collect(),
            _ => Vec::new(),
        };
        for target in targets {
            if let Some(&head) = position.get(target)
                && head <= index
            {
                for inside in &mut depth[head..=index] {
                    *inside += 1;
                }
            }
        }
    }

    let mut pinned = std::collections::HashSet::new();
    let mut weights: std::collections::HashMap<lamina_mir::VirtualReg, u64> =
        std::collections::HashMap::new();
    for (index, block) in func.blocks.iter().enumerate() {
        let weight = LOOP_WEIGHT.pow(depth[index].min(MAX_LOOP_WEIGHT_DEPTH));
        for inst in &block.instructions {
            if let lamina_mir::Instruction::StackAddr { slot, .. } = &inst.instr {
                pinned.insert(*slot);
            }
            for reg in inst.use_regs().into_iter().chain(inst.def_reg()) {
                if let Register::Virtual(vreg) = reg {
                    *weights.entry(*vreg).or_default() += weight;
                }
            }
        }
    }
    pinned.extend(func.stack_reservations.keys().copied());
    pinned.extend(func.stack_alignments.keys().copied());

    let mut ranked: Vec<(lamina_mir::VirtualReg, u64)> = weights
        .into_iter()
        .filter(|(vreg, weight)| {
            *weight >= MIN_WEIGHT && !pinned.contains(vreg) && stack_slots.contains_key(vreg)
        })
        .collect();
    // Sorted on the register too, the map's order must not change the code.
    ranked.sort_by_key(|(vreg, weight)| (std::cmp::Reverse(*weight), format!("{vreg:?}")));

    let mut used = Vec::new();
    for ((vreg, _), home) in ranked.into_iter().zip(HOMES) {
        stack_slots.insert(vreg, REGISTER_HOME + i32::from(home));
        used.push(home);
    }
    used
}

/// Where the prologue saves the `index`th register it took, under the locals.
pub(super) fn home_save_offset(stack_size: usize, index: usize) -> i32 {
    -((stack_size + 8 * (index + 1)) as i32)
}

pub(super) fn encode_home_restores(used: &[u8], stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    for (index, home) in used.iter().enumerate() {
        code.extend_from_slice(&encode_ldr_imm64(
            *home,
            29,
            home_save_offset(stack_size, index),
        )?);
    }
    Ok(code)
}

pub(super) fn encode_str(src_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let src = parse_register(src_reg)?;
    if let Some(home) = register_home(base_reg, offset) {
        return Ok(encode_mov_x(home, src));
    }
    encode_str_imm64(src, base_reg, offset)
}

pub(super) fn encode_ldr(dst_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let dst = parse_register(dst_reg)?;
    if let Some(home) = register_home(base_reg, offset) {
        return Ok(encode_mov_x(dst, home));
    }
    encode_ldr_imm64(dst, base_reg, offset)
}

pub(super) fn mir_scalar_ld_kind(
    ty: &lamina_mir::MirType,
) -> Result<AArch64ScalarLdKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1) => Ok(AArch64ScalarLdKind::I8U),
        MirType::Scalar(ScalarType::I8) => Ok(AArch64ScalarLdKind::I8S),
        MirType::Scalar(ScalarType::I16) => Ok(AArch64ScalarLdKind::I16S),
        MirType::Scalar(ScalarType::I32) => Ok(AArch64ScalarLdKind::I32S),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(AArch64ScalarLdKind::I64),
        MirType::Scalar(ScalarType::F32) => Ok(AArch64ScalarLdKind::I32S),
        MirType::Scalar(ScalarType::F64) => Ok(AArch64ScalarLdKind::I64),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT Load: unsupported MIR type {:?}",
            ty
        ))),
    }
}

pub(super) fn mir_scalar_st_kind(
    ty: &lamina_mir::MirType,
) -> Result<AArch64ScalarStKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Ok(AArch64ScalarStKind::I8),
        MirType::Scalar(ScalarType::I16) => Ok(AArch64ScalarStKind::I16),
        MirType::Scalar(ScalarType::I32) => Ok(AArch64ScalarStKind::I32),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(AArch64ScalarStKind::I64),
        MirType::Scalar(ScalarType::F32) => Ok(AArch64ScalarStKind::I32),
        MirType::Scalar(ScalarType::F64) => Ok(AArch64ScalarStKind::I64),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT Store: unsupported MIR type {:?}",
            ty
        ))),
    }
}

pub(super) fn encode_ldr_typed(
    dst_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let dst = parse_register(dst_reg)?;
    let kind = mir_scalar_ld_kind(ty)?;
    if let Some(home) = register_home(base_reg, offset) {
        return Ok(match kind {
            AArch64ScalarLdKind::I8S => encode_sbfm64(dst, home, 0, 7),
            AArch64ScalarLdKind::I8U => encode_ubfm64(dst, home, 0, 7),
            AArch64ScalarLdKind::I16S => encode_sbfm64(dst, home, 0, 15),
            AArch64ScalarLdKind::I32S => encode_sbfm64(dst, home, 0, 31),
            AArch64ScalarLdKind::I64 => encode_mov_x(dst, home),
        });
    }
    encode_ldr_scalar(dst, base_reg, offset, kind)
}

pub(super) fn encode_str_typed(
    src_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let src = parse_register(src_reg)?;
    let kind = mir_scalar_st_kind(ty)?;
    if let Some(home) = register_home(base_reg, offset) {
        return Ok(match kind {
            AArch64ScalarStKind::I8 => encode_ubfm64(home, src, 0, 7),
            AArch64ScalarStKind::I16 => encode_ubfm64(home, src, 0, 15),
            AArch64ScalarStKind::I32 => encode_ubfm64(home, src, 0, 31),
            AArch64ScalarStKind::I64 => encode_mov_x(home, src),
        });
    }
    encode_str_scalar(src, base_reg, offset, kind)
}

/// RET Xn = 0xD65F0000 | (n << 5)
pub(super) fn encode_ret(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD65F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// BR Xn = 0xD61F0000 | (n << 5)
#[allow(dead_code)]
pub(super) fn encode_br(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD61F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// BLR Xn = 0xD63F0000 | (n << 5)
pub(super) fn encode_blr(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD63F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

pub(super) fn encode_brk(imm16: u16) -> Vec<u8> {
    let w = 0xD420_0000u32 | (((imm16 as u32) & 0xFFFF) << 5);
    w.to_le_bytes().to_vec()
}

pub(super) fn encode_nop() -> Vec<u8> {
    0xD503_201Fu32.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(super) fn apply_i32_offset_to_reg(
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
    let imm_reg = parse_register(imm_reg_str)?;
    materialize_operand(
        assembler,
        &Operand::Immediate(Immediate::I64(o)),
        imm_reg,
        stack_slots,
        reg_alloc,
        code,
        stack_size,
    )?;
    code.extend_from_slice(&encode_add_rrr(base_reg, base_reg, imm_reg));
    reg_alloc.free_scratch(imm_reg_str);
    Ok(())
}

/// SBFM Xd, Xn, #immr, #imms (64-bit); used for sign-extension of low (imms+1) bits.
pub(super) fn encode_sbfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> Vec<u8> {
    let w = 0x9340_0000u32
        | ((immr as u32) << 16)
        | ((imms as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

pub(super) enum ShiftVKind {
    Lsl,
    Lsr,
    Asr,
}

/// LSLV / LSRV / ASRV Xd, Xn, Xm (64-bit variable shift; shift amount uses low 6 bits of Xm)
pub(super) fn encode_shiftv(which: ShiftVKind, rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let op = match which {
        ShiftVKind::Lsl => 0x9AC02000u32,
        ShiftVKind::Lsr => 0x9AC02400u32,
        ShiftVKind::Asr => 0x9AC02800u32,
    };
    let w = op | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

pub(super) fn encode_shiftv_w(which: ShiftVKind, rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let op = match which {
        ShiftVKind::Lsl => 0x1AC0_2000u32,
        ShiftVKind::Lsr => 0x1AC0_2400u32,
        ShiftVKind::Asr => 0x1AC0_2800u32,
    };
    let w = op | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(super) fn load_base_gpr(
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
            code.extend_from_slice(&encode_ldr(scratch, 29, *base_offset)?);
            let rn = parse_register(scratch)?;
            Ok((rn, Some(scratch)))
        }
        Register::Physical(p) => {
            let rn = parse_register(p.name)?;
            Ok((rn, None))
        }
    }
}

#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_indexed_address(
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
    let base_r = parse_register(base_s)?;
    let idx_r = parse_register(idx_s)?;
    let out_r = parse_register(out_addr_scratch)?;

    match base {
        Register::Virtual(v) => {
            let off = stack_slots.get(v).ok_or_else(|| {
                RasError::EncodingError("indexed address: missing base stack slot".into())
            })?;
            code.extend_from_slice(&encode_ldr(base_s, 29, *off)?);
        }
        Register::Physical(p) => {
            let br = parse_register(p.name)?;
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
            code.extend_from_slice(&encode_ldr(idx_s, 29, *off)?);
        }
        Register::Physical(p) => {
            let ir = parse_register(p.name)?;
            if ir != idx_r {
                code.extend_from_slice(&mov_x_dst_src(idx_r, ir));
            }
        }
    }

    if log2_scale > 0 {
        let sh_s = reg_alloc.alloc_scratch().unwrap_or("x15");
        let sh_r = parse_register(sh_s)?;
        let movz = 0xD280_0000u32 | (((log2_scale as u32) & 0xFFFF) << 5) | (sh_r as u32);
        code.extend_from_slice(&movz.to_le_bytes());
        code.extend_from_slice(&encode_shiftv(ShiftVKind::Lsl, idx_r, idx_r, sh_r));
        reg_alloc.free_scratch(sh_s);
    }

    code.extend_from_slice(&encode_add_rrr(out_r, base_r, idx_r));

    reg_alloc.free_scratch(base_s);
    reg_alloc.free_scratch(idx_s);

    Ok(())
}

/// Store a scratch register in the slot of `dst`.
#[cfg(feature = "encoder")]
pub(super) fn store_to_slot(
    code: &mut Vec<u8>,
    src_reg: &str,
    dst: &Register,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
) -> Result<(), RasError> {
    let Register::Virtual(vreg) = dst else {
        return Err(RasError::EncodingError(
            "AArch64 JIT: physical destination not supported".into(),
        ));
    };
    let offset = stack_slots.get(vreg).ok_or_else(|| {
        RasError::EncodingError(format!("No stack slot for virtual register: {:?}", vreg))
    })?;
    code.extend_from_slice(&encode_str(src_reg, 29, *offset)?);
    Ok(())
}

/// UBFM Xd, Xn, #immr, #imms (64-bit); used for zero-extension of low (imms+1) bits.
pub(super) fn encode_ubfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> Vec<u8> {
    let w = 0xD340_0000u32
        | ((immr as u32) << 16)
        | ((imms as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

/// Byte I/O builtins as raw Darwin syscalls, same sequences as the AOT backend.
/// Returns false when `name` is not one of them.
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_macos_io(
    assembler: &mut RasAssembler,
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<bool, RasError> {
    const SUB_SP_16: u32 = 0xD100_43FF;
    const ADD_SP_16: u32 = 0x9100_43FF;
    const MOV_X0_0: u32 = 0xD280_0000;
    const MOV_X0_1: u32 = 0xD280_0020;
    const MOV_X1_SP: u32 = 0x9100_03E1;
    const MOV_X2_1: u32 = 0xD280_0022;
    const MOV_X16_READ: u32 = 0xD280_0070;
    const MOV_X16_WRITE: u32 = 0xD280_0090;
    const SVC_0: u32 = 0xD400_0001;
    const MOV_X9_NEG2: u32 = 0x9280_0029;
    const MOV_X9_NEG1: u32 = 0x9280_0009;
    const CSEL_X0_X9_X0_CS: u32 = 0x9A80_2120;
    const CSEL_X0_X9_X0_EQ: u32 = 0x9A80_0120;
    const CMP_X0_0: u32 = 0xF100_001F;
    const DMB_SY: u32 = 0xD503_3FBF;
    const STRB_W9_SP: u32 = 0x3900_03E9;
    const LDRB_W9_SP: u32 = 0x3940_03E9;
    const CSINV_X0_X0_XZR_CC: u32 = 0xDA9F_3000;
    const SUBS_X0_X0_1: u32 = 0xF100_0400;
    const CSEL_X9_X9_X0_EQ: u32 = 0x9A80_0129;
    // skips x16, svc and the five result words
    const CBZ_X2_OVER_READ: u32 = 0xB400_0102;

    fn words(code: &mut Vec<u8>, list: &[u32]) {
        for word in list {
            code.extend_from_slice(&word.to_le_bytes());
        }
    }

    let mut load = |op: &Operand, dst: u8, code: &mut Vec<u8>, reg_alloc: &mut _| {
        materialize_operand(assembler, op, dst, stack_slots, reg_alloc, code, stack_size)
    };

    match (name, args) {
        ("lamina.writebyte", [value]) => {
            words(code, &[SUB_SP_16]);
            load(value, 9, code, reg_alloc)?;
            words(
                code,
                &[
                    STRB_W9_SP,
                    MOV_X0_1,
                    MOV_X1_SP,
                    MOV_X2_1,
                    MOV_X16_WRITE,
                    SVC_0,
                    MOV_X9_NEG2,
                    CSEL_X0_X9_X0_CS,
                    DMB_SY,
                ],
            );
            if let Some(dst) = ret {
                store_to_slot(code, "x0", dst, stack_slots)?;
            }
            words(code, &[ADD_SP_16]);
        }
        ("lamina.readbyte", []) => {
            words(
                code,
                &[
                    SUB_SP_16,
                    MOV_X0_0,
                    MOV_X1_SP,
                    MOV_X2_1,
                    MOV_X16_READ,
                    SVC_0,
                    CSINV_X0_X0_XZR_CC,
                ],
            );
            if let Some(dst) = ret {
                words(code, &[LDRB_W9_SP, SUBS_X0_X0_1, CSEL_X9_X9_X0_EQ]);
                store_to_slot(code, "x9", dst, stack_slots)?;
            }
            words(code, &[ADD_SP_16]);
        }
        ("lamina.write", [pointer, length]) => {
            load(pointer, 1, code, reg_alloc)?;
            load(length, 2, code, reg_alloc)?;
            words(
                code,
                &[
                    MOV_X0_1,
                    MOV_X16_WRITE,
                    SVC_0,
                    MOV_X9_NEG2,
                    CSEL_X0_X9_X0_CS,
                ],
            );
            if let Some(dst) = ret {
                store_to_slot(code, "x0", dst, stack_slots)?;
            }
        }
        ("lamina.read", [pointer, length]) => {
            load(pointer, 1, code, reg_alloc)?;
            load(length, 2, code, reg_alloc)?;
            words(
                code,
                &[
                    MOV_X0_0,
                    CBZ_X2_OVER_READ,
                    MOV_X16_READ,
                    SVC_0,
                    MOV_X9_NEG2,
                    CSEL_X0_X9_X0_CS,
                    CMP_X0_0,
                    MOV_X9_NEG1,
                    CSEL_X0_X9_X0_EQ,
                ],
            );
            if let Some(dst) = ret {
                store_to_slot(code, "x0", dst, stack_slots)?;
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// This encoder passes every value as one register word. Refuse what the AOT
/// backend passes any other way, so the caller falls back instead of running bad code.
#[cfg(feature = "encoder")]
pub(super) fn reject_unmodelled_abi(func: &lamina_mir::Function) -> Result<(), RasError> {
    let name = &func.sig.name;
    let refuse = |what: &str| {
        Err(RasError::EncodingError(format!(
            "AArch64 JIT: @{name} has {what}, not supported"
        )))
    };
    if func.sig.params.iter().any(|p| p.aggregate.is_some()) {
        return refuse("an aggregate parameter");
    }
    if func.sig.ret_aggregate.is_some() || func.sig.sret_claimed {
        return refuse("an aggregate return");
    }
    if !func.variadic_wide_arguments.is_empty() || !func.variadic_wide_results.is_empty() {
        return refuse("a wide variadic value");
    }
    // A slot that is also a parameter or call result holds a value split over registers.
    let mut slots = std::collections::HashSet::new();
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        if let lamina_mir::Instruction::StackAddr { slot, .. } = &inst.instr {
            slots.insert(*slot);
        }
    }
    let is_slot = |reg: &Register| matches!(reg, Register::Virtual(v) if slots.contains(v));
    if func.sig.params.iter().any(|p| is_slot(&p.reg)) {
        return refuse("a parameter held in memory");
    }
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        if let lamina_mir::Instruction::Call { ret: Some(ret), .. } = &inst.instr
            && is_slot(ret)
        {
            return refuse("a call result held in memory");
        }
    }
    Ok(())
}

/// Call a C symbol: integers in x0-x7, floats in d0-d7, and on Darwin every variadic
/// argument on the stack in its own 8 bytes.
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_extern_call(
    assembler: &mut RasAssembler,
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<(), RasError> {
    let module = assembler.current_module.ok_or_else(|| {
        RasError::EncodingError("AArch64 JIT: no module for extern call".to_string())
    })?;
    // SAFETY: compile_mir_to_binary_function sets this from the module it is still borrowing.
    let module = unsafe { &*module };
    let refuse = |what: &str| {
        Err(RasError::EncodingError(format!(
            "AArch64 JIT: extern call to @{name}: {what}"
        )))
    };
    let Some(func) = module
        .functions
        .get(name)
        .or_else(|| module.functions.get(name.trim_start_matches('@')))
    else {
        return refuse("no declaration in this module");
    };
    let sig = &func.sig;
    if sig.variadic && assembler.target_os != lamina_platform::TargetOperatingSystem::MacOS {
        return refuse("variadic call is only written for Darwin");
    }
    let fixed = sig.params.len();
    if args.len() < fixed || (!sig.variadic && args.len() != fixed) {
        return refuse("argument count does not match the declaration");
    }
    if sig.params.iter().any(|p| p.ty.is_vector()) || sig.ret_ty.is_some_and(|t| t.is_vector()) {
        return refuse("vector values not supported");
    }

    let (mut next_general, mut next_float) = (0u8, 0u8);
    for (param, arg) in sig.params.iter().zip(args) {
        if param.ty.is_float() {
            if next_float == 8 {
                return refuse("float stack arguments not supported");
            }
            let prefix = float_register_prefix(&param.ty)?;
            materialize_operand(assembler, arg, 16, stack_slots, reg_alloc, code, stack_size)?;
            encode_float_in(assembler, code, prefix, next_float, 16)?;
            next_float += 1;
        } else {
            if next_general == 8 {
                // Darwin packs narrow stack arguments, register words do not model that.
                return refuse("stack arguments not supported");
            }
            materialize_operand(
                assembler,
                arg,
                next_general,
                stack_slots,
                reg_alloc,
                code,
                stack_size,
            )?;
            next_general += 1;
        }
    }

    let variadic = &args[fixed..];
    let area = (variadic.len() * 8 + 15) & !15;
    if area > 0xFFF {
        return refuse("too many variadic arguments");
    }
    if area > 0 {
        let sub = 0xD100_03FFu32 | ((area as u32) << 10);
        code.extend_from_slice(&sub.to_le_bytes());
        for (index, arg) in variadic.iter().enumerate() {
            materialize_operand(assembler, arg, 16, stack_slots, reg_alloc, code, stack_size)?;
            code.extend_from_slice(&encode_str_imm64(16, 31, (index * 8) as i32)?);
        }
    }

    let address = resolve_symbol(assembler, name)?;
    materialize_operand(
        assembler,
        &Operand::Immediate(Immediate::I64(address as i64)),
        16,
        stack_slots,
        reg_alloc,
        code,
        stack_size,
    )?;
    code.extend_from_slice(&encode_blr(16)?);
    if area > 0 {
        let add = 0x9100_03FFu32 | ((area as u32) << 10);
        code.extend_from_slice(&add.to_le_bytes());
    }

    if let Some(dst) = ret {
        match sig.ret_ty {
            Some(ty) if ty.is_float() => {
                let prefix = float_register_prefix(&ty)?;
                encode_float_out(assembler, code, prefix, 0)?;
            }
            // C leaves the bits above a narrow return undefined.
            Some(ty) => {
                let bytes = ty.size_bytes(lamina_platform::PointerWidth::Bits64);
                if (1..=4).contains(&bytes) {
                    code.extend_from_slice(&encode_sbfm64(0, 0, 0, (bytes * 8 - 1) as u8));
                }
            }
            None => {}
        }
        store_to_slot(code, "x0", dst, stack_slots)?;
    }
    Ok(())
}

/// The address of a C symbol loaded in this process.
#[cfg(feature = "encoder")]
pub(super) fn resolve_symbol(assembler: &mut RasAssembler, name: &str) -> Result<u64, RasError> {
    let symbol = name.trim_start_matches('@');
    if !assembler.function_pointers.contains_key(symbol) {
        assembler.register_function(symbol)?;
    }
    assembler
        .function_pointers
        .get(symbol)
        .copied()
        .ok_or_else(|| RasError::EncodingError(format!("{symbol} not resolved")))
}

/// Give every global in the module storage and note its address next to the C symbols.
///
/// Compiled code holds these addresses and `ExecutableMemory` only owns the code, so
/// the storage is leaked. It lasts as long as the process, like a linked program's data.
#[cfg(feature = "encoder")]
pub(super) fn allocate_globals(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
) -> Result<(), RasError> {
    let mut placed = Vec::with_capacity(module.globals.len());
    for (name, global) in &module.globals {
        if global.thread_local.is_some() {
            return Err(RasError::EncodingError(format!(
                "AArch64 JIT: thread-local global ${name} not supported"
            )));
        }
        let size = global.size.max(1);
        let layout = std::alloc::Layout::from_size_align(size, global.align.max(1))
            .map_err(|e| RasError::EncodingError(format!("global ${name}: {e}")))?;
        // SAFETY: the layout has a non-zero size.
        let storage = unsafe { std::alloc::alloc_zeroed(layout) };
        if storage.is_null() {
            return Err(RasError::EncodingError(format!(
                "AArch64 JIT: cannot allocate global ${name}"
            )));
        }
        // SAFETY: fresh allocation of `size` bytes, never freed.
        let bytes = unsafe { std::slice::from_raw_parts_mut(storage, size) };
        if let Some(initializer) = &global.initializer {
            let Some(target) = bytes.get_mut(..initializer.len()) else {
                return Err(RasError::EncodingError(format!(
                    "AArch64 JIT: initializer of ${name} is larger than the global"
                )));
            };
            target.copy_from_slice(initializer);
        }
        assembler
            .function_pointers
            .insert(name.clone(), storage as u64);
        placed.push((global, bytes));
    }
    for (global, bytes) in placed {
        for relocation in &global.relocations {
            if module
                .functions
                .get(&relocation.symbol)
                .is_some_and(|f| !f.blocks.is_empty())
            {
                return Err(RasError::EncodingError(format!(
                    "AArch64 JIT: ${} holds the address of function @{}, not supported",
                    global.name, relocation.symbol
                )));
            }
            let address = resolve_symbol(assembler, &relocation.symbol)?;
            let value = address.wrapping_add(relocation.addend as u64).to_le_bytes();
            let slot = bytes
                .get_mut(relocation.offset..relocation.offset + relocation.size)
                .filter(|slot| slot.len() <= value.len())
                .ok_or_else(|| {
                    RasError::EncodingError(format!(
                        "AArch64 JIT: bad relocation in ${}",
                        global.name
                    ))
                })?;
            let width = slot.len();
            slot.copy_from_slice(&value[..width]);
        }
    }
    Ok(())
}

/// Encode one instruction through the text encoder, which already knows the float forms.
#[cfg(feature = "encoder")]
pub(super) fn encode_text(
    assembler: &mut RasAssembler,
    code: &mut Vec<u8>,
    line: &str,
) -> Result<(), RasError> {
    let (opcode, operands) = line.split_once(' ').unwrap_or((line, ""));
    let bytes = assembler.encoder.encode_instruction(&ParsedInstruction {
        opcode: opcode.to_string(),
        operands: operands
            .split(',')
            .map(str::trim)
            .filter(|operand| !operand.is_empty())
            .map(str::to_string)
            .collect(),
    })?;
    code.extend_from_slice(&bytes);
    Ok(())
}

/// `s` or `d` for a float type, refusing anything else.
#[cfg(feature = "encoder")]
pub(super) fn float_register_prefix(ty: &MirType) -> Result<&'static str, RasError> {
    match ty {
        MirType::Scalar(ScalarType::F32) => Ok("s"),
        MirType::Scalar(ScalarType::F64) => Ok("d"),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT: {:?} is not a scalar float",
            ty
        ))),
    }
}

/// Move float bits from a general register into `s<index>` or `d<index>`.
#[cfg(feature = "encoder")]
pub(super) fn encode_float_in(
    assembler: &mut RasAssembler,
    code: &mut Vec<u8>,
    prefix: &str,
    index: u8,
    general: u8,
) -> Result<(), RasError> {
    let source = if prefix == "s" { "w" } else { "x" };
    encode_text(
        assembler,
        code,
        &format!("fmov {prefix}{index}, {source}{general}"),
    )
}

/// Move float bits from `s0` or `d0` back to a general register.
#[cfg(feature = "encoder")]
pub(super) fn encode_float_out(
    assembler: &mut RasAssembler,
    code: &mut Vec<u8>,
    prefix: &str,
    general: u8,
) -> Result<(), RasError> {
    let target = if prefix == "s" { "w" } else { "x" };
    encode_text(
        assembler,
        code,
        &format!("fmov {target}{general}, {prefix}0"),
    )
}

pub(super) fn parse_register(reg: &str) -> Result<u8, RasError> {
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

/// Bit width of an integer type the JIT reads in place, refusing anything else.
#[cfg(feature = "encoder")]
pub(super) fn int_width_bits(ty: &MirType) -> Result<u8, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I8) => Ok(8),
        MirType::Scalar(ScalarType::I16) => Ok(16),
        MirType::Scalar(ScalarType::I32) => Ok(32),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(64),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT: integer operation on {:?} not supported",
            ty
        ))),
    }
}

/// Little-endian bytes of one A64 instruction word.
#[cfg(feature = "encoder")]
fn word_bytes(word: u32) -> Vec<u8> {
    word.to_le_bytes().to_vec()
}

/// Encode `IntUnary`. The operand is read at the width of `ty` (its upper bits are not
/// trusted), and the result is written back sign-extended, the same way `Load` leaves a
/// narrow value in its slot.
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_int_unary(
    assembler: &mut RasAssembler,
    op: lamina_mir::IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<(), RasError> {
    use lamina_mir::IntUnOp;
    // x9 holds the value, x10-x12 are temporaries. Nothing here reads the scratch pool.
    const X: u8 = 9;
    const T: u8 = 10;
    const M: u8 = 11;
    const U: u8 = 12;
    let bits = int_width_bits(ty)?;
    let narrow = bits < 64;

    materialize_operand(assembler, src, X, stack_slots, reg_alloc, code, stack_size)?;
    if narrow {
        code.extend_from_slice(&encode_ubfm64(X, X, 0, bits - 1));
    }

    match op {
        IntUnOp::Popcount => {
            // SWAR population count over the 64-bit zero-extended value.
            code.extend_from_slice(&encode_ubfm64(T, X, 1, 63));
            materialize_operand(
                assembler,
                &imm_i64(0x5555_5555_5555_5555),
                M,
                stack_slots,
                reg_alloc,
                code,
                stack_size,
            )?;
            code.extend_from_slice(&encode_and_rrr(T, T, M));
            code.extend_from_slice(&word_bytes(
                0xCB00_0000 | ((T as u32) << 16) | ((X as u32) << 5) | X as u32,
            ));
            materialize_operand(
                assembler,
                &imm_i64(0x3333_3333_3333_3333),
                M,
                stack_slots,
                reg_alloc,
                code,
                stack_size,
            )?;
            code.extend_from_slice(&encode_and_rrr(T, X, M));
            code.extend_from_slice(&encode_ubfm64(U, X, 2, 63));
            code.extend_from_slice(&encode_and_rrr(U, U, M));
            code.extend_from_slice(&encode_add_rrr(X, T, U));
            code.extend_from_slice(&encode_ubfm64(T, X, 4, 63));
            code.extend_from_slice(&encode_add_rrr(X, X, T));
            materialize_operand(
                assembler,
                &imm_i64(0x0F0F_0F0F_0F0F_0F0F),
                M,
                stack_slots,
                reg_alloc,
                code,
                stack_size,
            )?;
            code.extend_from_slice(&encode_and_rrr(X, X, M));
            materialize_operand(
                assembler,
                &imm_i64(0x0101_0101_0101_0101),
                M,
                stack_slots,
                reg_alloc,
                code,
                stack_size,
            )?;
            code.extend_from_slice(&word_bytes(
                0x9B00_7C00 | ((M as u32) << 16) | ((X as u32) << 5) | X as u32,
            ));
            code.extend_from_slice(&encode_ubfm64(X, X, 56, 63));
        }
        IntUnOp::Ctlz => {
            // CLZ of the zero-extended value counts the unused high bits too.
            code.extend_from_slice(&word_bytes(0xDAC0_1000 | ((X as u32) << 5) | X as u32));
            if narrow {
                code.extend_from_slice(&word_bytes(
                    0xD100_0000 | (u32::from(64 - bits) << 10) | ((X as u32) << 5) | X as u32,
                ));
            }
        }
        IntUnOp::Cttz => {
            // A bit just above the width caps the count at the width, and gives it for zero.
            if narrow {
                materialize_operand(
                    assembler,
                    &imm_i64(1i64 << bits),
                    T,
                    stack_slots,
                    reg_alloc,
                    code,
                    stack_size,
                )?;
                code.extend_from_slice(&word_bytes(
                    0xAA00_0000 | ((T as u32) << 16) | ((X as u32) << 5) | X as u32,
                ));
            }
            code.extend_from_slice(&word_bytes(0xDAC0_0000 | ((X as u32) << 5) | X as u32));
            code.extend_from_slice(&word_bytes(0xDAC0_1000 | ((X as u32) << 5) | X as u32));
        }
        IntUnOp::Bswap => match bits {
            8 => {}
            16 => code.extend_from_slice(&word_bytes(0x5AC0_0400 | ((X as u32) << 5) | X as u32)),
            32 => code.extend_from_slice(&word_bytes(0x5AC0_0800 | ((X as u32) << 5) | X as u32)),
            _ => code.extend_from_slice(&word_bytes(0xDAC0_0C00 | ((X as u32) << 5) | X as u32)),
        },
    }

    if narrow {
        code.extend_from_slice(&encode_sbfm64(X, X, 0, bits - 1));
    }
    store_to_slot(code, "x9", dst, stack_slots)
}

#[cfg(feature = "encoder")]
fn imm_i64(value: i64) -> Operand {
    Operand::Immediate(Immediate::I64(value))
}

/// Encode `IntOverflow`: whether the wrapping operation leaves the range of `ty`, read with
/// the given signedness, as an i1 that is 0 or 1 in the slot.
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_int_overflow(
    assembler: &mut RasAssembler,
    op: lamina_mir::IntOverflowOp,
    ty: &MirType,
    signed: bool,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<(), RasError> {
    use lamina_mir::IntOverflowOp;
    // x9 and x10 hold the operands, x11 the wrapped result, x12 and x14 are temporaries.
    const L: u8 = 9;
    const R: u8 = 10;
    const W: u8 = 11;
    const U: u8 = 12;
    const H: u8 = 14;
    // Condition codes.
    const NE: u32 = 0b0001;
    const HS: u32 = 0b0010;
    const LO: u32 = 0b0011;
    const VS: u32 = 0b0110;

    if matches!(ty, MirType::Scalar(ScalarType::Ptr)) {
        return Err(RasError::EncodingError(
            "AArch64 JIT: IntOverflow on a pointer not supported".into(),
        ));
    }
    let bits = int_width_bits(ty)?;

    materialize_operand(assembler, lhs, L, stack_slots, reg_alloc, code, stack_size)?;
    materialize_operand(assembler, rhs, R, stack_slots, reg_alloc, code, stack_size)?;

    // Register-form word: `base` with Rm, Rn and Rd filled in.
    let rrr = |code: &mut Vec<u8>, base: u32, rd: u8, rn: u8, rm: u8| {
        let word = base | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32;
        code.extend_from_slice(&word_bytes(word));
    };
    // CSET Wd, cond is CSINC with the inverted condition.
    let cset = |code: &mut Vec<u8>, cond: u32| {
        code.extend_from_slice(&word_bytes(0x9A9F_07E0 | ((cond ^ 1) << 12) | W as u32));
    };
    // CMP Rn, Rm is SUBS with the zero register as the destination.
    let cmp = |code: &mut Vec<u8>, rn: u8, rm: u8| {
        rrr(code, 0xEB00_001F, 31, rn, rm);
    };

    if bits < 64 {
        // Widened from the type to 64 bits, so the wide result cannot wrap.
        for reg in [L, R] {
            if signed {
                code.extend_from_slice(&encode_sbfm64(reg, reg, 0, bits - 1));
            } else {
                code.extend_from_slice(&encode_ubfm64(reg, reg, 0, bits - 1));
            }
        }
        match op {
            IntOverflowOp::Add => rrr(code, 0x8B00_0000, W, L, R),
            IntOverflowOp::Sub => rrr(code, 0xCB00_0000, W, L, R),
            IntOverflowOp::Mul => rrr(code, 0x9B00_7C00, W, L, R),
        }
        // In range exactly when the wide result equals its own low-bit extension.
        if signed {
            code.extend_from_slice(&encode_sbfm64(U, W, 0, bits - 1));
        } else {
            code.extend_from_slice(&encode_ubfm64(U, W, 0, bits - 1));
        }
        cmp(code, W, U);
        cset(code, NE);
    } else {
        match (op, signed) {
            (IntOverflowOp::Add, true) => {
                rrr(code, 0xAB00_0000, W, L, R);
                cset(code, VS);
            }
            (IntOverflowOp::Add, false) => {
                rrr(code, 0xAB00_0000, W, L, R);
                cset(code, HS);
            }
            (IntOverflowOp::Sub, true) => {
                rrr(code, 0xEB00_0000, W, L, R);
                cset(code, VS);
            }
            (IntOverflowOp::Sub, false) => {
                rrr(code, 0xEB00_0000, W, L, R);
                cset(code, LO);
            }
            (IntOverflowOp::Mul, true) => {
                rrr(code, 0x9B00_7C00, W, L, R);
                // SMULH gives the high half, which overflows unless it is the low half's sign.
                rrr(code, 0x9B40_7C00, U, L, R);
                code.extend_from_slice(&encode_sbfm64(H, W, 63, 63));
                cmp(code, U, H);
                cset(code, NE);
            }
            (IntOverflowOp::Mul, false) => {
                rrr(code, 0x9B00_7C00, W, L, R);
                // UMULH must be zero for the product to fit.
                rrr(code, 0x9BC0_7C00, U, L, R);
                cmp(code, U, 31);
                cset(code, NE);
            }
        }
    }
    store_to_slot(code, "x11", dst, stack_slots)
}
