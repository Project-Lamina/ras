//! AArch64 instruction encoding helpers used when lowering MIR.

use super::operand::materialize_operand;
use crate::assembler::core::RasAssembler;
use crate::encoder::aarch64::fixed::{enc_ldp_post_64, enc_stp_pre_64, encode_add_rrr};
use crate::encoder::aarch64::ldst_imm64::{
    AArch64ScalarLdKind, AArch64ScalarStKind, encode_ldr_imm64, encode_ldr_scalar, encode_str_imm64,
    encode_str_scalar,
};
use crate::error::RasError;
use lamina_mir::{Immediate, MirType, Operand, Register, ScalarType};

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

        if aligned_size > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "Stack size {} (aligned: {}) too large for single SUB instruction",
                stack_size, aligned_size
            )));
        }
        // Encoding matches clang/llvm-mc:
        //   sub sp, sp, #imm12  => 0xD10003FF | (imm12 << 10)
        let sub_sp = 0xD100_03FFu32 | ((aligned_size as u32) << 10);
        code.extend_from_slice(&sub_sp.to_le_bytes());
    }

    Ok(code)
}

/// Must restore SP before LDP (undo the SUB sp, sp, #aligned_size from prologue)
pub(super) fn encode_epilogue(aligned_stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    if aligned_stack_size > 0 {
        if aligned_stack_size > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "stack restore too large for single ADD: {}",
                aligned_stack_size
            )));
        }
        // Encoding matches clang/llvm-mc:
        //   add sp, sp, #imm12  => 0x910003FF | (imm12 << 10)
        let add_sp = 0x9100_03FFu32 | ((aligned_stack_size as u32) << 10);
        code.extend_from_slice(&add_sp.to_le_bytes());
    }

    let ldp = enc_ldp_post_64(29, 30, 31, 16)?;
    code.extend_from_slice(&ldp.to_le_bytes());

    Ok(code)
}

pub(super) fn encode_str(src_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let src = parse_register(src_reg)?;
    encode_str_imm64(src, base_reg, offset)
}

pub(super) fn encode_ldr(dst_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let dst = parse_register(dst_reg)?;
    encode_ldr_imm64(dst, base_reg, offset)
}

pub(super) fn mir_scalar_ld_kind(ty: &lamina_mir::MirType) -> Result<AArch64ScalarLdKind, RasError> {
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

pub(super) fn mir_scalar_st_kind(ty: &lamina_mir::MirType) -> Result<AArch64ScalarStKind, RasError> {
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

pub(super) fn encode_ldr_typed(
    dst_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let dst = parse_register(dst_reg)?;
    let kind = mir_scalar_ld_kind(ty)?;
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
