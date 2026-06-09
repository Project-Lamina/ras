//! RISC-V binary code generation (RV32I/RV64I + M-extension)
//!
//! Compiles MIR to RISC-V machine code for JIT execution.
//! Supports RV64 (default) and RV32 via the `rv64` flag.
//! Calling convention: RISC-V LP64 (Linux) — a0-a7 for args, a0 for return.

use crate::assembler::core::RasAssembler;
use crate::error::RasError;
use std::collections::HashMap;
use lamina_mir::{Immediate, Operand, Register, Instruction, IntBinOp, IntCmpOp, MirType, ScalarType, Operand as Op};

// ---------------------------------------------------------------------------
// RISC-V instruction format encoders
// ---------------------------------------------------------------------------

#[inline]
fn r_type(funct7: u8, rs2: u8, rs1: u8, funct3: u8, rd: u8, opcode: u8) -> [u8; 4] {
    let w = ((funct7 as u32) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((rd as u32) << 7)
        | (opcode as u32);
    w.to_le_bytes()
}

#[inline]
fn i_type(imm12: i32, rs1: u8, funct3: u8, rd: u8, opcode: u8) -> [u8; 4] {
    let w = ((imm12 as u32 & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((rd as u32) << 7)
        | (opcode as u32);
    w.to_le_bytes()
}

#[inline]
fn s_type(imm12: i32, rs2: u8, rs1: u8, funct3: u8, opcode: u8) -> [u8; 4] {
    let imm = imm12 as u32 & 0xFFF;
    let w = ((imm >> 5) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((imm & 0x1F) << 7)
        | (opcode as u32);
    w.to_le_bytes()
}

#[inline]
fn b_type(offset: i32, rs2: u8, rs1: u8, funct3: u8) -> [u8; 4] {
    let o = offset as u32;
    let imm12 = (o >> 12) & 1;
    let imm11 = (o >> 11) & 1;
    let imm10_5 = (o >> 5) & 0x3F;
    let imm4_1 = (o >> 1) & 0xF;
    let w = (imm12 << 31)
        | (imm10_5 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | ((funct3 as u32) << 12)
        | (imm4_1 << 8)
        | (imm11 << 7)
        | 0x63;
    w.to_le_bytes()
}

#[inline]
fn u_type(imm20: i32, rd: u8, opcode: u8) -> [u8; 4] {
    let w = ((imm20 as u32 & 0xF_FFFF) << 12) | ((rd as u32) << 7) | (opcode as u32);
    w.to_le_bytes()
}

#[inline]
fn j_type(offset: i32, rd: u8) -> [u8; 4] {
    let o = offset as u32;
    let imm20 = (o >> 20) & 1;
    let imm19_12 = (o >> 12) & 0xFF;
    let imm11 = (o >> 11) & 1;
    let imm10_1 = (o >> 1) & 0x3FF;
    let w = (imm20 << 31)
        | (imm10_1 << 21)
        | (imm11 << 20)
        | (imm19_12 << 12)
        | ((rd as u32) << 7)
        | 0x6F;
    w.to_le_bytes()
}

// ---------------------------------------------------------------------------
// Common instructions
// ---------------------------------------------------------------------------
// Register numbers used in generated code:
//  0  = x0/zero  (always 0)
//  1  = x1/ra    (return address)
//  2  = x2/sp    (stack pointer)
//  8  = x8/s0/fp (frame pointer)
//  5  = x5/t0    (scratch 1)
//  6  = x6/t1    (scratch 2)
//  7  = x7/t2    (scratch 3)
//  10 = x10/a0   (arg0 / return value)
// 10–17 = a0–a7  (arguments)
// 18–27 = s2–s11 (virtual register allocation pool)

const REG_ZERO: u8 = 0;
const REG_RA: u8 = 1;
const REG_SP: u8 = 2;
const REG_FP: u8 = 8; // s0/fp
const REG_T0: u8 = 5; // scratch
const REG_T1: u8 = 6; // scratch
const REG_T2: u8 = 7; // scratch / result staging

// Argument registers (a0–a7)
const ARG_REGS: [u8; 8] = [10, 11, 12, 13, 14, 15, 16, 17];

// Callee-saved virtual register pool: s2–s11
// s2–s11: available for future vreg pinning optimisation
#[allow(dead_code)]
const VREG_POOL: [u8; 10] = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

#[inline]
fn emit_addi(rd: u8, rs1: u8, imm: i32) -> [u8; 4] {
    i_type(imm, rs1, 0x0, rd, 0x13)
}

#[inline]
fn emit_add(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x0, rd, 0x33)
}
#[inline]
fn emit_addw(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x0, rd, 0x3B)
}

#[inline]
fn emit_sub(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x20, rs2, rs1, 0x0, rd, 0x33)
}
#[inline]
fn emit_subw(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x20, rs2, rs1, 0x0, rd, 0x3B)
}

#[inline]
fn emit_mul(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x01, rs2, rs1, 0x0, rd, 0x33)
}
#[inline]
fn emit_mulw(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x01, rs2, rs1, 0x0, rd, 0x3B)
}

#[inline]
fn emit_div(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x01, rs2, rs1, 0x4, rd, 0x33) // sdiv
}
#[inline]
fn emit_divu(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x01, rs2, rs1, 0x5, rd, 0x33)
}
#[inline]
fn emit_rem(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x01, rs2, rs1, 0x6, rd, 0x33)
}
#[inline]
fn emit_remu(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x01, rs2, rs1, 0x7, rd, 0x33)
}

#[inline]
fn emit_and(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x7, rd, 0x33)
}
#[inline]
fn emit_or(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x6, rd, 0x33)
}
#[inline]
fn emit_xor(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x4, rd, 0x33)
}

#[inline]
fn emit_sll(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x1, rd, 0x33)
}
#[inline]
fn emit_srl(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x5, rd, 0x33)
}
#[inline]
fn emit_sra(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x20, rs2, rs1, 0x5, rd, 0x33)
}
#[inline]
fn emit_sllw(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x1, rd, 0x3B)
}
#[inline]
fn emit_srlw(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x5, rd, 0x3B)
}
#[inline]
fn emit_sraw(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x20, rs2, rs1, 0x5, rd, 0x3B)
}

// SLT / SLTU: rd = (rs1 < rs2) ? 1 : 0
#[inline]
fn emit_slt(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x2, rd, 0x33)
}
#[inline]
fn emit_sltu(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    r_type(0x00, rs2, rs1, 0x3, rd, 0x33)
}

// SLTIU rd, rs, 1  →  seqz rd, rs  (rd = (rs == 0))
#[inline]
fn emit_sltiu_1(rd: u8, rs: u8) -> [u8; 4] {
    i_type(1, rs, 0x3, rd, 0x13)
}

// XORI rd, rs, 1  →  flip LSB (toggle 0/1 bool result)
#[inline]
fn emit_xori_1(rd: u8, rs: u8) -> [u8; 4] {
    i_type(1, rs, 0x4, rd, 0x13)
}

// Loads
#[inline]
fn emit_ld(rd: u8, rs1: u8, offset: i32) -> [u8; 4] {
    i_type(offset, rs1, 0x3, rd, 0x03) // LD (64-bit sign-extend)
}
#[inline]
fn emit_lw(rd: u8, rs1: u8, offset: i32) -> [u8; 4] {
    i_type(offset, rs1, 0x2, rd, 0x03) // LW (32-bit sign-extend)
}

// Stores
#[inline]
fn emit_sd(rs2: u8, rs1: u8, offset: i32) -> [u8; 4] {
    s_type(offset, rs2, rs1, 0x3, 0x23) // SD (64-bit)
}
#[inline]
fn emit_sw(rs2: u8, rs1: u8, offset: i32) -> [u8; 4] {
    s_type(offset, rs2, rs1, 0x2, 0x23) // SW (32-bit)
}

// LUI: rd = imm20 << 12
#[inline]
fn emit_lui(rd: u8, imm20: i32) -> [u8; 4] {
    u_type(imm20, rd, 0x37)
}

// JALR rd, rs1, offset — available for external/indirect calls
#[allow(dead_code)]
#[inline]
fn emit_jalr(rd: u8, rs1: u8, offset: i32) -> [u8; 4] {
    i_type(offset, rs1, 0x0, rd, 0x67)
}

// RET = JALR x0, 0(x1)
#[inline]
fn emit_ret() -> [u8; 4] {
    i_type(0, REG_RA, 0x0, REG_ZERO, 0x67)
}

// MV rd, rs  =  ADDI rd, rs, 0
#[inline]
fn emit_mv(rd: u8, rs: u8) -> [u8; 4] {
    emit_addi(rd, rs, 0)
}

// JAL x0, offset  (unconditional jump, discard return addr)
#[inline]
fn emit_jal_zero(offset: i32) -> [u8; 4] {
    j_type(offset, REG_ZERO)
}

// BEQ / BNE / BLT / BGE / BLTU / BGEU
// BEQ — used for future equality-branch optimisations
#[allow(dead_code)]
#[inline]
fn emit_beq(rs1: u8, rs2: u8, offset: i32) -> [u8; 4] {
    b_type(offset, rs2, rs1, 0x0)
}
#[inline]
fn emit_bne(rs1: u8, rs2: u8, offset: i32) -> [u8; 4] {
    b_type(offset, rs2, rs1, 0x1)
}

// ---------------------------------------------------------------------------
// Load an arbitrary i64 immediate into a register (≤4 instructions)
// ---------------------------------------------------------------------------
fn emit_li64(code: &mut Vec<u8>, rd: u8, imm: i64) {
    if (-2048..=2047).contains(&imm) {
        // addi rd, x0, imm
        code.extend_from_slice(&emit_addi(rd, REG_ZERO, imm as i32));
    } else if (-0x8000_0000i64..=0x7FFF_FFFFi64).contains(&imm) {
        // lui + addi (32-bit range)
        let lower = (imm as i32) & 0xFFF;
        let lower_sext = if lower >= 2048 { lower - 4096 } else { lower };
        let upper = ((imm as i32) - lower_sext + (1 << 11)) >> 12;
        code.extend_from_slice(&emit_lui(rd, upper));
        if lower_sext != 0 {
            code.extend_from_slice(&emit_addi(rd, rd, lower_sext));
        }
    } else {
        // Full 64-bit: LUI+ADDI for upper half, SLLI, LUI+ADDI for lower, OR
        // Build lower 32 bits in rd, upper 32 bits in t2, then combine.
        let lo32 = imm as i32;
        let hi32 = (imm >> 32) as i32;

        // Lower 32 into rd
        let lo_lower = lo32 & 0xFFF;
        let lo_lower_sext = if lo_lower >= 2048 {
            lo_lower - 4096
        } else {
            lo_lower
        };
        let lo_upper = (lo32 - lo_lower_sext + (1 << 11)) >> 12;
        if lo_upper != 0 {
            code.extend_from_slice(&emit_lui(rd, lo_upper));
            if lo_lower_sext != 0 {
                code.extend_from_slice(&emit_addi(rd, rd, lo_lower_sext));
            }
        } else {
            code.extend_from_slice(&emit_addi(rd, REG_ZERO, lo_lower_sext));
        }

        // Upper 32 into t2 (x7)
        let hi_lower = hi32 & 0xFFF;
        let hi_lower_sext = if hi_lower >= 2048 {
            hi_lower - 4096
        } else {
            hi_lower
        };
        let hi_upper = (hi32 - hi_lower_sext + (1 << 11)) >> 12;
        if hi_upper != 0 {
            code.extend_from_slice(&emit_lui(REG_T2, hi_upper));
            if hi_lower_sext != 0 {
                code.extend_from_slice(&emit_addi(REG_T2, REG_T2, hi_lower_sext));
            }
        } else {
            code.extend_from_slice(&emit_addi(REG_T2, REG_ZERO, hi_lower_sext));
        }

        // rd = (t2 << 32) | rd
        // slli t2, t2, 32
        let slli_w: u32 =
            ((32u32) << 20) | ((REG_T2 as u32) << 15) | (0x1 << 12) | ((REG_T2 as u32) << 7) | 0x13;
        code.extend_from_slice(&slli_w.to_le_bytes());
        code.extend_from_slice(&emit_or(rd, REG_T2, rd));
    }
}

// ---------------------------------------------------------------------------
// Stack helpers — all offsets are relative to FP (s0/x8)
// ---------------------------------------------------------------------------
fn emit_store_to_fp(code: &mut Vec<u8>, src: u8, offset: i32, rv64: bool) {
    if rv64 {
        code.extend_from_slice(&emit_sd(src, REG_FP, offset));
    } else {
        code.extend_from_slice(&emit_sw(src, REG_FP, offset));
    }
}

fn emit_load_from_fp(code: &mut Vec<u8>, dst: u8, offset: i32, rv64: bool) {
    if rv64 {
        code.extend_from_slice(&emit_ld(dst, REG_FP, offset));
    } else {
        code.extend_from_slice(&emit_lw(dst, REG_FP, offset));
    }
}

// ---------------------------------------------------------------------------
// Prologue / Epilogue
// ---------------------------------------------------------------------------
fn encode_prologue(frame_size: usize, rv64: bool) -> Result<Vec<u8>, RasError> {
    // Minimum frame: 16 bytes for ra + fp (8+8). Round up to 16-byte alignment.
    let min_frame = 16usize;
    let total = (frame_size + min_frame + 15) & !15;

    if total > 2047 {
        return Err(RasError::EncodingError(format!(
            "RISC-V frame size {} too large for single ADDI (max 2031 usable bytes)",
            total
        )));
    }

    let mut code = Vec::with_capacity(16);
    // addi sp, sp, -total
    code.extend_from_slice(&emit_addi(REG_SP, REG_SP, -(total as i32)));
    // sd/sw ra, (total-8)(sp)
    if rv64 {
        code.extend_from_slice(&emit_sd(REG_RA, REG_SP, (total - 8) as i32));
        // sd/sw fp, (total-16)(sp)
        code.extend_from_slice(&emit_sd(REG_FP, REG_SP, (total - 16) as i32));
    } else {
        code.extend_from_slice(&emit_sw(REG_RA, REG_SP, (total - 8) as i32));
        code.extend_from_slice(&emit_sw(REG_FP, REG_SP, (total - 16) as i32));
    }
    // addi fp, sp, total  (set frame pointer to top of frame)
    code.extend_from_slice(&emit_addi(REG_FP, REG_SP, total as i32));
    Ok(code)
}

fn encode_epilogue(frame_size: usize, rv64: bool) -> Result<Vec<u8>, RasError> {
    let min_frame = 16usize;
    let total = (frame_size + min_frame + 15) & !15;

    let mut code = Vec::with_capacity(16);
    if rv64 {
        code.extend_from_slice(&emit_ld(REG_RA, REG_SP, (total - 8) as i32));
        code.extend_from_slice(&emit_ld(REG_FP, REG_SP, (total - 16) as i32));
    } else {
        code.extend_from_slice(&emit_lw(REG_RA, REG_SP, (total - 8) as i32));
        code.extend_from_slice(&emit_lw(REG_FP, REG_SP, (total - 16) as i32));
    }
    // addi sp, sp, total
    code.extend_from_slice(&emit_addi(REG_SP, REG_SP, total as i32));
    code.extend_from_slice(&emit_ret());
    Ok(code)
}

// ---------------------------------------------------------------------------
// Operand materialisation — loads into `dst_reg`
// ---------------------------------------------------------------------------
fn materialize_operand(
    code: &mut Vec<u8>,
    operand: &lamina_mir::Operand,
    dst_reg: u8,
    stack_slots: &HashMap<lamina_mir::VirtualReg, i32>,
    rv64: bool,
) -> Result<(), RasError> {
    match operand {
        Operand::Immediate(imm) => {
            let v: i64 = match imm {
                Immediate::I8(x) => *x as i64,
                Immediate::I16(x) => *x as i64,
                Immediate::I32(x) => *x as i64,
                Immediate::I64(x) => *x,
                Immediate::F32(_) | Immediate::F64(_) => {
                    return Err(RasError::EncodingError(
                        "RISC-V JIT: float immediates not supported".to_string(),
                    ));
                }
            };
            emit_li64(code, dst_reg, v);
        }
        Operand::Register(Register::Virtual(vreg)) => {
            let slot = stack_slots.get(vreg).ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for vreg {:?}", vreg))
            })?;
            emit_load_from_fp(code, dst_reg, *slot, rv64);
        }
        Operand::Register(Register::Physical(phys)) => {
            let phys_num = riscv_reg_num(phys.name)?;
            if phys_num != dst_reg {
                code.extend_from_slice(&emit_mv(dst_reg, phys_num));
            }
        }
    }
    Ok(())
}

fn store_result(
    code: &mut Vec<u8>,
    src_reg: u8,
    dst: &lamina_mir::Register,
    stack_slots: &HashMap<lamina_mir::VirtualReg, i32>,
    rv64: bool,
) -> Result<(), RasError> {
    match dst {
        Register::Virtual(vreg) => {
            let slot = stack_slots.get(vreg).ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for vreg {:?}", vreg))
            })?;
            emit_store_to_fp(code, src_reg, *slot, rv64);
        }
        Register::Physical(phys) => {
            let phys_num = riscv_reg_num(phys.name)?;
            if phys_num != src_reg {
                code.extend_from_slice(&emit_mv(phys_num, src_reg));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ABI name → register number
// ---------------------------------------------------------------------------
fn riscv_reg_num(name: &str) -> Result<u8, RasError> {
    match name {
        "x0" | "zero" => Ok(0),
        "x1" | "ra" => Ok(1),
        "x2" | "sp" => Ok(2),
        "x3" | "gp" => Ok(3),
        "x4" | "tp" => Ok(4),
        "x5" | "t0" => Ok(5),
        "x6" | "t1" => Ok(6),
        "x7" | "t2" => Ok(7),
        "x8" | "s0" | "fp" => Ok(8),
        "x9" | "s1" => Ok(9),
        "x10" | "a0" => Ok(10),
        "x11" | "a1" => Ok(11),
        "x12" | "a2" => Ok(12),
        "x13" | "a3" => Ok(13),
        "x14" | "a4" => Ok(14),
        "x15" | "a5" => Ok(15),
        "x16" | "a6" => Ok(16),
        "x17" | "a7" => Ok(17),
        "x18" | "s2" => Ok(18),
        "x19" | "s3" => Ok(19),
        "x20" | "s4" => Ok(20),
        "x21" | "s5" => Ok(21),
        "x22" | "s6" => Ok(22),
        "x23" | "s7" => Ok(23),
        "x24" | "s8" => Ok(24),
        "x25" | "s9" => Ok(25),
        "x26" | "s10" => Ok(26),
        "x27" | "s11" => Ok(27),
        "x28" | "t3" => Ok(28),
        "x29" | "t4" => Ok(29),
        "x30" | "t5" => Ok(30),
        "x31" | "t6" => Ok(31),
        _ => Err(RasError::EncodingError(format!(
            "Unknown RISC-V register name: {}",
            name
        ))),
    }
}

// ---------------------------------------------------------------------------
// MIR instruction encoder
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn encode_instruction(
    code: &mut Vec<u8>,
    inst: &lamina_mir::Instruction,
    stack_slots: &HashMap<lamina_mir::VirtualReg, i32>,
    frame_size: usize,
    ret_ty: Option<&lamina_mir::MirType>,
    rv64: bool,
    // branch fixup tracking
    block_offsets: &HashMap<String, usize>,
    branch_fixups: &mut Vec<(usize, String, BranchFixupOp)>,
    base_offset: usize, // offset of `code[0]` in the final buffer
) -> Result<(), RasError> {

    match inst {
        // ---- IntBinary ----
        Instruction::IntBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let is_i32 = matches!(ty, MirType::Scalar(ScalarType::I32));

            materialize_operand(code, lhs, REG_T0, stack_slots, rv64)?;
            materialize_operand(code, rhs, REG_T1, stack_slots, rv64)?;

            let result = REG_T2;
            match op {
                IntBinOp::Add => {
                    if is_i32 {
                        code.extend_from_slice(&emit_addw(result, REG_T0, REG_T1));
                    } else {
                        code.extend_from_slice(&emit_add(result, REG_T0, REG_T1));
                    }
                }
                IntBinOp::Sub => {
                    if is_i32 {
                        code.extend_from_slice(&emit_subw(result, REG_T0, REG_T1));
                    } else {
                        code.extend_from_slice(&emit_sub(result, REG_T0, REG_T1));
                    }
                }
                IntBinOp::Mul => {
                    if is_i32 {
                        code.extend_from_slice(&emit_mulw(result, REG_T0, REG_T1));
                    } else {
                        code.extend_from_slice(&emit_mul(result, REG_T0, REG_T1));
                    }
                }
                IntBinOp::SDiv => code.extend_from_slice(&emit_div(result, REG_T0, REG_T1)),
                IntBinOp::UDiv => code.extend_from_slice(&emit_divu(result, REG_T0, REG_T1)),
                IntBinOp::SRem => code.extend_from_slice(&emit_rem(result, REG_T0, REG_T1)),
                IntBinOp::URem => code.extend_from_slice(&emit_remu(result, REG_T0, REG_T1)),
                IntBinOp::And => code.extend_from_slice(&emit_and(result, REG_T0, REG_T1)),
                IntBinOp::Or => code.extend_from_slice(&emit_or(result, REG_T0, REG_T1)),
                IntBinOp::Xor => code.extend_from_slice(&emit_xor(result, REG_T0, REG_T1)),
                IntBinOp::Shl => {
                    if is_i32 {
                        code.extend_from_slice(&emit_sllw(result, REG_T0, REG_T1));
                    } else {
                        code.extend_from_slice(&emit_sll(result, REG_T0, REG_T1));
                    }
                }
                IntBinOp::LShr => {
                    if is_i32 {
                        code.extend_from_slice(&emit_srlw(result, REG_T0, REG_T1));
                    } else {
                        code.extend_from_slice(&emit_srl(result, REG_T0, REG_T1));
                    }
                }
                IntBinOp::AShr => {
                    if is_i32 {
                        code.extend_from_slice(&emit_sraw(result, REG_T0, REG_T1));
                    } else {
                        code.extend_from_slice(&emit_sra(result, REG_T0, REG_T1));
                    }
                }
            }
            store_result(code, result, dst, stack_slots, rv64)?;
        }

        // ---- IntCmp ----
        Instruction::IntCmp {
            op,
            ty: _,
            dst,
            lhs,
            rhs,
        } => {
            materialize_operand(code, lhs, REG_T0, stack_slots, rv64)?;
            materialize_operand(code, rhs, REG_T1, stack_slots, rv64)?;

            // Produce 0/1 in REG_T2
            let r = REG_T2;
            match op {
                IntCmpOp::Eq => {
                    // sub t2, t0, t1; seqz t2, t2
                    code.extend_from_slice(&emit_sub(r, REG_T0, REG_T1));
                    code.extend_from_slice(&emit_sltiu_1(r, r));
                }
                IntCmpOp::Ne => {
                    // sub t2, t0, t1; snez t2, t2  (sltu t2, x0, t2)
                    code.extend_from_slice(&emit_sub(r, REG_T0, REG_T1));
                    code.extend_from_slice(&emit_sltu(r, REG_ZERO, r));
                }
                IntCmpOp::SLt => {
                    code.extend_from_slice(&emit_slt(r, REG_T0, REG_T1));
                }
                IntCmpOp::ULt => {
                    code.extend_from_slice(&emit_sltu(r, REG_T0, REG_T1));
                }
                IntCmpOp::SGt => {
                    // t0 > t1  ↔  t1 < t0
                    code.extend_from_slice(&emit_slt(r, REG_T1, REG_T0));
                }
                IntCmpOp::UGt => {
                    code.extend_from_slice(&emit_sltu(r, REG_T1, REG_T0));
                }
                IntCmpOp::SGe => {
                    // t0 >= t1  ↔  NOT(t0 < t1)
                    code.extend_from_slice(&emit_slt(r, REG_T0, REG_T1));
                    code.extend_from_slice(&emit_xori_1(r, r));
                }
                IntCmpOp::UGe => {
                    code.extend_from_slice(&emit_sltu(r, REG_T0, REG_T1));
                    code.extend_from_slice(&emit_xori_1(r, r));
                }
                IntCmpOp::SLe => {
                    // t0 <= t1  ↔  NOT(t1 < t0)
                    code.extend_from_slice(&emit_slt(r, REG_T1, REG_T0));
                    code.extend_from_slice(&emit_xori_1(r, r));
                }
                IntCmpOp::ULe => {
                    code.extend_from_slice(&emit_sltu(r, REG_T1, REG_T0));
                    code.extend_from_slice(&emit_xori_1(r, r));
                }
            }
            store_result(code, r, dst, stack_slots, rv64)?;
        }

        // ---- Ret ----
        Instruction::Ret { value } => {
            if let Some(v) = value
                && ret_ty.is_some()
            {
                // Move return value into a0 (x10)
                materialize_operand(code, v, ARG_REGS[0], stack_slots, rv64)?;
            }
            let epilogue = encode_epilogue(frame_size, rv64)?;
            code.extend_from_slice(&epilogue);
        }

        // ---- Jmp ----
        Instruction::Jmp { target } => {
            if let Some(&target_off) = block_offsets.get(target.as_str()) {
                // Target already known: direct encode
                let from_abs = base_offset + code.len();
                let delta = target_off as i64 - from_abs as i64;
                if !(-0x10_0000..0x10_0000).contains(&delta) {
                    return Err(RasError::EncodingError(format!(
                        "JAL offset out of range: {} bytes",
                        delta
                    )));
                }
                code.extend_from_slice(&emit_jal_zero(delta as i32));
            } else {
                // Forward reference: emit placeholder, record fixup
                let fixup_abs = base_offset + code.len();
                branch_fixups.push((fixup_abs, target.clone(), BranchFixupOp::Jal));
                code.extend_from_slice(&emit_jal_zero(0)); // placeholder
            }
        }

        // ---- Br ----
        Instruction::Br {
            cond,
            true_target,
            false_target,
        } => {
            // Load condition register into t0
            let cond_operand = Op::Register(cond.clone());
            materialize_operand(code, &cond_operand, REG_T0, stack_slots, rv64)?;

            // BNE t0, x0, true_target  (if t0 != 0, branch to true)
            // JAL x0, false_target
            let bne_abs = base_offset + code.len();
            code.extend_from_slice(&emit_bne(REG_T0, REG_ZERO, 0)); // placeholder
            branch_fixups.push((bne_abs, true_target.clone(), BranchFixupOp::Bne));

            let jal_abs = base_offset + code.len();
            code.extend_from_slice(&emit_jal_zero(0)); // placeholder
            branch_fixups.push((jal_abs, false_target.clone(), BranchFixupOp::Jal));
        }

        // ---- Call ----
        Instruction::Call { name, args, ret } => {
            // Move args into a0-a7
            for (i, arg) in args.iter().enumerate() {
                if i >= ARG_REGS.len() {
                    return Err(RasError::EncodingError(
                        "RISC-V JIT: more than 8 call arguments not yet supported".to_string(),
                    ));
                }
                materialize_operand(code, arg, ARG_REGS[i], stack_slots, rv64)?;
            }

            // JAL ra, target (internal calls only; external calls need JALR via GOT)
            let target_name = name.trim_start_matches('@');
            let jal_abs = base_offset + code.len();
            branch_fixups.push((jal_abs, target_name.to_string(), BranchFixupOp::JalRa));
            code.extend_from_slice(&j_type(0, REG_RA)); // placeholder

            // Capture return value from a0
            if let Some(dst_reg) = ret {
                store_result(code, ARG_REGS[0], dst_reg, stack_slots, rv64)?;
            }
        }

        // ---- TailCall ----
        Instruction::TailCall { name, args } => {
            for (i, arg) in args.iter().enumerate() {
                if i >= ARG_REGS.len() {
                    return Err(RasError::EncodingError(
                        "RISC-V JIT: more than 8 tail-call arguments not yet supported".to_string(),
                    ));
                }
                materialize_operand(code, arg, ARG_REGS[i], stack_slots, rv64)?;
            }
            // Emit epilogue then replace the trailing RET with a JAL to target.
            let epilogue = encode_epilogue(frame_size, rv64)?;
            code.extend_from_slice(&epilogue);
            // Truncate the 4-byte RET that encode_epilogue appends
            let new_len = code.len() - 4;
            code.truncate(new_len);
            let target_name = name.trim_start_matches('@');
            let jal_abs = base_offset + code.len();
            branch_fixups.push((jal_abs, target_name.to_string(), BranchFixupOp::Jal));
            code.extend_from_slice(&emit_jal_zero(0)); // placeholder
        }

        // ---- Unreachable ----
        Instruction::Unreachable => {
            // EBREAK — trap
            code.extend_from_slice(&i_type(1, 0, 0, 0, 0x73));
        }

        // ---- Unimplemented ----
        other => {
            return Err(RasError::EncodingError(format!(
                "RISC-V JIT: instruction not yet implemented: {:?}",
                other
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Branch fixup type
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
pub(crate) enum BranchFixupOp {
    Jal,   // JAL x0, offset  (unconditional, offset in J-type)
    JalRa, // JAL ra, offset  (call, offset in J-type)
    Bne,   // BNE rs1, x0, offset  (rs1 already baked in the placeholder)
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
#[cfg(feature = "encoder")]
pub fn compile_mir_riscv_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    function_name: Option<&str>,
    rv64: bool,
) -> Result<(Vec<u8>, HashMap<String, usize>), RasError> {
    use lamina_codegen::riscv::RiscVFrame;
    use lamina_mir::Register;

    let _ = assembler; // not needed for register allocation here

    let mut code: Vec<u8> = Vec::new();
    let mut function_offsets: HashMap<String, usize> = HashMap::new();

    // Collect functions to compile (sorted for deterministic layout)
    let functions_to_compile: Vec<(String, &lamina_mir::Function)> =
        if let Some(name) = function_name {
            module
                .functions
                .get(name)
                .map(|f| vec![(name.to_string(), f)])
                .unwrap_or_default()
        } else {
            let mut names: Vec<String> = module.functions.keys().cloned().collect();
            names.sort();
            names
                .into_iter()
                .filter_map(|n| module.functions.get(&n).map(|f| (n, f)))
                .collect()
        };

    let reserve: usize = functions_to_compile
        .iter()
        .map(|(_, f)| {
            128 + f
                .blocks
                .iter()
                .map(|b| b.instructions.len() * 12)
                .sum::<usize>()
        })
        .sum::<usize>()
        .max(256);
    code.reserve(reserve);

    // Cross-function fixups (for Call / TailCall between sibling functions)
    struct FnFixup {
        abs_offset: usize,
        target_name: String,
        op: BranchFixupOp,
    }
    let mut cross_fn_fixups: Vec<FnFixup> = Vec::new();

    for (func_name, func) in &functions_to_compile {
        function_offsets.insert(func_name.clone(), code.len());

        // --- Build stack slot map ---
        let mut stack_slots: HashMap<lamina_mir::VirtualReg, i32> = HashMap::new();
        let mut def_regs: std::collections::HashSet<lamina_mir::VirtualReg> =
            std::collections::HashSet::new();
        let mut used_regs: std::collections::HashSet<lamina_mir::VirtualReg> =
            std::collections::HashSet::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                {
                    def_regs.insert(*vreg);
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg {
                        used_regs.insert(*vreg);
                    }
                }
            }
        }

        for vreg in &def_regs {
            if !stack_slots.contains_key(vreg) {
                let idx = stack_slots.len();
                stack_slots.insert(*vreg, RiscVFrame::calculate_stack_offset(idx));
            }
        }
        for vreg in &used_regs {
            if !def_regs.contains(vreg) && !stack_slots.contains_key(vreg) {
                let idx = stack_slots.len();
                stack_slots.insert(*vreg, RiscVFrame::calculate_stack_offset(idx));
            }
        }

        let frame_size = stack_slots.len() * 8;

        // --- Prologue ---
        let prologue = encode_prologue(frame_size, rv64)?;
        code.extend_from_slice(&prologue);

        // --- Spill function parameters (a0–a7 → stack slots) ---
        for (i, param) in func.sig.params.iter().enumerate() {
            if let Register::Virtual(vreg) = &param.reg
                && let Some(slot) = stack_slots.get(vreg)
                && i < ARG_REGS.len()
            {
                emit_store_to_fp(&mut code, ARG_REGS[i], *slot, rv64);
            }
            // Stack args beyond 8 not yet supported
        }

        // --- Compile blocks ---
        let func_base = function_offsets[func_name.as_str()];

        // First pass: record block offsets (relative to start of `code`)
        // We'll do a single-pass encoding with forward-branch fixups tracked locally.
        let mut block_offsets: HashMap<String, usize> = HashMap::new();
        let mut local_fixups: Vec<(usize, String, BranchFixupOp)> = Vec::new();

        // We need a per-function local code buffer so absolute offsets work.
        // We'll emit into `code` directly and track via code.len().
        for block in &func.blocks {
            block_offsets.insert(block.label.clone(), code.len());

            for inst in block.body() {
                encode_instruction(
                    &mut code,
                    inst,
                    &stack_slots,
                    frame_size,
                    func.sig.ret_ty.as_ref(),
                    rv64,
                    &block_offsets,
                    &mut local_fixups,
                    0, // absolute offsets — offsets are already absolute into `code`
                )?;
            }

            if let Some(term) = block.terminator() {
                encode_instruction(
                    &mut code,
                    term,
                    &stack_slots,
                    frame_size,
                    func.sig.ret_ty.as_ref(),
                    rv64,
                    &block_offsets,
                    &mut local_fixups,
                    0,
                )?;
            }
        }

        // Patch intra-function branch fixups
        for (abs_off, target, op) in local_fixups {
            if let Some(&target_off) = block_offsets.get(&target) {
                patch_branch(&mut code, abs_off, target_off, op)?;
            } else {
                // Maybe an inter-function call target — defer
                cross_fn_fixups.push(FnFixup {
                    abs_offset: abs_off,
                    target_name: target,
                    op,
                });
            }
        }

        let _ = func_base;
    }

    // Patch cross-function call fixups
    for fixup in cross_fn_fixups {
        let target_off = *function_offsets.get(&fixup.target_name).ok_or_else(|| {
            RasError::EncodingError(format!(
                "RISC-V JIT: unresolved call target '{}'",
                fixup.target_name
            ))
        })?;
        patch_branch(&mut code, fixup.abs_offset, target_off, fixup.op)?;
    }

    Ok((code, function_offsets))
}

// ---------------------------------------------------------------------------
// Branch patch helper
// ---------------------------------------------------------------------------
fn patch_branch(
    code: &mut [u8],
    patch_abs: usize,
    target_abs: usize,
    op: BranchFixupOp,
) -> Result<(), RasError> {
    let delta = target_abs as i64 - patch_abs as i64;

    match op {
        BranchFixupOp::Jal => {
            if !(-0x10_0000i64..0x10_0000i64).contains(&delta) {
                return Err(RasError::EncodingError(format!(
                    "RISC-V JAL fixup: offset {} out of ±1MiB range",
                    delta
                )));
            }
            let word = j_type(delta as i32, REG_ZERO);
            code[patch_abs..patch_abs + 4].copy_from_slice(&word);
        }
        BranchFixupOp::JalRa => {
            if !(-0x10_0000i64..0x10_0000i64).contains(&delta) {
                return Err(RasError::EncodingError(format!(
                    "RISC-V JAL(ra) fixup: offset {} out of ±1MiB range",
                    delta
                )));
            }
            let word = j_type(delta as i32, REG_RA);
            code[patch_abs..patch_abs + 4].copy_from_slice(&word);
        }
        BranchFixupOp::Bne => {
            if !(-4096i64..4096i64).contains(&delta) {
                return Err(RasError::EncodingError(format!(
                    "RISC-V BNE fixup: offset {} out of ±4KiB range",
                    delta
                )));
            }
            // Re-encode BNE t0, x0, delta preserving original rs1/rs2
            let word = b_type(delta as i32, REG_ZERO, REG_T0, 0x1);
            code[patch_abs..patch_abs + 4].copy_from_slice(&word);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prologue_epilogue_round_trip() {
        let p = encode_prologue(0, true).unwrap();
        let e = encode_epilogue(0, true).unwrap();
        // Prologue: 4 instructions = 16 bytes; epilogue: 4 instructions = 16 bytes
        assert_eq!(p.len(), 16);
        assert_eq!(e.len(), 16);
    }

    #[test]
    fn emit_li64_small() {
        let mut code = Vec::new();
        emit_li64(&mut code, 10, 42);
        assert_eq!(code.len(), 4); // single ADDI
        let word = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(word & 0x7F, 0x13); // OP-IMM opcode
        assert_eq!((word >> 20) as i32, 42);
    }

    #[test]
    fn emit_li64_lui_addi() {
        let mut code = Vec::new();
        emit_li64(&mut code, 10, 0x1000); // 4096 — needs LUI+ADDI
        assert!(code.len() >= 4);
    }

    #[test]
    fn riscv_reg_num_roundtrip() {
        assert_eq!(riscv_reg_num("a0").unwrap(), 10);
        assert_eq!(riscv_reg_num("sp").unwrap(), 2);
        assert_eq!(riscv_reg_num("x31").unwrap(), 31);
        assert!(riscv_reg_num("bogus").is_err());
    }
}
