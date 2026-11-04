//! Run MIR through the AArch64 JIT on the host and check what the code returns.
#![cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]

use lamina_mir::{
    block::Block,
    function::{Function, Parameter, Signature},
    instruction::{AddressMode, Immediate, Instruction, IntBinOp, MemoryAttrs, Operand},
    module::Module,
    register::{Register, VirtualReg},
    types::{MirType, ScalarType},
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::{ExecutableMemory, RasAssembler};

const I64: MirType = MirType::Scalar(ScalarType::I64);

#[cfg(target_os = "macos")]
const HOST_OS: TargetOperatingSystem = TargetOperatingSystem::MacOS;
#[cfg(target_os = "linux")]
const HOST_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;

fn v(id: u32) -> Register {
    Register::Virtual(VirtualReg::gpr(id))
}

fn imm(value: i64) -> Operand {
    Operand::Immediate(Immediate::I64(value))
}

fn reg(id: u32) -> Operand {
    Operand::Register(v(id))
}

/// Registers sort by name in the frame, so these push id 99 past the short STUR range.
fn fill_frame(block: &mut Block, ids: std::ops::Range<u32>) {
    for id in ids {
        block.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: I64,
            dst: v(id),
            lhs: imm(i64::from(id)),
            rhs: imm(0),
        });
    }
}

fn run(module: &Module, entry: &str) -> i64 {
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    let (code, offsets) = assembler
        .compile_mir_to_binary_function(module, None)
        .expect("compile");
    let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
    memory.write_code(&code).expect("write");
    memory.make_executable().expect("protect");
    let start = unsafe { memory.code_start().add(offsets[entry]) };
    let function: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(start) };
    unsafe { function() }
}

fn at_v1(offset: i16) -> AddressMode {
    AddressMode::BaseOffset {
        address_space: Default::default(),
        base: v(1),
        offset,
    }
}

fn module_of(functions: Vec<Function>) -> Module {
    let mut module = Module::new("jit_run");
    for function in functions {
        module.add_function(function);
    }
    module
}

#[test]
fn select_keeps_false_value_when_cond_slot_is_far() {
    let mut entry = Block::new("entry");
    fill_frame(&mut entry, 0..98);
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(98),
        lhs: imm(1234),
        rhs: imm(0),
    });
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(99),
        lhs: imm(0),
        rhs: imm(0),
    });
    entry.push(Instruction::Select {
        ty: I64,
        dst: v(100),
        cond: v(99),
        true_val: imm(7),
        false_val: reg(98),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(100)),
    });
    let mut function = Function::new(Signature::new("pick").with_return(I64));
    function.add_block(entry);
    assert_eq!(run(&module_of(vec![function]), "pick"), 1234);
}

#[test]
fn stack_argument_lands_in_far_slot() {
    // id 99 sorts last, so the tenth argument gets the deepest slot.
    let params: Vec<Parameter> = (200..209)
        .chain([99])
        .map(|id| Parameter::new(v(id), I64))
        .collect();
    let mut callee_entry = Block::new("entry");
    fill_frame(&mut callee_entry, 0..99);
    callee_entry.push(Instruction::Ret {
        value: Some(reg(99)),
    });
    let mut callee = Function::new(
        Signature::new("callee")
            .with_params(params)
            .with_return(I64),
    );
    callee.add_block(callee_entry);

    let mut caller_entry = Block::new("entry");
    caller_entry.push(Instruction::Call {
        name: "callee".to_string(),
        args: (0..10).map(|n| imm(n + 500)).collect(),
        ret: Some(v(0)),
    });
    caller_entry.push(Instruction::Ret {
        value: Some(reg(0)),
    });
    let mut caller = Function::new(Signature::new("caller").with_return(I64));
    caller.add_block(caller_entry);

    assert_eq!(run(&module_of(vec![callee, caller]), "caller"), 509);
}

#[test]
fn frame_past_imm12_keeps_values() {
    let mut entry = Block::new("entry");
    fill_frame(&mut entry, 0..800);
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(800),
        lhs: reg(799),
        rhs: reg(99),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(800)),
    });
    let mut function = Function::new(Signature::new("big").with_return(I64));
    function.add_block(entry);
    assert_eq!(run(&module_of(vec![function]), "big"), 898);
}

#[test]
fn copy_and_bitcast_move_the_value() {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(0),
        src: imm(41),
    });
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(1),
        src: reg(0),
    });
    entry.push(Instruction::Bitcast {
        from: I64,
        to: I64,
        dst: v(2),
        src: reg(1),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(2)),
    });
    let mut function = Function::new(Signature::new("copy").with_return(I64));
    function.add_block(entry);
    assert_eq!(run(&module_of(vec![function]), "copy"), 41);
}

#[test]
fn stack_addr_reaches_reserved_block() {
    let slot = VirtualReg::gpr(0);
    let mut entry = Block::new("entry");
    entry.push(Instruction::StackAddr { dst: v(1), slot });
    for (offset, value) in [(0i16, 5i64), (24, 37)] {
        entry.push(Instruction::Store {
            ty: I64,
            src: imm(value),
            addr: at_v1(offset),
            attrs: MemoryAttrs::default(),
        });
    }
    entry.push(Instruction::Load {
        ty: I64,
        dst: v(2),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::Load {
        ty: I64,
        dst: v(3),
        addr: at_v1(24),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(4),
        lhs: reg(2),
        rhs: reg(3),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(4)),
    });
    let mut function = Function::new(Signature::new("block").with_return(I64));
    function.stack_reservations.insert(slot, 3);
    function.add_block(entry);
    assert_eq!(run(&module_of(vec![function]), "block"), 42);
}

fn convert(from: ScalarType, to: ScalarType, signed: bool, value: i64) -> i64 {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(0),
        src: imm(value),
    });
    entry.push(Instruction::IntConvert {
        from: MirType::Scalar(from),
        to: MirType::Scalar(to),
        signed,
        dst: v(1),
        src: reg(0),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(1)),
    });
    let mut function = Function::new(Signature::new("convert").with_return(I64));
    function.add_block(entry);
    run(&module_of(vec![function]), "convert")
}

#[test]
fn int_convert_extends_and_truncates() {
    assert_eq!(convert(ScalarType::I8, ScalarType::I64, true, 0xFF), -1);
    assert_eq!(convert(ScalarType::I8, ScalarType::I64, false, 0xFF), 0xFF);
    assert_eq!(
        convert(ScalarType::I32, ScalarType::I64, true, 0xFFFF_FFFF),
        -1
    );
    assert_eq!(
        convert(ScalarType::I32, ScalarType::I64, false, 0xFFFF_FFFF),
        0xFFFF_FFFF
    );
    assert_eq!(convert(ScalarType::I64, ScalarType::I8, true, 0x1_7F), 0x7F);
    assert_eq!(
        convert(ScalarType::I64, ScalarType::I8, false, 0x1_80),
        -128
    );
}
