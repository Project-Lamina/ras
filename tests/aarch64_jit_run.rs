//! Run MIR through the AArch64 JIT on the host and check what the code returns.
#![cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]

use lamina_mir::{
    block::Block,
    function::{Function, Parameter, Signature},
    instruction::{AddressMode, Immediate, Instruction, IntBinOp, IntCmpOp, MemoryAttrs, Operand},
    module::{Global, Module},
    register::{Register, VirtualReg},
    types::{MirType, ScalarType},
};
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};
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

#[cfg(target_os = "macos")]
#[test]
fn writebyte_reports_one_byte_written() {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Call {
        name: "lamina.writebyte".to_string(),
        args: vec![imm(10)],
        ret: Some(v(0)),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(0)),
    });
    let mut function = Function::new(Signature::new("write").with_return(I64));
    function.add_block(entry);
    assert_eq!(run(&module_of(vec![function]), "write"), 1);
}

#[test]
fn aggregate_abi_is_refused_not_miscompiled() {
    let mut entry = Block::new("entry");
    entry.push(Instruction::StackAddr {
        dst: v(1),
        slot: VirtualReg::gpr(0),
    });
    entry.push(Instruction::Ret {
        value: Some(imm(0)),
    });
    let mut function = Function::new(
        Signature::new("wide")
            .with_params(vec![Parameter::new(v(0), I64)])
            .with_return(I64),
    );
    function.add_block(entry);
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    let error = assembler
        .compile_mir_to_binary_function(&module_of(vec![function]), None)
        .expect_err("refused");
    assert!(error.to_string().contains("held in memory"), "{error}");
}

#[test]
fn extern_declaration_calls_the_loaded_symbol() {
    let mut labs = Function::new(
        Signature::new("labs")
            .with_params(vec![Parameter::new(v(0), I64)])
            .with_return(I64),
    );
    labs.blocks.clear();
    let mut entry = Block::new("entry");
    entry.push(Instruction::Call {
        name: "labs".to_string(),
        args: vec![imm(-42)],
        ret: Some(v(0)),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(0)),
    });
    let mut main = Function::new(Signature::new("main").with_return(I64));
    main.add_block(entry);
    assert_eq!(run(&module_of(vec![labs, main]), "main"), 42);
}

#[test]
fn extern_narrow_return_ignores_upper_bits() {
    let mut abs = Function::new(
        Signature::new("abs")
            .with_params(vec![Parameter::new(v(0), MirType::Scalar(ScalarType::I32))])
            .with_return(MirType::Scalar(ScalarType::I32)),
    );
    abs.blocks.clear();
    let mut entry = Block::new("entry");
    entry.push(Instruction::Call {
        name: "abs".to_string(),
        args: vec![imm(-7)],
        ret: Some(v(0)),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(0)),
    });
    let mut main = Function::new(Signature::new("main").with_return(I64));
    main.add_block(entry);
    assert_eq!(run(&module_of(vec![abs, main]), "main"), 7);
}

#[test]
fn global_reads_initializer_and_keeps_stores() {
    let mut entry = Block::new("entry");
    entry.push(Instruction::SymbolAddress {
        dst: v(1),
        symbol: "counter".to_string(),
    });
    entry.push(Instruction::Load {
        ty: I64,
        dst: v(2),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(3),
        lhs: reg(2),
        rhs: imm(2),
    });
    entry.push(Instruction::Store {
        ty: I64,
        src: reg(3),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::Load {
        ty: I64,
        dst: v(4),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(4)),
    });
    let mut function = Function::new(Signature::new("bump").with_return(I64));
    function.add_block(entry);
    let mut module = module_of(vec![function]);
    let global = Global::new("counter", I64, PointerWidth::Bits64)
        .with_initializer(40i64.to_le_bytes().to_vec());
    module.globals.insert("counter".to_string(), global);
    assert_eq!(run(&module, "bump"), 42);
}

fn switch_on(value: i64) -> i64 {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(0),
        src: imm(value),
    });
    entry.push(Instruction::Switch {
        value: v(0),
        cases: vec![(1, "one".to_string()), (-70000, "far".to_string())],
        default: "other".to_string(),
    });
    let mut function = Function::new(Signature::new("switch").with_return(I64));
    function.add_block(entry);
    for (label, result) in [("one", 10), ("far", 20), ("other", 30)] {
        let mut block = Block::new(label);
        block.push(Instruction::Ret {
            value: Some(imm(result)),
        });
        function.add_block(block);
    }
    run(&module_of(vec![function]), "switch")
}

#[test]
fn switch_takes_matching_case_or_default() {
    assert_eq!(switch_on(1), 10);
    assert_eq!(switch_on(-70000), 20);
    assert_eq!(switch_on(5), 30);
}

fn looping_sum(name: &str, limit: i64) -> Function {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(0),
        src: imm(0),
    });
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(1),
        src: imm(0),
    });
    entry.push(Instruction::Jmp {
        target: "head".to_string(),
    });
    let mut head = Block::new("head");
    head.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(0),
        lhs: reg(0),
        rhs: reg(1),
    });
    head.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(1),
        lhs: reg(1),
        rhs: imm(1),
    });
    head.push(Instruction::IntCmp {
        op: IntCmpOp::SLt,
        ty: I64,
        dst: v(2),
        lhs: reg(1),
        rhs: imm(limit),
    });
    head.push(Instruction::Br {
        cond: v(2),
        true_target: "head".to_string(),
        false_target: "done".to_string(),
    });
    let mut done = Block::new("done");
    done.push(Instruction::Ret {
        value: Some(reg(0)),
    });
    let mut function = Function::new(Signature::new(name).with_return(I64));
    function.add_block(entry);
    function.add_block(head);
    function.add_block(done);
    function
}

#[test]
fn hot_values_survive_a_call_that_uses_the_same_registers() {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(0),
        src: imm(1000),
    });
    entry.push(Instruction::Jmp {
        target: "again".to_string(),
    });
    let mut again = Block::new("again");
    again.push(Instruction::Call {
        name: "inner".to_string(),
        args: Vec::new(),
        ret: Some(v(1)),
    });
    again.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(0),
        lhs: reg(0),
        rhs: reg(1),
    });
    again.push(Instruction::IntCmp {
        op: IntCmpOp::SLt,
        ty: I64,
        dst: v(2),
        lhs: reg(0),
        rhs: imm(1100),
    });
    again.push(Instruction::Br {
        cond: v(2),
        true_target: "again".to_string(),
        false_target: "out".to_string(),
    });
    let mut out = Block::new("out");
    out.push(Instruction::Ret {
        value: Some(reg(0)),
    });
    let mut outer = Function::new(Signature::new("outer").with_return(I64));
    outer.add_block(entry);
    outer.add_block(again);
    outer.add_block(out);
    // inner adds 0..5 = 10, so the loop lands on 1100
    let module = module_of(vec![outer, looping_sum("inner", 5)]);
    assert_eq!(run(&module, "outer"), 1100);
}

#[test]
fn narrow_value_in_a_register_reads_at_its_width() {
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: I64,
        dst: v(0),
        src: imm(0x1_0000_0005),
    });
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: MirType::Scalar(ScalarType::I32),
        dst: v(1),
        lhs: reg(0),
        rhs: imm(-10),
    });
    entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: MirType::Scalar(ScalarType::I32),
        dst: v(2),
        lhs: reg(1),
        rhs: reg(1),
    });
    entry.push(Instruction::Ret {
        value: Some(reg(2)),
    });
    let mut function =
        Function::new(Signature::new("narrow").with_return(MirType::Scalar(ScalarType::I32)));
    function.add_block(entry);
    assert_eq!(run(&module_of(vec![function]), "narrow") as i32, -10);
}
