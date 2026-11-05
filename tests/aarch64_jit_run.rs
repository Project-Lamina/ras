//! Run MIR through the AArch64 JIT on the host and check what the code returns.
#![cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]

use lamina_mir::{
    block::Block,
    function::{Function, Parameter, Signature},
    instruction::{
        AddressMode, AtomicBinOp, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, Instruction,
        IntBinOp, IntCmpOp, IntOverflowOp, IntUnOp, MemoryAttrs, MemoryOrdering, Operand,
    },
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

const F64_TY: MirType = MirType::Scalar(ScalarType::F64);
const F32_TY: MirType = MirType::Scalar(ScalarType::F32);

fn f64_imm(value: f64) -> Operand {
    Operand::Immediate(Immediate::F64(value))
}

/// Run a block that ends by returning a register, as its raw 64 bits.
fn block_result(build: impl FnOnce(&mut Block)) -> i64 {
    let mut entry = Block::new("entry");
    build(&mut entry);
    let mut function = Function::new(Signature::new("result").with_return(I64));
    function.add_block(entry);
    run(&module_of(vec![function]), "result")
}

fn ret(entry: &mut Block, id: u32) {
    entry.push(Instruction::Ret {
        value: Some(reg(id)),
    });
}

#[test]
fn float_arithmetic_and_constants() {
    let bits = block_result(|entry| {
        entry.push(Instruction::FloatBinary {
            op: FloatBinOp::FMul,
            ty: F64_TY,
            dst: v(0),
            lhs: f64_imm(1.5),
            rhs: f64_imm(4.0),
        });
        entry.push(Instruction::FloatBinary {
            op: FloatBinOp::FSub,
            ty: F64_TY,
            dst: v(1),
            lhs: reg(0),
            rhs: f64_imm(0.25),
        });
        entry.push(Instruction::FloatUnary {
            op: FloatUnOp::FNeg,
            ty: F64_TY,
            dst: v(2),
            src: reg(1),
        });
        ret(entry, 2);
    });
    assert_eq!(f64::from_bits(bits as u64), -5.75);
}

fn compares(op: FloatCmpOp, lhs: f64, rhs: f64) -> i64 {
    block_result(|entry| {
        entry.push(Instruction::FloatCmp {
            op,
            ty: F64_TY,
            dst: v(0),
            lhs: f64_imm(lhs),
            rhs: f64_imm(rhs),
        });
        ret(entry, 0);
    })
}

#[test]
fn float_compares_are_false_on_nan_except_ne() {
    for (op, less_than) in [
        (FloatCmpOp::Eq, 0),
        (FloatCmpOp::Ne, 1),
        (FloatCmpOp::Lt, 1),
        (FloatCmpOp::Le, 1),
        (FloatCmpOp::Gt, 0),
        (FloatCmpOp::Ge, 0),
    ] {
        assert_eq!(compares(op, 1.0, 2.0), less_than, "{op:?}");
        let on_nan = i64::from(matches!(op, FloatCmpOp::Ne));
        assert_eq!(compares(op, f64::NAN, 2.0), on_nan, "{op:?} nan");
    }
}

fn to_int(to: ScalarType, signed: bool, value: f64) -> i64 {
    block_result(|entry| {
        entry.push(Instruction::FloatToInt {
            from: F64_TY,
            to: MirType::Scalar(to),
            signed,
            dst: v(0),
            src: f64_imm(value),
        });
        ret(entry, 0);
    })
}

#[test]
fn float_to_int_truncates_and_saturates() {
    assert_eq!(to_int(ScalarType::I64, true, -2.9), -2);
    assert_eq!(to_int(ScalarType::I64, true, f64::NAN), 0);
    assert_eq!(to_int(ScalarType::I8, true, 1000.0), 127);
    assert_eq!(to_int(ScalarType::I8, true, -1000.0), -128);
    assert_eq!(to_int(ScalarType::I8, false, 1000.0), 255);
    assert_eq!(to_int(ScalarType::I32, true, 3e10), i64::from(i32::MAX));
}

#[test]
fn int_to_float_and_back_through_f32() {
    let bits = block_result(|entry| {
        entry.push(Instruction::IntToFloat {
            from: I64,
            to: F32_TY,
            signed: true,
            dst: v(0),
            src: imm(-3),
        });
        entry.push(Instruction::FloatConvert {
            from: F32_TY,
            to: F64_TY,
            dst: v(1),
            src: reg(0),
        });
        ret(entry, 1);
    });
    assert_eq!(f64::from_bits(bits as u64), -3.0);
}

fn narrow_op(op: IntBinOp, ty: ScalarType, lhs: i64, rhs: i64) -> i64 {
    let ty = MirType::Scalar(ty);
    block_result(|entry| {
        let address = VirtualReg::gpr(50);
        entry.push(Instruction::StackAddr {
            dst: v(1),
            slot: address,
        });
        entry.push(Instruction::Store {
            ty,
            src: imm(lhs),
            addr: at_v1(0),
            attrs: MemoryAttrs::default(),
        });
        entry.push(Instruction::Load {
            ty,
            dst: v(2),
            addr: at_v1(0),
            attrs: MemoryAttrs::default(),
        });
        entry.push(Instruction::IntBinary {
            op,
            ty,
            dst: v(3),
            lhs: reg(2),
            rhs: imm(rhs),
        });
        entry.push(Instruction::IntConvert {
            from: ty,
            to: I64,
            signed: false,
            dst: v(4),
            src: reg(3),
        });
        ret(entry, 4);
    })
}

#[test]
fn narrow_unsigned_division_reads_the_type_bits() {
    assert_eq!(narrow_op(IntBinOp::UDiv, ScalarType::I16, 50000, 2), 25000);
    assert_eq!(
        narrow_op(IntBinOp::URem, ScalarType::I16, 50000, 7),
        50000 % 7
    );
    assert_eq!(narrow_op(IntBinOp::UDiv, ScalarType::I8, 200, 3), 66);
}

#[test]
fn narrow_shifts_read_the_type_bits_and_mask_the_amount() {
    assert_eq!(
        narrow_op(IntBinOp::LShr, ScalarType::I16, 0xF000, 4),
        0x0F00
    );
    assert_eq!(narrow_op(IntBinOp::LShr, ScalarType::I8, 0x80, 9), 0x40);
    assert_eq!(narrow_op(IntBinOp::Shl, ScalarType::I8, 1, 9), 2);
    assert_eq!(narrow_op(IntBinOp::AShr, ScalarType::I8, 0x80, 1), 0xC0);
}

#[test]
fn bool_arithmetic_is_byte_wide_like_the_aot_backend() {
    assert_eq!(narrow_op(IntBinOp::Add, ScalarType::I1, 1, 1), 2);
}

#[test]
fn extern_float_goes_through_float_registers() {
    let mut ldexp = Function::new(
        Signature::new("ldexp")
            .with_params(vec![
                Parameter::new(v(0), F64_TY),
                Parameter::new(v(1), MirType::Scalar(ScalarType::I32)),
            ])
            .with_return(F64_TY),
    );
    ldexp.blocks.clear();
    let mut entry = Block::new("entry");
    entry.push(Instruction::Call {
        name: "ldexp".to_string(),
        args: vec![f64_imm(1.5), imm(3)],
        ret: Some(v(0)),
    });
    ret(&mut entry, 0);
    let mut main = Function::new(Signature::new("main").with_return(I64));
    main.add_block(entry);
    let bits = run(&module_of(vec![ldexp, main]), "main");
    assert_eq!(f64::from_bits(bits as u64), 12.0);
}

#[cfg(target_os = "macos")]
#[test]
fn variadic_arguments_go_on_the_stack_on_darwin() {
    let mut snprintf = Function::new(
        Signature::new("snprintf")
            .with_params(vec![
                Parameter::new(v(0), MirType::Scalar(ScalarType::Ptr)),
                Parameter::new(v(1), I64),
                Parameter::new(v(2), MirType::Scalar(ScalarType::Ptr)),
            ])
            .with_return(MirType::Scalar(ScalarType::I32)),
    );
    snprintf.sig.variadic = true;
    snprintf.blocks.clear();

    let mut entry = Block::new("entry");
    entry.push(Instruction::StackAddr {
        dst: v(1),
        slot: VirtualReg::gpr(50),
    });
    entry.push(Instruction::SymbolAddress {
        dst: v(2),
        symbol: "format".to_string(),
    });
    entry.push(Instruction::Call {
        name: "snprintf".to_string(),
        args: vec![reg(1), imm(32), reg(2), imm(123), imm(-45)],
        ret: Some(v(3)),
    });
    ret(&mut entry, 3);
    let mut main = Function::new(Signature::new("main").with_return(I64));
    main.stack_reservations.insert(VirtualReg::gpr(50), 4);
    main.add_block(entry);
    let mut module = module_of(vec![snprintf, main]);
    let format = Global::new("format", I64, PointerWidth::Bits64)
        .with_layout(10, 1)
        .with_initializer(b"%lld|%lld\0".to_vec());
    module.globals.insert("format".to_string(), format);
    // "123|-45"
    assert_eq!(run(&module, "main"), 7);
}

fn two_functions_with_pointer_call() -> Module {
    let mut seven = Function::new(
        Signature::new("seven")
            .with_params(vec![Parameter::new(v(0), I64)])
            .with_return(I64),
    );
    let mut seven_entry = Block::new("entry");
    seven_entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(1),
        lhs: reg(0),
        rhs: imm(7),
    });
    ret(&mut seven_entry, 1);
    seven.add_block(seven_entry);

    let mut entry = Block::new("entry");
    entry.push(Instruction::SymbolAddress {
        dst: v(0),
        symbol: "seven".to_string(),
    });
    entry.push(Instruction::CallIndirect {
        target: reg(0),
        arg_types: vec![I64],
        args: vec![imm(35)],
        variadic_fixed: None,
        ret: Some(v(1)),
        ret_ty: Some(I64),
    });
    ret(&mut entry, 1);
    let mut main = Function::new(Signature::new("main").with_return(I64));
    main.add_block(entry);
    module_of(vec![seven, main])
}

#[test]
fn function_address_and_indirect_call() {
    assert_eq!(run(&two_functions_with_pointer_call(), "main"), 42);
}

/// Compiles every function in `functions` into one block and returns what each one of the
/// `names` returns, in order.
fn run_named(functions: Vec<Function>, names: &[String]) -> Vec<i64> {
    let module = module_of(functions);
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    let (code, offsets) = assembler
        .compile_mir_to_binary_function(&module, None)
        .expect("compile");
    let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
    memory.write_code(&code).expect("write");
    memory.make_executable().expect("protect");
    names
        .iter()
        .map(|name| {
            let start = unsafe { memory.code_start().add(offsets[name.as_str()]) };
            let function: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(start) };
            unsafe { function() }
        })
        .collect()
}

/// Stores `value` at `ty`, reads it back, applies `op` and widens the result to i64.
fn unary_function(name: &str, op: IntUnOp, ty: ScalarType, value: i64) -> Function {
    let ty = MirType::Scalar(ty);
    let mut entry = Block::new("entry");
    entry.push(Instruction::StackAddr {
        dst: v(1),
        slot: VirtualReg::gpr(50),
    });
    entry.push(Instruction::Store {
        ty,
        src: imm(value),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::Load {
        ty,
        dst: v(2),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::IntUnary {
        op,
        ty,
        dst: v(3),
        src: reg(2),
    });
    entry.push(Instruction::IntConvert {
        from: ty,
        to: I64,
        signed: true,
        dst: v(4),
        src: reg(3),
    });
    ret(&mut entry, 4);
    let mut function = Function::new(Signature::new(name).with_return(I64));
    function.add_block(entry);
    function
}

fn unary_result(op: IntUnOp, ty: ScalarType, value: i64) -> i64 {
    let name = "unary".to_string();
    run_named(vec![unary_function(&name, op, ty, value)], &[name])[0]
}

#[test]
fn int_unary_matches_mir_evaluation_at_every_width() {
    const EDGES: [i64; 22] = [
        0,
        1,
        -1,
        2,
        3,
        0x7F,
        -0x80,
        0xFF,
        0x100,
        0x0102,
        0x8000,
        -0x8000,
        0x7FFF,
        0x0102_0304,
        -0x8000_0000,
        0x7FFF_FFFF,
        0x8000_0001,
        0x0123_4567_89AB_CDEF,
        i64::MIN,
        i64::MAX,
        0x40,
        0xF0F0,
    ];
    let widths = [
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
    ];
    let ops = [
        IntUnOp::Popcount,
        IntUnOp::Ctlz,
        IntUnOp::Cttz,
        IntUnOp::Bswap,
    ];
    let mut functions = Vec::new();
    let mut expected = Vec::new();
    let mut names = Vec::new();
    for op in ops {
        for ty in widths {
            for value in EDGES {
                let name = format!("u{}", functions.len());
                functions.push(unary_function(&name, op, ty, value));
                expected.push((
                    op,
                    ty,
                    value,
                    IntUnOp::evaluate(op, MirType::Scalar(ty), value, PointerWidth::Bits64),
                ));
                names.push(name);
            }
        }
    }
    let got = run_named(functions, &names);
    for (index, (op, ty, value, want)) in expected.iter().enumerate() {
        assert_eq!(got[index], *want, "{op} {ty:?} of {value:#x}");
    }
}

#[test]
fn int_unary_hand_checked_edges() {
    assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I8, 0), 8);
    assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I8, 1), 7);
    assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I16, 0), 16);
    assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I32, -1), 0);
    assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I64, 0), 64);
    assert_eq!(unary_result(IntUnOp::Cttz, ScalarType::I8, 0), 8);
    assert_eq!(unary_result(IntUnOp::Cttz, ScalarType::I16, 0x8000), 15);
    assert_eq!(unary_result(IntUnOp::Cttz, ScalarType::I64, 0), 64);
    assert_eq!(unary_result(IntUnOp::Popcount, ScalarType::I8, -1), 8);
    assert_eq!(unary_result(IntUnOp::Popcount, ScalarType::I64, -1), 64);
    assert_eq!(
        unary_result(IntUnOp::Popcount, ScalarType::I64, i64::MIN),
        1
    );
    assert_eq!(
        unary_result(IntUnOp::Bswap, ScalarType::I16, 0x0102),
        0x0201
    );
    assert_eq!(
        unary_result(IntUnOp::Bswap, ScalarType::I32, 0x0102_0304),
        0x0403_0201
    );
    assert_eq!(unary_result(IntUnOp::Bswap, ScalarType::I8, -128), -128);
    assert_eq!(
        unary_result(IntUnOp::Bswap, ScalarType::I64, 0x0102_0304_0506_0708),
        0x0807_0605_0403_0201
    );
}

fn typed_imm(ty: ScalarType, value: i64) -> Operand {
    Operand::Immediate(match ty {
        ScalarType::I8 => Immediate::I8(value as i8),
        ScalarType::I16 => Immediate::I16(value as i16),
        ScalarType::I32 => Immediate::I32(value as i32),
        _ => Immediate::I64(value),
    })
}

/// Reads `lhs` and `rhs` at `ty` (from memory, or `rhs` as an immediate) and returns the
/// IntOverflow flag widened to i64.
fn overflow_function(
    name: &str,
    op: IntOverflowOp,
    ty: ScalarType,
    signed: bool,
    lhs: i64,
    rhs: i64,
    rhs_immediate: bool,
) -> Function {
    let mir_ty = MirType::Scalar(ty);
    let slot = VirtualReg::gpr(50);
    let mut entry = Block::new("entry");
    entry.push(Instruction::StackAddr { dst: v(1), slot });
    entry.push(Instruction::Store {
        ty: mir_ty,
        src: typed_imm(ty, lhs),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    entry.push(Instruction::Load {
        ty: mir_ty,
        dst: v(2),
        addr: at_v1(0),
        attrs: MemoryAttrs::default(),
    });
    let rhs_operand = if rhs_immediate {
        typed_imm(ty, rhs)
    } else {
        entry.push(Instruction::Store {
            ty: mir_ty,
            src: typed_imm(ty, rhs),
            addr: at_v1(8),
            attrs: MemoryAttrs::default(),
        });
        entry.push(Instruction::Load {
            ty: mir_ty,
            dst: v(5),
            addr: at_v1(8),
            attrs: MemoryAttrs::default(),
        });
        reg(5)
    };
    entry.push(Instruction::IntOverflow {
        op,
        ty: mir_ty,
        signed,
        dst: v(3),
        lhs: reg(2),
        rhs: rhs_operand,
    });
    entry.push(Instruction::IntConvert {
        from: MirType::Scalar(ScalarType::I1),
        to: I64,
        signed: false,
        dst: v(4),
        src: reg(3),
    });
    ret(&mut entry, 4);
    let mut function = Function::new(Signature::new(name).with_return(I64));
    function.stack_reservations.insert(slot, 2);
    function.add_block(entry);
    function
}

fn overflow_result(op: IntOverflowOp, ty: ScalarType, signed: bool, lhs: i64, rhs: i64) -> i64 {
    let name = "overflow".to_string();
    let function = overflow_function(&name, op, ty, signed, lhs, rhs, true);
    run_named(vec![function], &[name])[0]
}

#[test]
fn int_overflow_matches_mir_at_every_width_and_signedness() {
    let edges: [(ScalarType, [i64; 10]); 4] = [
        (ScalarType::I8, [-128, -127, -2, -1, 0, 1, 2, 63, 64, 127]),
        (
            ScalarType::I16,
            [
                i64::from(i16::MIN),
                -256,
                -2,
                -1,
                0,
                1,
                255,
                256,
                0x4000,
                i64::from(i16::MAX),
            ],
        ),
        (
            ScalarType::I32,
            [
                i64::from(i32::MIN),
                -65536,
                -2,
                -1,
                0,
                1,
                65536,
                0x7FFF,
                0x4000_0000,
                i64::from(i32::MAX),
            ],
        ),
        (
            ScalarType::I64,
            [
                i64::MIN,
                i64::MIN + 1,
                -2,
                -1,
                0,
                1,
                2,
                0x1_0000_0000,
                0x4000_0000_0000_0000,
                i64::MAX,
            ],
        ),
    ];
    let ops = [IntOverflowOp::Add, IntOverflowOp::Sub, IntOverflowOp::Mul];
    let mut functions = Vec::new();
    let mut names = Vec::new();
    let mut expected = Vec::new();
    for (ty, values) in edges {
        for op in ops {
            for signed in [true, false] {
                for (i, &lhs) in values.iter().enumerate() {
                    for (j, &rhs) in values.iter().enumerate() {
                        let name = format!("o{}", functions.len());
                        functions.push(overflow_function(
                            &name,
                            op,
                            ty,
                            signed,
                            lhs,
                            rhs,
                            (i + j) % 2 == 0,
                        ));
                        let want = op.overflows(
                            MirType::Scalar(ty),
                            signed,
                            lhs,
                            rhs,
                            PointerWidth::Bits64,
                        );
                        expected.push((ty, op, signed, lhs, rhs, i64::from(want)));
                        names.push(name);
                    }
                }
            }
        }
    }
    let got = run_named(functions, &names);
    for (index, (ty, op, signed, lhs, rhs, want)) in expected.iter().enumerate() {
        assert_eq!(got[index], *want, "{op} {ty:?} signed={signed} {lhs} {rhs}");
    }
}

#[test]
fn int_overflow_hand_checked_boundaries() {
    use IntOverflowOp::{Add, Mul, Sub};
    use ScalarType::{I8, I16, I32, I64};
    // (op, width, signed, lhs, rhs, overflows)
    let cases = [
        (Add, I8, true, 127, 1, 1),
        (Add, I8, false, 127, 1, 0),
        (Add, I8, false, 255, 1, 1),
        (Add, I8, false, 254, 1, 0),
        (Sub, I8, true, -128, 1, 1),
        (Sub, I8, false, 0, 1, 1),
        (Mul, I8, false, -1, 2, 1),
        (Mul, I8, true, -128, -1, 1),
        (Mul, I16, true, -1, i64::from(i16::MIN), 1),
        (Add, I32, false, 0xFFFF_FFFF, 1, 1),
        (Mul, I32, true, 65536, 65536, 1),
        (Mul, I32, false, 65536, 65535, 0),
        (Sub, I64, true, 0, i64::MIN, 1),
        (Sub, I64, false, 0, i64::MIN, 1),
        (Sub, I64, false, 0, -1, 1),
        (Sub, I64, true, i64::MIN, 0, 0),
        (Add, I64, true, i64::MAX, 1, 1),
        (Add, I64, false, -1, 1, 1),
        (Add, I64, false, 1, 1, 0),
        (Mul, I64, true, i64::MIN, 1, 0),
        (Mul, I64, true, -1, i64::MIN, 1),
        (Mul, I64, true, 0x1_0000_0000, 0x7FFF_FFFF, 0),
        (Mul, I64, false, i64::MAX, 2, 0),
        (Mul, I64, false, i64::MAX, 3, 1),
        (Mul, I64, false, -1, 1, 0),
        (Mul, I64, false, 0x1_0000_0000, 0x1_0000_0000, 1),
    ];
    for (op, ty, signed, lhs, rhs, want) in cases {
        assert_eq!(
            overflow_result(op, ty, signed, lhs, rhs),
            want,
            "{op} {ty:?} signed={signed} {lhs} {rhs}"
        );
    }
}

const ALL_WIDTHS: [ScalarType; 4] = [
    ScalarType::I8,
    ScalarType::I16,
    ScalarType::I32,
    ScalarType::I64,
];

/// The value a `ty` holds after storing `raw`, read back sign-extended.
fn canonical(ty: ScalarType, raw: i64) -> i64 {
    MirType::Scalar(ty).signed_integer(raw, PointerWidth::Bits64)
}

fn mask_of(ty: ScalarType) -> u64 {
    match ty {
        ScalarType::I8 => 0xFF,
        ScalarType::I16 => 0xFFFF,
        ScalarType::I32 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

fn slot_addr(dst: u32, slot: u32) -> Instruction {
    Instruction::StackAddr {
        dst: v(dst),
        slot: VirtualReg::gpr(slot),
    }
}

fn at_base(base: u32) -> AddressMode {
    AddressMode::BaseOffset {
        address_space: Default::default(),
        base: v(base),
        offset: 0,
    }
}

fn widen(entry: &mut Block, ty: ScalarType, src: u32, dst: u32) {
    entry.push(Instruction::IntConvert {
        from: MirType::Scalar(ty),
        to: I64,
        signed: true,
        dst: v(dst),
        src: reg(src),
    });
}

/// A function that makes its result from `body`, which runs after the slot is set up.
fn atomic_function(name: &str, body: impl FnOnce(&mut Block)) -> Function {
    let mut entry = Block::new("entry");
    entry.push(slot_addr(1, 50));
    body(&mut entry);
    let mut function = Function::new(Signature::new(name).with_return(I64));
    function.stack_reservations.insert(VirtualReg::gpr(50), 1);
    function.add_block(entry);
    function
}

#[test]
fn atomic_load_and_store_round_trip_every_width_and_ordering() {
    use MemoryOrdering::{Acquire, Relaxed, Release, SeqCst};
    let values: [i64; 6] = [0, 1, -1, 0x7F, -0x8000, 0x1234_5678_9ABC_DEF0];
    let orderings = [(Relaxed, Relaxed), (Release, Acquire), (SeqCst, SeqCst)];
    let mut functions = Vec::new();
    let mut names = Vec::new();
    let mut expected = Vec::new();
    for ty in ALL_WIDTHS {
        for (store_order, load_order) in orderings {
            for (index, &value) in values.iter().enumerate() {
                let name = format!("rt{}", functions.len());
                let from_register = index % 2 == 1;
                functions.push(atomic_function(&name, |entry| {
                    let stored = if from_register {
                        entry.push(Instruction::Copy {
                            ty: I64,
                            dst: v(5),
                            src: imm(value),
                        });
                        reg(5)
                    } else {
                        typed_imm(ty, value)
                    };
                    entry.push(Instruction::AtomicStore {
                        ty: MirType::Scalar(ty),
                        src: stored,
                        addr: at_base(1),
                        ordering: store_order,
                    });
                    entry.push(Instruction::AtomicLoad {
                        ty: MirType::Scalar(ty),
                        dst: v(2),
                        addr: at_base(1),
                        ordering: load_order,
                    });
                    widen(entry, ty, 2, 3);
                    ret(entry, 3);
                }));
                names.push(name);
                expected.push((ty, store_order, load_order, value, canonical(ty, value)));
            }
        }
    }
    let got = run_named(functions, &names);
    for (index, (ty, store_order, load_order, value, want)) in expected.iter().enumerate() {
        assert_eq!(
            got[index], *want,
            "{ty:?} store {store_order:?} load {load_order:?} of {value:#x}"
        );
    }
}

fn atomic_binop(op: AtomicBinOp, a: i64, b: i64) -> i64 {
    match op {
        AtomicBinOp::Add => a.wrapping_add(b),
        AtomicBinOp::Sub => a.wrapping_sub(b),
        AtomicBinOp::And => a & b,
        AtomicBinOp::Or => a | b,
        AtomicBinOp::Xor => a ^ b,
        AtomicBinOp::Exchange => b,
    }
}

#[test]
fn atomic_binary_returns_old_and_stores_new_at_every_width() {
    use AtomicBinOp::{Add, And, Exchange, Or, Sub, Xor};
    let ops = [Add, Sub, And, Or, Xor, Exchange];
    let values: [i64; 8] = [0, 1, -1, 0x7F, -0x80, 0x100, 0x7FFF_FFFF, i64::MIN];
    let mut functions = Vec::new();
    let mut names = Vec::new();
    let mut expected = Vec::new();
    for ty in ALL_WIDTHS {
        for op in ops {
            for (i, &a) in values.iter().enumerate() {
                for (j, &b) in values.iter().enumerate() {
                    // Old value and stored value are read by two functions over the same setup.
                    let build = |name: &str, read_old: bool| {
                        atomic_function(name, |entry| {
                            entry.push(Instruction::AtomicStore {
                                ty: MirType::Scalar(ty),
                                src: typed_imm(ty, a),
                                addr: at_base(1),
                                ordering: MemoryOrdering::Relaxed,
                            });
                            let value = if (i + j) % 2 == 0 {
                                typed_imm(ty, b)
                            } else {
                                entry.push(Instruction::Copy {
                                    ty: I64,
                                    dst: v(6),
                                    src: imm(b),
                                });
                                reg(6)
                            };
                            entry.push(Instruction::AtomicBinary {
                                op,
                                ty: MirType::Scalar(ty),
                                dst: v(2),
                                addr: at_base(1),
                                value,
                                ordering: MemoryOrdering::SeqCst,
                            });
                            if read_old {
                                widen(entry, ty, 2, 3);
                            } else {
                                entry.push(Instruction::AtomicLoad {
                                    ty: MirType::Scalar(ty),
                                    dst: v(4),
                                    addr: at_base(1),
                                    ordering: MemoryOrdering::Acquire,
                                });
                                widen(entry, ty, 4, 3);
                            }
                            ret(entry, 3);
                        })
                    };
                    let old_name = format!("bin{}", functions.len());
                    functions.push(build(&old_name, true));
                    names.push(old_name);
                    let new_name = format!("bin{}", functions.len());
                    functions.push(build(&new_name, false));
                    names.push(new_name);
                    let result = atomic_binop(op, a, b);
                    expected.push((ty, op, a, b, canonical(ty, a), canonical(ty, result)));
                }
            }
        }
    }
    let got = run_named(functions, &names);
    for (index, (ty, op, a, b, old, new)) in expected.iter().enumerate() {
        assert_eq!(got[2 * index], *old, "{op} old {ty:?} {a:#x} {b:#x}");
        assert_eq!(got[2 * index + 1], *new, "{op} new {ty:?} {a:#x} {b:#x}");
    }
}

#[test]
fn compare_exchange_swaps_only_on_a_width_match() {
    let values: [i64; 5] = [0, 1, -1, 0x7F, -0x80];
    let mut functions = Vec::new();
    let mut names = Vec::new();
    let mut expected = Vec::new();
    for ty in ALL_WIDTHS {
        for &current in &values {
            for &wanted in &values {
                // Narrow expected values also come with bits above the width set.
                let expected_raw = if ty == ScalarType::I64 {
                    wanted
                } else {
                    wanted ^ 0x1234_5600_0000_0000
                };
                let desired = 42;
                let build = |name: &str, read_old: bool| {
                    atomic_function(name, |entry| {
                        entry.push(Instruction::AtomicStore {
                            ty: MirType::Scalar(ty),
                            src: typed_imm(ty, current),
                            addr: at_base(1),
                            ordering: MemoryOrdering::Relaxed,
                        });
                        entry.push(Instruction::Copy {
                            ty: I64,
                            dst: v(5),
                            src: imm(expected_raw),
                        });
                        entry.push(Instruction::AtomicCompareExchange {
                            ty: MirType::Scalar(ty),
                            dst: v(2),
                            addr: at_base(1),
                            expected: reg(5),
                            desired: typed_imm(ty, desired),
                            success_ordering: MemoryOrdering::SeqCst,
                            failure_ordering: MemoryOrdering::Acquire,
                        });
                        if read_old {
                            widen(entry, ty, 2, 3);
                        } else {
                            entry.push(Instruction::AtomicLoad {
                                ty: MirType::Scalar(ty),
                                dst: v(4),
                                addr: at_base(1),
                                ordering: MemoryOrdering::SeqCst,
                            });
                            widen(entry, ty, 4, 3);
                        }
                        ret(entry, 3);
                    })
                };
                let old_name = format!("cas{}", functions.len());
                functions.push(build(&old_name, true));
                names.push(old_name);
                let new_name = format!("cas{}", functions.len());
                functions.push(build(&new_name, false));
                names.push(new_name);
                let matched =
                    (canonical(ty, current) as u64 & mask_of(ty)) == (wanted as u64 & mask_of(ty));
                let after = if matched {
                    canonical(ty, desired)
                } else {
                    canonical(ty, current)
                };
                expected.push((ty, current, wanted, canonical(ty, current), after));
            }
        }
    }
    let got = run_named(functions, &names);
    for (index, (ty, current, wanted, old, after)) in expected.iter().enumerate() {
        assert_eq!(got[2 * index], *old, "cas old {ty:?} {current} {wanted}");
        assert_eq!(
            got[2 * index + 1],
            *after,
            "cas new {ty:?} {current} {wanted}"
        );
    }
}

/// A function taking an address and adding one to the cell there, atomically.
fn bump_function(name: &str, ty: ScalarType) -> Function {
    let mut entry = Block::new("entry");
    entry.push(Instruction::AtomicBinary {
        op: AtomicBinOp::Add,
        ty: MirType::Scalar(ty),
        dst: v(1),
        addr: at_base(0),
        value: typed_imm(ty, 1),
        ordering: MemoryOrdering::SeqCst,
    });
    ret(&mut entry, 1);
    let mut function = Function::new(
        Signature::new(name)
            .with_params(vec![Parameter::new(v(0), I64)])
            .with_return(I64),
    );
    function.add_block(entry);
    function
}

#[test]
fn atomic_add_from_several_threads_loses_no_increment() {
    let module = module_of(vec![bump_function("bump64", ScalarType::I64)]);
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    let (code, offsets) = assembler
        .compile_mir_to_binary_function(&module, None)
        .expect("compile");
    let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
    memory.write_code(&code).expect("write");
    memory.make_executable().expect("protect");
    let base = memory.code_start() as usize;
    let wide = base + offsets["bump64"];
    let wide_fn: unsafe extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(wide) };
    let cell = Box::into_raw(Box::new(0i64)) as usize;
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(move || {
                for _ in 0..2000 {
                    unsafe { wide_fn(cell as i64) };
                }
            });
        }
    });
    let total = unsafe { *Box::from_raw(cell as *mut i64) };
    assert_eq!(total, 8000);
}

#[test]
fn narrow_atomic_add_leaves_neighbouring_bytes_alone() {
    let module = module_of(vec![bump_function("bump8", ScalarType::I8)]);
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    let (code, offsets) = assembler
        .compile_mir_to_binary_function(&module, None)
        .expect("compile");
    let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
    memory.write_code(&code).expect("write");
    memory.make_executable().expect("protect");
    let start = memory.code_start() as usize + offsets["bump8"];
    let bump: unsafe extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(start) };
    let cell = Box::into_raw(Box::new(0xAAAA_AAAA_AAAA_AA00u64)) as usize;
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(move || {
                for _ in 0..1000 {
                    unsafe { bump(cell as i64) };
                }
            });
        }
    });
    let after = unsafe { *Box::from_raw(cell as *mut u64) };
    // 4000 increments of one byte wrap to 160 and touch nothing else.
    assert_eq!(after, 0xAAAA_AAAA_AAAA_AAA0);
}

#[test]
fn atomic_orderings_the_mir_rejects_are_refused() {
    let refuse = |instruction: Instruction| {
        let mut entry = Block::new("entry");
        entry.push(slot_addr(1, 50));
        entry.push(instruction);
        entry.push(Instruction::Ret { value: None });
        let mut function = Function::new(Signature::new("bad").with_return(I64));
        function.stack_reservations.insert(VirtualReg::gpr(50), 1);
        function.add_block(entry);
        let module = module_of(vec![function]);
        let mut assembler =
            RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
        assembler
            .compile_mir_to_binary_function(&module, None)
            .is_err()
    };
    let i64ty = MirType::Scalar(ScalarType::I64);
    assert!(refuse(Instruction::AtomicLoad {
        ty: i64ty,
        dst: v(2),
        addr: at_v1(0),
        ordering: MemoryOrdering::Release,
    }));
    assert!(refuse(Instruction::AtomicStore {
        ty: i64ty,
        src: imm(1),
        addr: at_v1(0),
        ordering: MemoryOrdering::Acquire,
    }));
    assert!(refuse(Instruction::AtomicCompareExchange {
        ty: i64ty,
        dst: v(2),
        addr: at_v1(0),
        expected: imm(0),
        desired: imm(1),
        success_ordering: MemoryOrdering::Relaxed,
        failure_ordering: MemoryOrdering::Release,
    }));
    assert!(refuse(Instruction::AtomicBinary {
        op: AtomicBinOp::Add,
        ty: MirType::Scalar(ScalarType::F64),
        dst: v(2),
        addr: at_v1(0),
        value: Operand::Immediate(Immediate::F64(1.0)),
        ordering: MemoryOrdering::SeqCst,
    }));
}

/// Runs `entry`, which takes one i64 argument, and returns its result.
fn run_with_address(module: &Module, entry: &str, argument: i64) -> i64 {
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    let (code, offsets) = assembler
        .compile_mir_to_binary_function(module, None)
        .expect("compile");
    let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
    memory.write_code(&code).expect("write");
    memory.make_executable().expect("protect");
    let start = unsafe { memory.code_start().add(offsets[entry]) };
    let function: unsafe extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(start) };
    unsafe { function(argument) }
}

/// Calls the address in its argument through `args` and `arg_types`, returning the i64 result.
fn indirect_caller(arg_types: Vec<MirType>, args: Vec<Operand>) -> Module {
    let mut entry = Block::new("entry");
    entry.push(Instruction::CallIndirect {
        target: reg(0),
        arg_types,
        args,
        variadic_fixed: None,
        ret: Some(v(1)),
        ret_ty: Some(I64),
    });
    ret(&mut entry, 1);
    let mut main = Function::new(
        Signature::new("main")
            .with_params(vec![Parameter::new(v(0), I64)])
            .with_return(I64),
    );
    main.add_block(entry);
    module_of(vec![main])
}

extern "C" fn c_mixed(a: i64, c: i64, b: f64) -> i64 {
    a * 100 + c * 10 + (b * 10.0) as i64
}

extern "C" fn c_float_pair(x: f32, y: f64) -> i64 {
    (x * 2.0) as i64 * 1000 + y as i64
}

#[test]
fn indirect_call_passes_floats_to_a_c_function() {
    let mixed = indirect_caller(vec![I64, I64, F64_TY], vec![imm(7), imm(5), f64_imm(2.5)]);
    assert_eq!(
        run_with_address(
            &mixed,
            "main",
            c_mixed as extern "C" fn(i64, i64, f64) -> i64 as usize as i64
        ),
        775
    );
    let pair = indirect_caller(
        vec![MirType::Scalar(ScalarType::F32), F64_TY],
        vec![Operand::Immediate(Immediate::F32(1.5)), f64_imm(7.0)],
    );
    assert_eq!(
        run_with_address(
            &pair,
            "main",
            c_float_pair as extern "C" fn(f32, f64) -> i64 as usize as i64
        ),
        3007
    );
}

#[test]
fn indirect_call_passes_floats_to_a_jit_function() {
    let mut callee_entry = Block::new("entry");
    callee_entry.push(Instruction::Bitcast {
        from: F64_TY,
        to: I64,
        dst: v(2),
        src: reg(1),
    });
    callee_entry.push(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: I64,
        dst: v(3),
        lhs: reg(0),
        rhs: reg(2),
    });
    ret(&mut callee_entry, 3);
    let mut callee = Function::new(
        Signature::new("takes_float")
            .with_params(vec![
                Parameter::new(v(0), I64),
                Parameter::new(v(1), F64_TY),
            ])
            .with_return(I64),
    );
    callee.add_block(callee_entry);

    let mut entry = Block::new("entry");
    entry.push(Instruction::SymbolAddress {
        dst: v(0),
        symbol: "takes_float".to_string(),
    });
    entry.push(Instruction::CallIndirect {
        target: reg(0),
        arg_types: vec![I64, F64_TY],
        args: vec![imm(40), f64_imm(2.5)],
        variadic_fixed: None,
        ret: Some(v(1)),
        ret_ty: Some(I64),
    });
    ret(&mut entry, 1);
    let mut main = Function::new(Signature::new("main").with_return(I64));
    main.add_block(entry);
    let module = module_of(vec![callee, main]);
    assert_eq!(
        run(&module, "main"),
        40i64.wrapping_add(2.5f64.to_bits() as i64)
    );
}

#[test]
fn indirect_call_refuses_a_float_before_an_integer() {
    let module = indirect_caller(vec![F64_TY, I64], vec![f64_imm(1.0), imm(2)]);
    let mut assembler = RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
    assert!(
        assembler
            .compile_mir_to_binary_function(&module, None)
            .is_err()
    );
}

/// DMB ISH and DMB ISHLD, as the words the fence encoder writes.
const DMB_ISH: u32 = 0xD503_3BBF;
const DMB_ISHLD: u32 = 0xD503_39BF;

fn count_word(code: &[u8], word: u32) -> usize {
    code.chunks_exact(4)
        .filter(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) == word)
        .count()
}

#[test]
fn fence_encodes_the_barrier_each_ordering_needs() {
    use MemoryOrdering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
    let cases = [
        (Relaxed, 0, 0),
        (Acquire, 0, 1),
        (Release, 1, 0),
        (AcqRel, 1, 0),
        (SeqCst, 1, 0),
    ];
    for (ordering, ish, ishld) in cases {
        let name = format!("fence_{ordering:?}");
        let function = atomic_function(&name, |entry| {
            entry.push(Instruction::Store {
                ty: I64,
                src: imm(42),
                addr: at_base(1),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::Fence { ordering });
            entry.push(Instruction::Load {
                ty: I64,
                dst: v(2),
                addr: at_base(1),
                attrs: MemoryAttrs::default(),
            });
            ret(entry, 2);
        });
        let module = module_of(vec![function]);
        let mut assembler =
            RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
        let (code, _) = assembler
            .compile_mir_to_binary_function(&module, None)
            .expect("compile");
        assert_eq!(count_word(&code, DMB_ISH), ish, "{ordering:?} DMB ISH");
        assert_eq!(
            count_word(&code, DMB_ISHLD),
            ishld,
            "{ordering:?} DMB ISHLD"
        );
        assert_eq!(
            run_named(vec![module.functions[&name].clone()], &[name])[0],
            42
        );
    }
}
