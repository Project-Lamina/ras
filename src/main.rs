//! ras - drop-in replacement for as/gas

use lamina_platform::Target;
use std::{
    env,
    ffi::OsStr,
    io::{self, Read},
    path::{Path, PathBuf},
    str::FromStr,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

fn init_logging() {
    let level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(log::LevelFilter::Warn);
    log::set_max_level(level);
    let _ = log::set_logger(&StderrLogger);
}

fn print_usage() {
    eprintln!("ras - drop-in replacement for as/gas (assembly to object file)");
    eprintln!();
    eprintln!("Usage: ras [options] [input.s]");
    eprintln!("       ras [options] -o <output.o> [input.s]");
    eprintln!();
    eprintln!("  If no input or input is '-', reads from stdin.");
    eprintln!("  If -o is omitted, writes to a.out.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -o <file>, --output <file>   Output object file (default: a.out)");
    eprintln!("  --target <arch_os>          Target (e.g. x86_64_linux, aarch64_macos)");
    eprintln!("  -L, --keep-locals           Retain temporary local symbols");
    eprintln!("  -v, --verbose               Verbose output");
    eprintln!("  --version                   Print version");
    eprintln!("  -h, --help                  This help");
    eprintln!();
    eprintln!("GCC driver (ras as assembler, weld as linker):");
    eprintln!("  mkdir -p toolchain-prefix && ln -sf \"$(command -v ras)\" toolchain-prefix/as");
    eprintln!("  gcc -B\"$PWD/toolchain-prefix\" -fuse-ld=weld -o prog main.c ...");
    eprintln!("  (install `weld` on PATH; symlink argv0 `as` enables GNU-style flags from gcc.)");
}

fn program_name_is_gnu_as(arg0: &str) -> bool {
    Path::new(arg0)
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|base| base == "as" || base.ends_with("-as"))
}

/// When the binary is invoked as `as` (e.g. symlink `as` -> `ras`), accept argv from the GCC driver.
fn parse_gnu_as_invocation(program_args: &[String]) -> Result<Options, String> {
    let mut keep_locals = false;
    let mut output_file = None;
    let mut input_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < program_args.len() {
        let a = program_args[i].as_str();
        match a {
            "-o" => {
                let next = program_args
                    .get(i + 1)
                    .ok_or_else(|| "Missing argument for -o".to_string())?;
                output_file = Some(PathBuf::from(next));
                i += 2;
            }
            "-I" | "-J" | "-arch" => {
                let _ = program_args
                    .get(i + 1)
                    .ok_or_else(|| format!("Missing argument for {}", a))?;
                i += 2;
            }
            "-L" | "--keep-locals" => {
                keep_locals = true;
                i += 1;
            }
            "--64" | "--32" | "-n" | "-q" | "-s" | "-g" | "-Z" | "-K" | "-Q" | "-k" | "-v"
            | "-V" | "-w" | "-d" => {
                i += 1;
            }
            a if a.starts_with("-march=")
                || a.starts_with("-mcpu=")
                || a.starts_with("-mabi=")
                || a.starts_with("-mtune=")
                || a.starts_with("--gdwarf-")
                || a.starts_with("-debug-prefix-map=")
                || a.starts_with("-fdebug-prefix-map=")
                || a.starts_with("-mmacosx-version-min=")
                || a.starts_with("-mbss-plt")
                || a.starts_with("-mplt=") =>
            {
                i += 1;
            }
            a if a.starts_with("-m") && a.len() > 2 => {
                i += 1;
            }
            a if a.starts_with('-') => {
                i += 1;
            }
            "-" => {
                if input_path.is_some() {
                    return Err("gnu as compat: multiple input files".to_string());
                }
                input_path = Some(PathBuf::from("-"));
                i += 1;
            }
            _ => {
                if input_path.is_some() {
                    return Err("gnu as compat: multiple input files".to_string());
                }
                input_path = Some(PathBuf::from(a));
                i += 1;
            }
        }
    }

    let input_file =
        Some(input_path.ok_or_else(|| "gnu as compat: missing input .s file".to_string())?);
    Ok(Options {
        input_file,
        output_file,
        target: None,
        verbose: false,
        keep_locals,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    let options = match parse_args() {
        Ok(opts) => opts,
        Err(e) => {
            if e == "help" {
                print_usage();
                return Ok(());
            }
            if e == "version" {
                println!("ras {}", VERSION);
                return Ok(());
            }
            eprintln!("Error: {}", e);
            print_usage();
            std::process::exit(1);
        }
    };

    if options.verbose {
        log::set_max_level(log::max_level().max(log::LevelFilter::Info));
    }

    let output_path = options
        .output_file
        .unwrap_or_else(|| PathBuf::from("a.out"));

    let target = if let Some(target_str) = &options.target {
        Target::from_str(target_str)
            .map_err(|e| format!("Invalid target '{}': {}", target_str, e))?
    } else {
        Target::detect_host()
    };

    let input_desc = options
        .input_file
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<stdin>".to_string());
    log::info!("Assembling {} -> {}", input_desc, output_path.display());
    log::info!("Target: {}", target);

    let mut ras = ras::Ras::with_object_write_options(
        target.architecture,
        target.operating_system,
        ras::ObjectWriteOptions {
            keep_locals: options.keep_locals,
            ..Default::default()
        },
    )
    .map_err(|e| format!("Failed to create assembler: {}", e))?;

    let asm_text = read_assembly_input(&options.input_file)?;
    ras.assemble(&asm_text, &output_path)
        .map_err(|e| format!("Assembly failed: {}", e))?;

    log::info!("Assembly completed successfully");

    Ok(())
}

struct Options {
    input_file: Option<PathBuf>,
    output_file: Option<PathBuf>,
    target: Option<String>,
    verbose: bool,
    keep_locals: bool,
}

fn parse_args() -> Result<Options, String> {
    let args: Vec<String> = env::args().collect();
    let arg0 = args.first().map(String::as_str).unwrap_or("");
    if program_name_is_gnu_as(arg0) {
        return parse_gnu_as_invocation(&args[1..]);
    }

    let mut options = Options {
        input_file: None,
        output_file: None,
        target: None,
        verbose: false,
        keep_locals: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for output file".to_string());
                }
                options.output_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--target" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for target".to_string());
                }
                options.target = Some(args[i + 1].clone());
                i += 2;
            }
            "-v" | "--verbose" => {
                options.verbose = true;
                i += 1;
            }
            "-L" | "--keep-locals" => {
                options.keep_locals = true;
                i += 1;
            }
            "--64" | "--32" | "--x32" => i += 1,
            "--version" => return Err("version".to_string()),
            "-h" | "--help" => return Err("help".to_string()),
            _ => {
                if args[i].starts_with('-') {
                    return Err(format!("Unknown option: {}", args[i]));
                }
                if options.input_file.is_none() {
                    options.input_file = Some(PathBuf::from(&args[i]));
                } else {
                    return Err(format!("Unexpected argument: {}", args[i]));
                }
                i += 1;
            }
        }
    }

    Ok(options)
}

fn read_assembly_input(input_file: &Option<PathBuf>) -> Result<String, Box<dyn std::error::Error>> {
    match input_file {
        None => {
            let mut s = String::new();
            io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("Failed to read stdin: {}", e))?;
            Ok(s)
        }
        Some(path) => {
            if path.as_os_str() == OsStr::new("-") {
                let mut s = String::new();
                io::stdin()
                    .read_to_string(&mut s)
                    .map_err(|e| format!("Failed to read stdin: {}", e))?;
                Ok(s)
            } else if path.exists() {
                std::fs::read_to_string(path)
                    .map_err(|e| format!("Failed to read {}: {}", path.display(), e).into())
            } else {
                Err(format!("Input file '{}' does not exist.", path.display()).into())
            }
        }
    }
}

#[cfg(test)]
mod gnu_as_compat_tests {
    use super::*;

    #[test]
    fn gcc_passes_64_o_and_input() {
        let args = vec![
            "--64".to_string(),
            "-o".to_string(),
            "out.o".to_string(),
            "f.s".to_string(),
        ];
        let o = parse_gnu_as_invocation(&args).expect("parse");
        assert_eq!(o.output_file, Some(PathBuf::from("out.o")));
        assert_eq!(o.input_file, Some(PathBuf::from("f.s")));
    }

    #[test]
    fn ignores_include_and_march() {
        let args = vec![
            "-I".to_string(),
            "/tmp".to_string(),
            "-march=armv8-a".to_string(),
            "-o".to_string(),
            "x.o".to_string(),
            "a.s".to_string(),
        ];
        let o = parse_gnu_as_invocation(&args).expect("parse");
        assert_eq!(o.input_file, Some(PathBuf::from("a.s")));
    }
}
