//! ras - drop-in replacement for as/gas

use std::env;
use std::ffi::OsStr;
use std::io::{self, Read};
use std::path::PathBuf;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    eprintln!("  -v, --verbose               Verbose output");
    eprintln!("  --version                   Print version");
    eprintln!("  -h, --help                  This help");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let output_path = options.output_file.unwrap_or_else(|| PathBuf::from("a.out"));

    use lamina_platform::Target;
    use std::str::FromStr;
    let target = if let Some(target_str) = &options.target {
        Target::from_str(target_str)
            .map_err(|e| format!("Invalid target '{}': {}", target_str, e))?
    } else {
        Target::detect_host()
    };

    if options.verbose {
        let input_desc = options
            .input_file
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<stdin>".to_string());
        println!("[ras] Assembling {} -> {}", input_desc, output_path.display());
        println!("[ras] Target: {}", target);
    }

    let mut ras = ras::Ras::new(target.architecture, target.operating_system)
        .map_err(|e| format!("Failed to create assembler: {}", e))?;

    let asm_text = read_assembly_input(&options.input_file)?;
    ras.assemble(&asm_text, &output_path)
        .map_err(|e| format!("Assembly failed: {}", e))?;

    if options.verbose {
        println!("[ras] Assembly completed successfully");
    }

    Ok(())
}

struct Options {
    input_file: Option<PathBuf>,
    output_file: Option<PathBuf>,
    target: Option<String>,
    verbose: bool,
}

fn parse_args() -> Result<Options, String> {
    let args: Vec<String> = env::args().collect();
    let mut options = Options {
        input_file: None,
        output_file: None,
        target: None,
        verbose: false,
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
            io::stdin().read_to_string(&mut s)
                .map_err(|e| format!("Failed to read stdin: {}", e))?;
            Ok(s)
        }
        Some(path) => {
            if path.as_os_str() == OsStr::new("-") {
                let mut s = String::new();
                io::stdin().read_to_string(&mut s)
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
