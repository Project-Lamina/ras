//! ras - drop-in replacement for as/gas

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ras::main()
}
