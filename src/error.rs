//! Error types for ras assembler

#[derive(Debug)]
pub enum RasError {
    IoError(String),
    ParseError(String),
    EncodingError(String),
    ObjectError(String),
    UnsupportedTarget(String),
    InvalidInput(String),
}

impl std::fmt::Display for RasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RasError::IoError(msg) => write!(f, "I/O error: {}", msg),
            RasError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            RasError::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
            RasError::ObjectError(msg) => write!(f, "Object file error: {}", msg),
            RasError::UnsupportedTarget(msg) => write!(f, "Unsupported target: {}", msg),
            RasError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for RasError {}
