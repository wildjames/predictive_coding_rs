use std::{error::Error as StdError, fmt, io, path::PathBuf, process::ExitStatus};

// Wrap result to that we always return the custom error type
pub type Result<T> = std::result::Result<T, PredictiveCodingError>;

#[derive(Debug)]
pub enum PredictiveCodingError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    JsonDeserialize {
        path: PathBuf,
        source: serde_json::Error,
    },
    JsonSerialize {
        path: PathBuf,
        source: serde_json::Error,
    },
    Csv {
        operation: &'static str,
        path: PathBuf,
        source: csv::Error,
    },
    Validation {
        message: String,
    },
    InvalidData {
        message: String,
    },
    CommandIo {
        command: String,
        source: io::Error,
    },
    CommandFailed {
        command: String,
        status: ExitStatus,
        stderr: String,
    },
}

impl PredictiveCodingError {
    pub fn io(operation: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        PredictiveCodingError::Io {
            operation,
            path: path.into(),
            source,
        }
    }

    pub fn json_deserialize(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        PredictiveCodingError::JsonDeserialize {
            path: path.into(),
            source,
        }
    }

    pub fn json_serialize(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        PredictiveCodingError::JsonSerialize {
            path: path.into(),
            source,
        }
    }

    pub fn csv(operation: &'static str, path: impl Into<PathBuf>, source: csv::Error) -> Self {
        PredictiveCodingError::Csv {
            operation,
            path: path.into(),
            source,
        }
    }

    pub fn validation(message: impl Into<String>) -> Self {
        PredictiveCodingError::Validation {
            message: message.into(),
        }
    }

    pub fn invalid_data(message: impl Into<String>) -> Self {
        PredictiveCodingError::InvalidData {
            message: message.into(),
        }
    }

    pub fn command_io(command: impl Into<String>, source: io::Error) -> Self {
        PredictiveCodingError::CommandIo {
            command: command.into(),
            source,
        }
    }

    pub fn command_failed(
        command: impl Into<String>,
        status: ExitStatus,
        stderr: impl Into<String>,
    ) -> Self {
        PredictiveCodingError::CommandFailed {
            command: command.into(),
            status,
            stderr: stderr.into(),
        }
    }
}

impl fmt::Display for PredictiveCodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictiveCodingError::Io {
                operation,
                path,
                source,
            } => write!(f, "failed to {operation} {}: {source}", path.display()),
            PredictiveCodingError::JsonDeserialize { path, source } => {
                write!(f, "failed to parse JSON from {}: {source}", path.display())
            }
            PredictiveCodingError::JsonSerialize { path, source } => {
                write!(f, "failed to write JSON to {}: {source}", path.display())
            }
            PredictiveCodingError::Csv {
                operation,
                path,
                source,
            } => write!(
                f,
                "failed to {operation} CSV at {}: {source}",
                path.display()
            ),
            PredictiveCodingError::Validation { message } => {
                write!(f, "validation error: {message}")
            }
            PredictiveCodingError::InvalidData { message } => write!(f, "invalid data: {message}"),
            PredictiveCodingError::CommandIo { command, source } => {
                write!(f, "failed to execute command '{command}': {source}")
            }
            PredictiveCodingError::CommandFailed {
                command,
                status,
                stderr,
            } => {
                if stderr.trim().is_empty() {
                    write!(f, "command '{command}' exited with status {status}")
                } else {
                    write!(
                        f,
                        "command '{command}' exited with status {status}: {}",
                        stderr.trim()
                    )
                }
            }
        }
    }
}

impl StdError for PredictiveCodingError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            PredictiveCodingError::Io { source, .. } => Some(source),
            PredictiveCodingError::JsonDeserialize { source, .. } => Some(source),
            PredictiveCodingError::JsonSerialize { source, .. } => Some(source),
            PredictiveCodingError::Csv { source, .. } => Some(source),
            PredictiveCodingError::CommandIo { source, .. } => Some(source),
            PredictiveCodingError::Validation { .. }
            | PredictiveCodingError::InvalidData { .. }
            | PredictiveCodingError::CommandFailed { .. } => None,
        }
    }
}
