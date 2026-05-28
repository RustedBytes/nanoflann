use std::fmt;

/// Crate-local result type.
pub type Result<T> = std::result::Result<T, KdTreeError>;

/// Errors returned by the safe Rust KD-tree API.
#[derive(Debug)]
pub enum KdTreeError {
    /// The tree dimensionality must be greater than zero.
    InvalidDimensionality,
    /// `leaf_max_size == 0` can lead to non-terminating splits in the original
    /// algorithm, so the Rust port rejects it during construction.
    InvalidLeafMaxSize,
    /// A query slice did not contain enough components for the tree dimension.
    QueryDimensionalityMismatch { expected_at_least: usize, got: usize },
    /// A bounding box did not match the tree dimension.
    BoundingBoxDimensionalityMismatch { expected: usize, got: usize },
    /// A dataset-provided bounding box had an invalid shape or bounds.
    InvalidBoundingBox(String),
    /// `find_*` was called before an index was built.
    IndexNotBuilt,
    /// The dataset does not contain points required by an index operation.
    IndexOutOfBounds { index: usize, len: usize },
    /// Dynamic insertion must be contiguous unless it reactivates a lazily
    /// removed point.
    NonContiguousInsertion { expected: usize, got: usize },
    /// The configured dynamic index capacity has been exceeded.
    MaximumPointCountExceeded { maximum_point_count: usize },
    /// Point-cloud construction received rows with inconsistent dimensionality.
    InconsistentPointDimensionality { expected: usize, got: usize, row: usize },
    /// A flat matrix buffer length was not `rows * cols`.
    MatrixSizeMismatch { rows: usize, cols: usize, len: usize },
    /// Binary index data was not produced by this crate's serializer.
    InvalidIndexFile(String),
    /// I/O error while saving or loading an index.
    Io(std::io::Error),
}

impl fmt::Display for KdTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensionality => write!(f, "KD-tree dimensionality must be greater than zero"),
            Self::InvalidLeafMaxSize => write!(f, "leaf_max_size must be greater than zero"),
            Self::QueryDimensionalityMismatch { expected_at_least, got } => write!(
                f,
                "query has {got} dimensions, expected at least {expected_at_least}"
            ),
            Self::BoundingBoxDimensionalityMismatch { expected, got } => {
                write!(f, "bounding box has {got} dimensions, expected {expected}")
            }
            Self::InvalidBoundingBox(msg) => write!(f, "invalid bounding box: {msg}"),
            Self::IndexNotBuilt => write!(f, "KD-tree index has not been built"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "point index {index} is outside dataset length {len}")
            }
            Self::NonContiguousInsertion { expected, got } => write!(
                f,
                "dynamic insertion must add the next point index {expected}, got {got}"
            ),
            Self::MaximumPointCountExceeded { maximum_point_count } => write!(
                f,
                "dynamic index capacity exceeded; maximum_point_count={maximum_point_count}"
            ),
            Self::InconsistentPointDimensionality { expected, got, row } => write!(
                f,
                "point row {row} has {got} dimensions, expected {expected}"
            ),
            Self::MatrixSizeMismatch { rows, cols, len } => write!(
                f,
                "matrix buffer has length {len}, expected rows * cols = {}",
                (*rows).saturating_mul(*cols)
            ),
            Self::InvalidIndexFile(msg) => write!(f, "invalid index file: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for KdTreeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for KdTreeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
