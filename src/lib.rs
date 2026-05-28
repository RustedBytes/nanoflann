//! Safe Rust KD-tree port inspired by the nanoflann header-only C++ API.
//!
//! The crate keeps nanoflann's core semantics: adaptor-style datasets,
//! squared L2 distances, strict radius comparisons, inclusive box search, and a
//! dynamic logarithmic multi-tree wrapper with lazy deletion.

mod dataset;
mod dynamic;
mod error;
mod metric;
mod real;
mod result_set;
mod tree;

pub use dataset::{KdTreeDataset, MatrixDataset, MatrixLayout, PointCloud};
pub use dynamic::DynamicKdTree;
pub use error::{KdTreeError, Result};
pub use metric::{DistanceMetric, L2Simple, L1, L2, SO2, SO3};
pub use real::Real;
pub use result_set::{KnnResultSet, RadiusResultSet, ResultItem, RknnResultSet};
pub use tree::{Interval, KdTree, KdTreeParams, SearchParameters};
