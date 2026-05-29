use crate::error::{KdTreeError, Result};
use crate::real::Real;
use crate::tree::Interval;
use std::cell::RefCell;
use std::rc::Rc;

/// Data source interface expected by the KD-tree.
///
/// This mirrors nanoflann's `kdtree_get_point_count()`, `kdtree_get_pt()`, and
/// optional `kdtree_get_bbox()` adaptor methods while using Rust traits instead
/// of C++ templates.
pub trait KdTreeDataset<F: Real> {
    /// Number of points currently exposed by the dataset.
    fn kdtree_get_point_count(&self) -> usize;

    /// Component `dim` of point `idx`.
    ///
    /// Implementations may panic on invalid indices; the KD-tree validates its
    /// own generated indices, and query dimensionality is checked before search.
    fn kdtree_get_pt(&self, idx: usize, dim: usize) -> F;

    /// Optional precomputed bounding box. Return `true` if `bbox` was filled.
    fn kdtree_get_bbox(&self, _bbox: &mut [Interval<F>]) -> bool {
        false
    }
}

/// Simple in-memory point-cloud dataset.
#[derive(Clone, Debug)]
pub struct PointCloud<F: Real> {
    points: Vec<Vec<F>>,
    dim: usize,
}

impl<F: Real> PointCloud<F> {
    /// Creates a point cloud after checking that all rows share one dimension.
    pub fn new(points: Vec<Vec<F>>) -> Result<Self> {
        let dim = points.first().map_or(0, Vec::len);
        for (row, point) in points.iter().enumerate() {
            if point.len() != dim {
                return Err(KdTreeError::InconsistentPointDimensionality {
                    expected: dim,
                    got: point.len(),
                    row,
                });
            }
        }
        Ok(Self { points, dim })
    }

    /// Creates an empty point cloud with a known point dimension.
    pub fn empty(dim: usize) -> Self {
        Self {
            points: Vec::new(),
            dim,
        }
    }

    /// Appends one point and returns its index.
    pub fn push(&mut self, point: Vec<F>) -> Result<usize> {
        if point.len() != self.dim {
            return Err(KdTreeError::InconsistentPointDimensionality {
                expected: self.dim,
                got: point.len(),
                row: self.points.len(),
            });
        }
        let idx = self.points.len();
        self.points.push(point);
        Ok(idx)
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn points(&self) -> &[Vec<F>] {
        &self.points
    }
}

impl<F: Real> KdTreeDataset<F> for PointCloud<F> {
    #[inline]
    fn kdtree_get_point_count(&self) -> usize {
        self.points.len()
    }

    #[inline]
    fn kdtree_get_pt(&self, idx: usize, dim: usize) -> F {
        self.points[idx][dim]
    }
}

impl<F: Real> KdTreeDataset<F> for Rc<RefCell<PointCloud<F>>> {
    #[inline]
    fn kdtree_get_point_count(&self) -> usize {
        self.borrow().kdtree_get_point_count()
    }

    #[inline]
    fn kdtree_get_pt(&self, idx: usize, dim: usize) -> F {
        self.borrow().kdtree_get_pt(idx, dim)
    }

    #[inline]
    fn kdtree_get_bbox(&self, bbox: &mut [Interval<F>]) -> bool {
        self.borrow().kdtree_get_bbox(bbox)
    }
}

/// Interpretation of a flat row-major matrix buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLayout {
    /// Each matrix row is one point. Point dimension is `cols`.
    RowMajorPoints,
    /// Each matrix column is one point. Point dimension is `rows`.
    ColumnMajorPoints,
}

/// Flat matrix dataset adaptor similar in spirit to `KDTreeEigenMatrixAdaptor`.
///
/// The backing matrix is stored as a row-major flat buffer regardless of whether
/// rows or columns are interpreted as points.
#[derive(Clone, Debug)]
pub struct MatrixDataset<F: Real> {
    data: Vec<F>,
    rows: usize,
    cols: usize,
    layout: MatrixLayout,
}

impl<F: Real> MatrixDataset<F> {
    pub fn new(data: Vec<F>, rows: usize, cols: usize, layout: MatrixLayout) -> Result<Self> {
        if data.len() != rows.saturating_mul(cols) {
            return Err(KdTreeError::MatrixSizeMismatch {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Self {
            data,
            rows,
            cols,
            layout,
        })
    }

    pub fn point_dim(&self) -> usize {
        match self.layout {
            MatrixLayout::RowMajorPoints => self.cols,
            MatrixLayout::ColumnMajorPoints => self.rows,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }
}

impl<F: Real> KdTreeDataset<F> for MatrixDataset<F> {
    #[inline]
    fn kdtree_get_point_count(&self) -> usize {
        match self.layout {
            MatrixLayout::RowMajorPoints => self.rows,
            MatrixLayout::ColumnMajorPoints => self.cols,
        }
    }

    #[inline]
    fn kdtree_get_pt(&self, idx: usize, dim: usize) -> F {
        match self.layout {
            MatrixLayout::RowMajorPoints => self.data[idx * self.cols + dim],
            MatrixLayout::ColumnMajorPoints => self.data[dim * self.cols + idx],
        }
    }
}
