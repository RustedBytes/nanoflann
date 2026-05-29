use crate::real::Real;
use std::cmp::Ordering;

/// One nearest-neighbor result item.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResultItem<F: Real> {
    /// Index of the sample in the dataset.
    pub index: usize,
    /// Distance from sample to query point. L2 distances are squared.
    pub distance: F,
}

impl<F: Real> ResultItem<F> {
    pub fn new(index: usize, distance: F) -> Self {
        Self { index, distance }
    }
}

pub trait ResultSet<F: Real> {
    fn add_point(&mut self, distance: F, index: usize) -> bool;
    fn worst_dist(&self) -> F;
    #[allow(dead_code)]
    fn size(&self) -> usize;
    fn full(&self) -> bool;
    fn sort(&mut self);
}

#[inline]
fn compare_distance<F: Real>(a: &ResultItem<F>, b: &ResultItem<F>) -> Ordering {
    a.distance
        .partial_cmp(&b.distance)
        .unwrap_or(Ordering::Equal)
}

#[inline]
fn insert_sorted_bounded<F: Real>(
    items: &mut Vec<ResultItem<F>>,
    item: ResultItem<F>,
    capacity: usize,
    first_match: bool,
) {
    if capacity == 0 {
        return;
    }

    let mut pos = items.len();
    while pos > 0 {
        let prev = items[pos - 1];
        let should_shift = prev.distance > item.distance
            || (first_match && prev.distance == item.distance && prev.index > item.index);
        if should_shift {
            pos -= 1;
        } else {
            break;
        }
    }

    if pos < capacity {
        items.insert(pos, item);
        if items.len() > capacity {
            items.pop();
        }
    }
}

/// Result set for k-nearest-neighbor queries.
#[derive(Clone, Debug)]
pub struct KnnResultSet<F: Real> {
    capacity: usize,
    items: Vec<ResultItem<F>>,
    first_match: bool,
}

impl<F: Real> KnnResultSet<F> {
    pub fn new(capacity: usize) -> Self {
        Self::with_first_match(capacity, false)
    }

    pub fn with_first_match(capacity: usize, first_match: bool) -> Self {
        Self {
            capacity,
            items: Vec::with_capacity(capacity),
            first_match,
        }
    }

    pub fn items(&self) -> &[ResultItem<F>] {
        &self.items
    }

    pub fn into_vec(self) -> Vec<ResultItem<F>> {
        self.items
    }
}

impl<F: Real> ResultSet<F> for KnnResultSet<F> {
    #[inline]
    fn add_point(&mut self, distance: F, index: usize) -> bool {
        insert_sorted_bounded(
            &mut self.items,
            ResultItem::new(index, distance),
            self.capacity,
            self.first_match,
        );
        true
    }

    #[inline]
    fn worst_dist(&self) -> F {
        if self.items.len() < self.capacity || self.items.is_empty() {
            F::max_value()
        } else {
            self.items[self.items.len() - 1].distance
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.items.len()
    }

    #[inline]
    fn full(&self) -> bool {
        self.items.len() == self.capacity
    }

    #[inline]
    fn sort(&mut self) {
        // KNN insertion keeps the vector sorted already.
    }
}

/// Result set for radius-limited k-nearest-neighbor queries.
#[derive(Clone, Debug)]
pub struct RknnResultSet<F: Real> {
    capacity: usize,
    radius: F,
    items: Vec<ResultItem<F>>,
    first_match: bool,
}

impl<F: Real> RknnResultSet<F> {
    pub fn new(capacity: usize, radius: F) -> Self {
        Self::with_first_match(capacity, radius, false)
    }

    pub fn with_first_match(capacity: usize, radius: F, first_match: bool) -> Self {
        Self {
            capacity,
            radius,
            items: Vec::with_capacity(capacity),
            first_match,
        }
    }

    pub fn items(&self) -> &[ResultItem<F>] {
        &self.items
    }

    pub fn into_vec(self) -> Vec<ResultItem<F>> {
        self.items
    }
}

impl<F: Real> ResultSet<F> for RknnResultSet<F> {
    #[inline]
    fn add_point(&mut self, distance: F, index: usize) -> bool {
        insert_sorted_bounded(
            &mut self.items,
            ResultItem::new(index, distance),
            self.capacity,
            self.first_match,
        );
        true
    }

    #[inline]
    fn worst_dist(&self) -> F {
        if self.items.len() < self.capacity || self.items.is_empty() {
            self.radius
        } else {
            self.items[self.items.len() - 1].distance
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.items.len()
    }

    #[inline]
    fn full(&self) -> bool {
        self.items.len() == self.capacity
    }

    #[inline]
    fn sort(&mut self) {
        // RKNN insertion keeps the vector sorted already.
    }
}

/// Result set for radius queries. The radius comparison is strict (`distance < radius`),
/// matching nanoflann's `RadiusResultSet::addPoint()`.
#[derive(Clone, Debug)]
pub struct RadiusResultSet<F: Real> {
    radius: F,
    items: Vec<ResultItem<F>>,
}

impl<F: Real> RadiusResultSet<F> {
    pub fn new(radius: F) -> Self {
        Self {
            radius,
            items: Vec::new(),
        }
    }

    pub fn with_capacity(radius: F, capacity: usize) -> Self {
        Self {
            radius,
            items: Vec::with_capacity(capacity),
        }
    }

    pub fn items(&self) -> &[ResultItem<F>] {
        &self.items
    }

    pub fn into_vec(self) -> Vec<ResultItem<F>> {
        self.items
    }

    pub fn worst_item(&self) -> Option<ResultItem<F>> {
        self.items
            .iter()
            .copied()
            .max_by(|a, b| compare_distance(a, b))
    }
}

impl<F: Real> ResultSet<F> for RadiusResultSet<F> {
    #[inline]
    fn add_point(&mut self, distance: F, index: usize) -> bool {
        if distance < self.radius {
            self.items.push(ResultItem::new(index, distance));
        }
        true
    }

    #[inline]
    fn worst_dist(&self) -> F {
        self.radius
    }

    #[inline]
    fn size(&self) -> usize {
        self.items.len()
    }

    #[inline]
    fn full(&self) -> bool {
        true
    }

    #[inline]
    fn sort(&mut self) {
        self.items.sort_by(compare_distance);
    }
}

/// A fixed-capacity, stack-allocated result set for k-nearest neighbor queries.
///
/// This type is the recommended way to perform small-k KNN searches with **zero heap
/// allocations** in the result buffer (the main remaining hot-path allocation identified
/// in assembly analysis).
///
/// `K` is the maximum number of neighbors to return. For `k <= 32` this is typically
/// faster and more cache-friendly than a heap-based approach.
///
/// # Example
/// ```rust
/// use nanoflann::{KdTree, SmallKnnResultSet, L2, KdTreeParams};
/// # use nanoflann::PointCloud;
/// # let cloud = PointCloud::new(vec![vec![0.0, 0.0], vec![1.0, 0.0]]).unwrap();
/// # let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default()).unwrap();
///
/// let mut result: SmallKnnResultSet<f64, 8> = SmallKnnResultSet::new();
/// tree.knn_search_into(&[0.1, 0.2], &mut result, Default::default()).unwrap();
///
/// for item in result.as_slice() {
///     println!("{}: {}", item.index, item.distance);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct SmallKnnResultSet<F: Real, const K: usize> {
    len: usize,
    first_match: bool,
    items: [std::mem::MaybeUninit<ResultItem<F>>; K],
}

impl<F: Real, const K: usize> SmallKnnResultSet<F, K> {
    /// Creates a new empty result set with capacity `K`.
    pub fn new() -> Self {
        Self::with_first_match(false)
    }

    /// Creates a new empty result set with capacity `K` and the given `first_match` policy.
    pub fn with_first_match(first_match: bool) -> Self {
        Self {
            len: 0,
            first_match,
            items: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    /// Returns the number of results currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the result set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the result set has reached its capacity.
    #[inline]
    pub fn full(&self) -> bool {
        self.len == K
    }

    /// Returns a slice of the current results.
    ///
    /// The results are kept sorted by distance (ascending) when using the default
    /// insertion policy.
    #[inline]
    pub fn as_slice(&self) -> &[ResultItem<F>] {
        // SAFETY: We only ever expose initialized elements [0..len]
        unsafe { std::slice::from_raw_parts(self.items.as_ptr() as *const ResultItem<F>, self.len) }
    }

    /// Consumes the result set and returns the results as a `Vec`.
    ///
    /// This is useful when you need owned data but still want the zero-allocation
    /// search path.
    pub fn into_vec(self) -> Vec<ResultItem<F>> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            // SAFETY: elements [0..len] are initialized
            let item = unsafe { self.items[i].assume_init() };
            out.push(item);
        }
        // Prevent double-drop of the MaybeUninit contents
        std::mem::forget(self);
        out
    }
}

impl<F: Real, const K: usize> Default for SmallKnnResultSet<F, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Real, const K: usize> Drop for SmallKnnResultSet<F, K> {
    fn drop(&mut self) {
        for i in 0..self.len {
            // SAFETY: only drop initialized elements
            unsafe {
                self.items[i].assume_init_drop();
            }
        }
    }
}

impl<F: Real, const K: usize> ResultSet<F> for SmallKnnResultSet<F, K> {
    #[inline]
    fn add_point(&mut self, distance: F, index: usize) -> bool {
        if K == 0 {
            return true;
        }

        let item = ResultItem::new(index, distance);

        // Find insertion position (same logic as insert_sorted_bounded)
        let mut pos = self.len;
        while pos > 0 {
            // SAFETY: pos-1 < len, so it is initialized
            let prev = unsafe { self.items[pos - 1].assume_init() };
            let should_shift = prev.distance > item.distance
                || (self.first_match && prev.distance == item.distance && prev.index > item.index);
            if should_shift {
                pos -= 1;
            } else {
                break;
            }
        }

        if pos < K {
            // Make room if necessary
            if self.len < K {
                self.len += 1;
            } else if pos < self.len {
                // Shift elements right to make room (we are at capacity)
                // SAFETY: we are moving initialized elements [pos .. len-1]
                unsafe {
                    let src = self.items.as_ptr().add(pos);
                    let dst = self.items.as_mut_ptr().add(pos + 1);
                    std::ptr::copy(src, dst, self.len - pos - 1);
                    // Drop the last element that fell off
                    self.items[self.len - 1].assume_init_drop();
                }
            } else {
                // New item is worse than everything we have and we're full
                return true;
            }

            // Write the new item
            // SAFETY: pos is within the initialized or newly extended range
            self.items[pos].write(item);
        }
        true
    }

    #[inline]
    fn worst_dist(&self) -> F {
        if self.len < K || self.len == 0 {
            F::max_value()
        } else {
            // SAFETY: len-1 is valid and initialized
            unsafe { self.items[self.len - 1].assume_init().distance }
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    fn full(&self) -> bool {
        self.len == K
    }

    #[inline]
    fn sort(&mut self) {
        // Insertion already keeps the array sorted for the common case.
        // If the user requested unsorted results we would need to sort here,
        // but for SmallKnnResultSet we keep the sorted invariant for simplicity.
    }
}
