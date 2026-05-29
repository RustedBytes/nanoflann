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

pub(crate) trait ResultSet<F: Real> {
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
