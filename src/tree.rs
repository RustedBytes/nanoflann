use crate::dataset::KdTreeDataset;
use crate::error::{KdTreeError, Result};
use crate::metric::DistanceMetric;
use crate::real::Real;
use crate::result_set::{KnnResultSet, RadiusResultSet, ResultItem, ResultSet, RknnResultSet};
use std::io::{Read, Write};

const INDEX_MAGIC: &[u8; 7] = b"NKDRS01";

/// Closed interval for one dimension of a bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval<F: Real> {
    pub low: F,
    pub high: F,
}

impl<F: Real> Interval<F> {
    pub fn new(low: F, high: F) -> Self {
        Self { low, high }
    }

    pub fn zero() -> Self {
        Self {
            low: F::zero(),
            high: F::zero(),
        }
    }
}

/// Search options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParameters<F: Real> {
    /// Epsilon for approximate search. `0` performs exact pruning.
    pub eps: F,
    /// Whether radius-style results should be sorted by ascending distance.
    pub sorted: bool,
}

impl<F: Real> Default for SearchParameters<F> {
    fn default() -> Self {
        Self {
            eps: F::zero(),
            sorted: true,
        }
    }
}

/// KD-tree build parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KdTreeParams {
    /// Maximum number of points in a leaf node. Default: 10.
    pub leaf_max_size: usize,
    /// Construct the object without immediately building the index.
    pub skip_initial_build: bool,
    /// Kept for API familiarity with nanoflann. This safe Rust port currently
    /// builds recursively on one thread.
    pub n_thread_build: usize,
    /// Tie-break equal distances by smaller index, similar to defining
    /// `NANOFLANN_FIRST_MATCH` in the C++ header.
    pub first_match: bool,
}

impl Default for KdTreeParams {
    fn default() -> Self {
        Self {
            leaf_max_size: 10,
            skip_initial_build: false,
            n_thread_build: 1,
            first_match: false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Node<F: Real> {
    Leaf {
        left: usize,
        right: usize,
    },
    Split {
        divfeat: usize,
        divlow: F,
        divhigh: F,
        child1: Box<Node<F>>,
        child2: Box<Node<F>>,
    },
}

/// Safe Rust KD-tree equivalent of nanoflann's static single-index adaptor.
pub struct KdTree<'a, F, D, M>
where
    F: Real,
    D: KdTreeDataset<F>,
    M: DistanceMetric<F, D>,
{
    dataset: &'a D,
    metric: M,
    params: KdTreeParams,
    dim: usize,
    pub(crate) v_acc: Vec<usize>,
    pub(crate) root_node: Option<Box<Node<F>>>,
    root_bbox: Vec<Interval<F>>,
    size: usize,
    size_at_index_build: usize,
}

impl<'a, F, D, M> KdTree<'a, F, D, M>
where
    F: Real,
    D: KdTreeDataset<F>,
    M: DistanceMetric<F, D>,
{
    /// Creates a KD-tree over `dataset`. Unless `params.skip_initial_build` is
    /// set, this also builds the index.
    pub fn new(dim: usize, dataset: &'a D, metric: M, params: KdTreeParams) -> Result<Self> {
        if dim == 0 {
            return Err(KdTreeError::InvalidDimensionality);
        }
        if params.leaf_max_size == 0 {
            return Err(KdTreeError::InvalidLeafMaxSize);
        }

        let size = dataset.kdtree_get_point_count();
        let mut tree = Self {
            dataset,
            metric,
            params,
            dim,
            v_acc: Vec::new(),
            root_node: None,
            root_bbox: vec![Interval::zero(); dim],
            size,
            size_at_index_build: 0,
        };

        if !params.skip_initial_build {
            tree.build_index()?;
        }

        Ok(tree)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn size_at_index_build(&self) -> usize {
        self.size_at_index_build
    }

    pub fn params(&self) -> KdTreeParams {
        self.params
    }

    pub fn root_bbox(&self) -> &[Interval<F>] {
        &self.root_bbox
    }

    /// Builds or rebuilds the static index over all current dataset points.
    pub fn build_index(&mut self) -> Result<()> {
        self.size = self.dataset.kdtree_get_point_count();
        self.init_vind();
        self.root_node = None;
        self.size_at_index_build = self.size;

        if self.size == 0 {
            self.root_bbox = vec![Interval::zero(); self.dim];
            return Ok(());
        }

        self.compute_bounding_box_current_indices()?;
        let mut bbox = self.root_bbox.clone();
        let root = self.divide_tree(0, self.v_acc.len(), &mut bbox)?;
        self.root_bbox = bbox;
        self.root_node = Some(root);
        Ok(())
    }

    pub(crate) fn rebuild_current_indices(&mut self) -> Result<()> {
        self.size = self.v_acc.len();
        self.root_node = None;
        self.size_at_index_build = self.size;

        if self.v_acc.is_empty() {
            self.root_bbox = vec![Interval::zero(); self.dim];
            return Ok(());
        }

        self.validate_indices()?;
        self.compute_bounding_box_current_indices()?;
        let mut bbox = self.root_bbox.clone();
        let root = self.divide_tree(0, self.v_acc.len(), &mut bbox)?;
        self.root_bbox = bbox;
        self.root_node = Some(root);
        Ok(())
    }

    fn init_vind(&mut self) {
        self.v_acc.clear();
        self.v_acc.extend(0..self.size);
    }

    fn validate_indices(&self) -> Result<()> {
        let len = self.dataset.kdtree_get_point_count();
        for &idx in &self.v_acc {
            if idx >= len {
                return Err(KdTreeError::IndexOutOfBounds { index: idx, len });
            }
        }
        Ok(())
    }

    fn dataset_get(&self, idx: usize, component: usize) -> F {
        self.dataset.kdtree_get_pt(idx, component)
    }

    fn compute_bounding_box_current_indices(&mut self) -> Result<()> {
        let mut bbox = vec![Interval::zero(); self.dim];

        if self.dataset.kdtree_get_bbox(&mut bbox) {
            Self::validate_bbox_shape(self.dim, &bbox)?;
            self.root_bbox = bbox;
            return Ok(());
        }

        if self.v_acc.is_empty() {
            return Err(KdTreeError::InvalidBoundingBox(
                "cannot compute a bounding box for an empty index".to_owned(),
            ));
        }

        for dim in 0..self.dim {
            let value = self.dataset_get(self.v_acc[0], dim);
            bbox[dim] = Interval::new(value, value);
        }

        for &idx in self.v_acc.iter().skip(1) {
            for dim in 0..self.dim {
                let value = self.dataset_get(idx, dim);
                if value < bbox[dim].low {
                    bbox[dim].low = value;
                }
                if value > bbox[dim].high {
                    bbox[dim].high = value;
                }
            }
        }

        self.root_bbox = bbox;
        Ok(())
    }

    fn validate_bbox_shape(dim: usize, bbox: &[Interval<F>]) -> Result<()> {
        if bbox.len() != dim {
            return Err(KdTreeError::BoundingBoxDimensionalityMismatch {
                expected: dim,
                got: bbox.len(),
            });
        }
        for (i, interval) in bbox.iter().enumerate() {
            if interval.high < interval.low {
                return Err(KdTreeError::InvalidBoundingBox(format!(
                    "dimension {i} has high < low"
                )));
            }
        }
        Ok(())
    }

    fn divide_tree(
        &mut self,
        left: usize,
        right: usize,
        bbox: &mut Vec<Interval<F>>,
    ) -> Result<Box<Node<F>>> {
        debug_assert!(left < right);
        let count = right - left;

        if count <= self.params.leaf_max_size {
            for dim in 0..self.dim {
                let value = self.dataset_get(self.v_acc[left], dim);
                bbox[dim] = Interval::new(value, value);
            }
            for offset in (left + 1)..right {
                let idx = self.v_acc[offset];
                for dim in 0..self.dim {
                    let value = self.dataset_get(idx, dim);
                    if value < bbox[dim].low {
                        bbox[dim].low = value;
                    }
                    if value > bbox[dim].high {
                        bbox[dim].high = value;
                    }
                }
            }
            return Ok(Box::new(Node::Leaf { left, right }));
        }

        let (index, cutfeat, cutval) = self.middle_split(left, count, bbox)?;

        let mut left_bbox = bbox.clone();
        left_bbox[cutfeat].high = cutval;
        let child1 = self.divide_tree(left, left + index, &mut left_bbox)?;

        let mut right_bbox = bbox.clone();
        right_bbox[cutfeat].low = cutval;
        let child2 = self.divide_tree(left + index, right, &mut right_bbox)?;

        let divlow = left_bbox[cutfeat].high;
        let divhigh = right_bbox[cutfeat].low;

        for dim in 0..self.dim {
            bbox[dim].low = if left_bbox[dim].low < right_bbox[dim].low {
                left_bbox[dim].low
            } else {
                right_bbox[dim].low
            };
            bbox[dim].high = if left_bbox[dim].high > right_bbox[dim].high {
                left_bbox[dim].high
            } else {
                right_bbox[dim].high
            };
        }

        Ok(Box::new(Node::Split {
            divfeat: cutfeat,
            divlow,
            divhigh,
            child1,
            child2,
        }))
    }

    fn middle_split(
        &mut self,
        ind: usize,
        count: usize,
        bbox: &[Interval<F>],
    ) -> Result<(usize, usize, F)> {
        debug_assert!(count > 0);
        let eps = F::from_f64(0.00001);
        let one_minus_eps = F::one() - eps;

        let mut max_span = bbox[0].high - bbox[0].low;
        for dim in 1..self.dim {
            let span = bbox[dim].high - bbox[dim].low;
            if span > max_span {
                max_span = span;
            }
        }

        let mut candidates = Vec::with_capacity(self.dim);
        for dim in 0..self.dim {
            if bbox[dim].high - bbox[dim].low >= one_minus_eps * max_span {
                candidates.push(dim);
            }
        }
        if candidates.is_empty() {
            candidates.push(0);
        }

        let mut cutfeat = 0;
        let mut max_spread = F::from_f64(-1.0);
        let mut min_elem = F::zero();
        let mut max_elem = F::zero();

        for dim in candidates {
            let first = self.dataset_get(self.v_acc[ind], dim);
            let mut local_min = first;
            let mut local_max = first;

            for k in 1..count {
                let value = self.dataset_get(self.v_acc[ind + k], dim);
                if value < local_min {
                    local_min = value;
                }
                if value > local_max {
                    local_max = value;
                }
            }

            let spread = local_max - local_min;
            if spread > max_spread {
                cutfeat = dim;
                max_spread = spread;
                min_elem = local_min;
                max_elem = local_max;
            }
        }

        let two = F::from_f64(2.0);
        let mut cutval = (bbox[cutfeat].low + bbox[cutfeat].high) / two;
        if cutval < min_elem {
            cutval = min_elem;
        }
        if cutval > max_elem {
            cutval = max_elem;
        }

        let (lim1, lim2) = self.plane_split(ind, count, cutfeat, cutval);
        let half = count / 2;
        let index = if lim1 > half {
            lim1
        } else if lim2 < half {
            lim2
        } else {
            half
        };

        if index == 0 || index >= count {
            return Err(KdTreeError::InvalidBoundingBox(
                "split did not divide the point set".to_owned(),
            ));
        }

        Ok((index, cutfeat, cutval))
    }

    fn plane_split(
        &mut self,
        ind: usize,
        count: usize,
        cutfeat: usize,
        cutval: F,
    ) -> (usize, usize) {
        let mut left = 0usize;
        let mut mid = 0usize;
        let mut right = count; // exclusive

        while mid < right {
            let value = self.dataset_get(self.v_acc[ind + mid], cutfeat);
            if value < cutval {
                self.v_acc.swap(ind + left, ind + mid);
                left += 1;
                mid += 1;
            } else if value > cutval {
                right -= 1;
                self.v_acc.swap(ind + mid, ind + right);
            } else {
                mid += 1;
            }
        }

        (left, mid)
    }

    fn compute_initial_distances(&self, query: &[F], dists: &mut [F]) -> F {
        let mut dist = F::zero();
        for dim in 0..self.dim {
            if query[dim] < self.root_bbox[dim].low {
                dists[dim] = self
                    .metric
                    .accum_dist(query[dim], self.root_bbox[dim].low, dim);
                dist = dist + dists[dim];
            }
            if query[dim] > self.root_bbox[dim].high {
                dists[dim] = self
                    .metric
                    .accum_dist(query[dim], self.root_bbox[dim].high, dim);
                dist = dist + dists[dim];
            }
        }
        dist
    }

    fn ensure_query_dim(&self, query: &[F]) -> Result<()> {
        if query.len() < self.dim {
            return Err(KdTreeError::QueryDimensionalityMismatch {
                expected_at_least: self.dim,
                got: query.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn find_neighbors_set<R: ResultSet<F>>(
        &self,
        result: &mut R,
        query: &[F],
        search_params: SearchParameters<F>,
        active: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<bool> {
        self.ensure_query_dim(query)?;
        if self.size == 0 {
            return Ok(false);
        }
        let root = self
            .root_node
            .as_deref()
            .ok_or(KdTreeError::IndexNotBuilt)?;

        let eps_error = F::one() + search_params.eps;
        let mut dists = vec![F::zero(); self.dim];
        let dist = self.compute_initial_distances(query, &mut dists);
        self.search_level(result, query, root, dist, &mut dists, eps_error, active)?;

        if search_params.sorted {
            result.sort();
        }
        Ok(result.full())
    }

    fn search_level<R: ResultSet<F>>(
        &self,
        result_set: &mut R,
        query: &[F],
        node: &Node<F>,
        mut mindist: F,
        dists: &mut [F],
        eps_error: F,
        active: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<bool> {
        match node {
            Node::Leaf { left, right } => {
                for offset in *left..*right {
                    let idx = self.v_acc[offset];
                    if let Some(is_active) = active {
                        if !is_active(idx) {
                            continue;
                        }
                    }
                    let dist = self
                        .metric
                        .eval_metric(self.dataset, query, idx, self.dim, None);
                    if dist < result_set.worst_dist() && !result_set.add_point(dist, idx) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Node::Split {
                divfeat,
                divlow,
                divhigh,
                child1,
                child2,
            } => {
                let idx = *divfeat;
                let value = query[idx];
                let diff1 = value - *divlow;
                let diff2 = value - *divhigh;

                let (best_child, other_child, cut_dist) = if diff1 + diff2 < F::zero() {
                    (
                        child1.as_ref(),
                        child2.as_ref(),
                        self.metric.accum_dist(value, *divhigh, idx),
                    )
                } else {
                    (
                        child2.as_ref(),
                        child1.as_ref(),
                        self.metric.accum_dist(value, *divlow, idx),
                    )
                };

                if !self.search_level(
                    result_set, query, best_child, mindist, dists, eps_error, active,
                )? {
                    return Ok(false);
                }

                let old_dist = dists[idx];
                mindist = mindist + cut_dist - old_dist;
                dists[idx] = cut_dist;

                if mindist * eps_error <= result_set.worst_dist() {
                    if !self.search_level(
                        result_set,
                        query,
                        other_child,
                        mindist,
                        dists,
                        eps_error,
                        active,
                    )? {
                        return Ok(false);
                    }
                }

                dists[idx] = old_dist;
                Ok(true)
            }
        }
    }

    /// Returns the `num_closest` nearest neighbors.
    pub fn knn_search(&self, query: &[F], num_closest: usize) -> Result<Vec<ResultItem<F>>> {
        self.knn_search_with_params(query, num_closest, SearchParameters::default())
    }

    pub fn knn_search_with_params(
        &self,
        query: &[F],
        num_closest: usize,
        search_params: SearchParameters<F>,
    ) -> Result<Vec<ResultItem<F>>> {
        let mut result = KnnResultSet::with_first_match(num_closest, self.params.first_match);
        self.find_neighbors_set(&mut result, query, search_params, None)?;
        Ok(result.into_vec())
    }

    /// Returns all neighbors with `distance < radius`.
    pub fn radius_search(&self, query: &[F], radius: F) -> Result<Vec<ResultItem<F>>> {
        self.radius_search_with_params(query, radius, SearchParameters::default())
    }

    pub fn radius_search_with_params(
        &self,
        query: &[F],
        radius: F,
        search_params: SearchParameters<F>,
    ) -> Result<Vec<ResultItem<F>>> {
        let mut result = RadiusResultSet::new(radius);
        self.find_neighbors_set(&mut result, query, search_params, None)?;
        Ok(result.into_vec())
    }

    /// Returns the first `num_closest` neighbors that are also within `radius`.
    pub fn rknn_search(
        &self,
        query: &[F],
        num_closest: usize,
        radius: F,
    ) -> Result<Vec<ResultItem<F>>> {
        let mut result =
            RknnResultSet::with_first_match(num_closest, radius, self.params.first_match);
        self.find_neighbors_set(&mut result, query, SearchParameters::default(), None)?;
        Ok(result.into_vec())
    }

    /// Finds all point indices inside `bbox`. Bounds are inclusive.
    pub fn find_within_box(&self, bbox: &[Interval<F>]) -> Result<Vec<usize>> {
        Self::validate_bbox_shape(self.dim, bbox)?;
        if self.size == 0 {
            return Ok(Vec::new());
        }
        let root = self
            .root_node
            .as_deref()
            .ok_or(KdTreeError::IndexNotBuilt)?;
        let mut found = Vec::new();
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            match node {
                Node::Leaf { left, right } => {
                    for offset in *left..*right {
                        let idx = self.v_acc[offset];
                        if self.contains(bbox, idx) {
                            found.push(idx);
                        }
                    }
                }
                Node::Split {
                    divfeat,
                    divlow,
                    divhigh,
                    child1,
                    child2,
                } => {
                    if bbox[*divfeat].low <= *divlow {
                        stack.push(child1.as_ref());
                    }
                    if bbox[*divfeat].high >= *divhigh {
                        stack.push(child2.as_ref());
                    }
                }
            }
        }

        Ok(found)
    }

    fn contains(&self, bbox: &[Interval<F>], idx: usize) -> bool {
        for dim in 0..self.dim {
            let point = self.dataset.kdtree_get_pt(idx, dim);
            if point < bbox[dim].low || point > bbox[dim].high {
                return false;
            }
        }
        true
    }

    /// Saves the built index using a portable little-endian binary format.
    ///
    /// Like nanoflann, this stores only the index structure and not the dataset;
    /// load it into a tree over the same point data.
    pub fn save_index<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(INDEX_MAGIC)?;
        write_u64(writer, self.dim)?;
        write_u64(writer, self.size)?;
        write_u64(writer, self.size_at_index_build)?;
        write_u64(writer, self.params.leaf_max_size)?;
        write_u64(writer, self.v_acc.len())?;
        for &idx in &self.v_acc {
            write_u64(writer, idx)?;
        }
        write_u64(writer, self.root_bbox.len())?;
        for interval in &self.root_bbox {
            write_f64(writer, interval.low.to_f64())?;
            write_f64(writer, interval.high.to_f64())?;
        }
        match self.root_node.as_deref() {
            Some(node) => {
                writer.write_all(&[1])?;
                write_node(writer, node)?;
            }
            None => writer.write_all(&[0])?,
        }
        Ok(())
    }

    /// Loads an index previously saved with [`save_index`](Self::save_index).
    pub fn load_index<R: Read>(&mut self, reader: &mut R) -> Result<()> {
        let mut magic = [0u8; 7];
        reader.read_exact(&mut magic)?;
        if &magic != INDEX_MAGIC {
            return Err(KdTreeError::InvalidIndexFile("bad magic".to_owned()));
        }

        let dim = read_u64(reader)?;
        if dim != self.dim {
            return Err(KdTreeError::InvalidIndexFile(format!(
                "saved dimensionality {dim} does not match tree dimensionality {}",
                self.dim
            )));
        }
        self.size = read_u64(reader)?;
        self.size_at_index_build = read_u64(reader)?;
        self.params.leaf_max_size = read_u64(reader)?;

        let v_len = read_u64(reader)?;
        self.v_acc.clear();
        self.v_acc.reserve(v_len);
        for _ in 0..v_len {
            self.v_acc.push(read_u64(reader)?);
        }
        self.validate_indices()?;

        let bbox_len = read_u64(reader)?;
        if bbox_len != self.dim {
            return Err(KdTreeError::InvalidIndexFile(format!(
                "saved bbox has {bbox_len} dimensions, expected {}",
                self.dim
            )));
        }
        self.root_bbox.clear();
        for _ in 0..bbox_len {
            let low = F::from_f64(read_f64(reader)?);
            let high = F::from_f64(read_f64(reader)?);
            self.root_bbox.push(Interval::new(low, high));
        }
        Self::validate_bbox_shape(self.dim, &self.root_bbox)?;

        let mut present = [0u8; 1];
        reader.read_exact(&mut present)?;
        self.root_node = if present[0] == 0 {
            None
        } else if present[0] == 1 {
            Some(read_node(reader)?)
        } else {
            return Err(KdTreeError::InvalidIndexFile(format!(
                "invalid root-node presence byte {}",
                present[0]
            )));
        };
        Ok(())
    }
}

fn write_u64<W: Write>(writer: &mut W, value: usize) -> Result<()> {
    writer.write_all(&(value as u64).to_le_bytes())?;
    Ok(())
}

fn read_u64<R: Read>(reader: &mut R) -> Result<usize> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes) as usize)
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn write_node<W: Write, F: Real>(writer: &mut W, node: &Node<F>) -> Result<()> {
    match node {
        Node::Leaf { left, right } => {
            writer.write_all(&[0])?;
            write_u64(writer, *left)?;
            write_u64(writer, *right)?;
        }
        Node::Split {
            divfeat,
            divlow,
            divhigh,
            child1,
            child2,
        } => {
            writer.write_all(&[1])?;
            write_u64(writer, *divfeat)?;
            write_f64(writer, divlow.to_f64())?;
            write_f64(writer, divhigh.to_f64())?;
            write_node(writer, child1)?;
            write_node(writer, child2)?;
        }
    }
    Ok(())
}

fn read_node<R: Read, F: Real>(reader: &mut R) -> Result<Box<Node<F>>> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        0 => {
            let left = read_u64(reader)?;
            let right = read_u64(reader)?;
            Ok(Box::new(Node::Leaf { left, right }))
        }
        1 => {
            let divfeat = read_u64(reader)?;
            let divlow = F::from_f64(read_f64(reader)?);
            let divhigh = F::from_f64(read_f64(reader)?);
            let child1 = read_node(reader)?;
            let child2 = read_node(reader)?;
            Ok(Box::new(Node::Split {
                divfeat,
                divlow,
                divhigh,
                child1,
                child2,
            }))
        }
        other => Err(KdTreeError::InvalidIndexFile(format!(
            "invalid node tag {other}"
        ))),
    }
}
