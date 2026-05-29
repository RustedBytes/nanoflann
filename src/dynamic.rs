use crate::dataset::KdTreeDataset;
use crate::error::{KdTreeError, Result};
use crate::metric::DistanceMetric;
use crate::real::Real;
use crate::result_set::{KnnResultSet, RadiusResultSet, ResultItem, RknnResultSet};
use crate::tree::{KdTree, KdTreeParams, SearchParameters};
use std::collections::HashMap;

/// Dynamic KD-tree implemented as a logarithmic set of static subtrees with
/// lazy deletion, matching the high-level approach in nanoflann's dynamic
/// adaptor.
pub struct DynamicKdTree<'a, F, D, M>
where
    F: Real,
    D: KdTreeDataset<F>,
    M: DistanceMetric<F, D>,
{
    dataset: &'a D,
    metric: M,
    params: KdTreeParams,
    dim: usize,
    tree_count: usize,
    maximum_point_count: usize,
    point_count: usize,
    tree_index: Vec<Option<usize>>,
    removed_points: HashMap<usize, usize>,
    indices: Vec<KdTree<'a, F, D, M>>,
}

impl<'a, F, D, M> DynamicKdTree<'a, F, D, M>
where
    F: Real,
    D: KdTreeDataset<F>,
    M: DistanceMetric<F, D>,
{
    /// Creates a dynamic KD-tree. Existing dataset points are inserted
    /// immediately, just like the C++ dynamic adaptor constructor.
    pub fn new(
        dim: usize,
        dataset: &'a D,
        metric: M,
        params: KdTreeParams,
        maximum_point_count: usize,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(KdTreeError::InvalidDimensionality);
        }
        if params.leaf_max_size == 0 {
            return Err(KdTreeError::InvalidLeafMaxSize);
        }

        let safe_max = maximum_point_count.max(1);
        let tree_count = floor_log2(safe_max) + 1;
        let mut this = Self {
            dataset,
            metric,
            params,
            dim,
            tree_count,
            maximum_point_count: safe_max,
            point_count: 0,
            tree_index: Vec::new(),
            removed_points: HashMap::new(),
            indices: Vec::new(),
        };
        this.init_indices()?;

        let initial = dataset.kdtree_get_point_count();
        if initial > 0 {
            this.add_points(0, initial - 1)?;
        }
        Ok(this)
    }

    fn init_indices(&mut self) -> Result<()> {
        self.indices.clear();
        let mut params = self.params;
        params.skip_initial_build = true;
        self.indices.reserve(self.tree_count);
        for _ in 0..self.tree_count {
            let mut tree = KdTree::new(self.dim, self.dataset, self.metric.clone(), params)?;
            // A skipped-build static tree initializes with the backing dataset's
            // current point count. Dynamic subtrees start physically empty, so
            // reset their internal size to the length of `v_acc` (zero).
            tree.rebuild_current_indices()?;
            self.indices.push(tree);
        }
        Ok(())
    }

    fn first_zero_bit(mut num: usize) -> usize {
        let mut pos = 0usize;
        while num & 1 == 1 {
            num >>= 1;
            pos += 1;
        }
        pos
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn point_count(&self) -> usize {
        self.point_count
    }

    pub fn active_count(&self) -> usize {
        self.tree_index
            .iter()
            .filter(|entry| entry.is_some())
            .count()
    }

    pub fn all_indices(&self) -> &[KdTree<'a, F, D, M>] {
        &self.indices
    }

    /// Inserts points in the inclusive range `[start, end]`.
    ///
    /// New insertions must be contiguous with the current point count. Calling
    /// this on a removed point index reactivates the existing physical entry
    /// instead of adding a duplicate, preserving the behavior of the updated C++
    /// code.
    pub fn add_points(&mut self, start: usize, end: usize) -> Result<()> {
        if start > end {
            return Ok(());
        }
        let dataset_len = self.dataset.kdtree_get_point_count();
        if end >= dataset_len {
            return Err(KdTreeError::IndexOutOfBounds {
                index: end,
                len: dataset_len,
            });
        }

        let mut max_index_to_rebuild: Option<usize> = None;

        for idx in start..=end {
            if let Some(tree_id) = self.removed_points.remove(&idx) {
                if self.tree_index.len() <= idx {
                    self.tree_index.resize(idx + 1, None);
                }
                self.tree_index[idx] = Some(tree_id);
                continue;
            }

            if idx != self.point_count {
                return Err(KdTreeError::NonContiguousInsertion {
                    expected: self.point_count,
                    got: idx,
                });
            }
            if self.point_count >= self.maximum_point_count {
                return Err(KdTreeError::MaximumPointCountExceeded {
                    maximum_point_count: self.maximum_point_count,
                });
            }

            let pos = Self::first_zero_bit(self.point_count);
            if pos >= self.indices.len() {
                return Err(KdTreeError::MaximumPointCountExceeded {
                    maximum_point_count: self.maximum_point_count,
                });
            }
            max_index_to_rebuild = Some(max_index_to_rebuild.map_or(pos, |m| m.max(pos)));

            if self.tree_index.len() <= idx {
                self.tree_index.resize(idx + 1, None);
            }
            self.tree_index[idx] = Some(pos);

            for tree_id in 0..pos {
                let (left_part, right_part) = self.indices.split_at_mut(pos);
                let src_tree = &mut left_part[tree_id];
                let dst_tree = &mut right_part[0];
                for point_idx in src_tree.v_acc.drain(..) {
                    dst_tree.v_acc.push(point_idx);
                    if self
                        .tree_index
                        .get(point_idx)
                        .and_then(|entry| *entry)
                        .is_some()
                    {
                        self.tree_index[point_idx] = Some(pos);
                    } else {
                        self.removed_points.insert(point_idx, pos);
                    }
                }
            }

            self.indices[pos].v_acc.push(idx);
            self.point_count += 1;
        }

        if let Some(max_index) = max_index_to_rebuild {
            for tree_id in 0..=max_index {
                self.indices[tree_id].rebuild_current_indices()?;
            }
        }
        Ok(())
    }

    /// Lazily removes a point. Out-of-range and already-removed indices are no-ops.
    pub fn remove_point(&mut self, idx: usize) {
        if idx >= self.point_count {
            return;
        }
        if let Some(Some(tree_id)) = self.tree_index.get(idx).copied() {
            self.removed_points.insert(idx, tree_id);
            self.tree_index[idx] = None;
        }
    }

    fn is_active(&self, idx: usize) -> bool {
        self.tree_index.get(idx).and_then(|entry| *entry).is_some()
    }

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
        self.find_neighbors_result_set(&mut result, query, search_params)?;
        Ok(result.into_vec())
    }

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
        self.find_neighbors_result_set(&mut result, query, search_params)?;
        Ok(result.into_vec())
    }

    pub fn rknn_search(
        &self,
        query: &[F],
        num_closest: usize,
        radius: F,
    ) -> Result<Vec<ResultItem<F>>> {
        let mut result =
            RknnResultSet::with_first_match(num_closest, radius, self.params.first_match);
        self.find_neighbors_result_set(&mut result, query, SearchParameters::default())?;
        Ok(result.into_vec())
    }

    fn find_neighbors_result_set<R: crate::result_set::ResultSet<F>>(
        &self,
        result: &mut R,
        query: &[F],
        search_params: SearchParameters<F>,
    ) -> Result<bool> {
        let active = |idx: usize| self.is_active(idx);
        let mut per_tree_params = search_params;
        per_tree_params.sorted = false;
        for tree in &self.indices {
            tree.find_neighbors_set(
                result,
                query,
                per_tree_params,
                crate::tree::FnFilter(active),
            )?;
        }
        if search_params.sorted {
            result.sort();
        }
        Ok(result.full())
    }
}

fn floor_log2(value: usize) -> usize {
    usize::BITS as usize - 1 - value.leading_zeros() as usize
}
