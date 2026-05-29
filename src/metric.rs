#![allow(clippy::needless_range_loop)]

use crate::dataset::KdTreeDataset;
use crate::real::Real;

/// Distance metric interface used by tree construction and pruning.
pub trait DistanceMetric<F: Real, D: KdTreeDataset<F>>: Clone {
    /// Full distance between a query vector and point `b_idx` in `dataset`.
    ///
    /// For L2-family metrics, this returns squared distance, matching
    /// nanoflann's convention.
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        worst_dist: Option<F>,
    ) -> F;

    /// Per-dimension contribution used by KD-tree bounding-box pruning.
    fn accum_dist(&self, a: F, b: F, dim: usize) -> F;
}

/// Manhattan/L1 metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct L1;

/// Squared Euclidean metric. Equivalent to nanoflann's generic `metric_L2`.
#[derive(Clone, Copy, Debug, Default)]
pub struct L2;

/// Squared Euclidean metric without unrolling/early-abort behavior. Equivalent
/// to nanoflann's `metric_L2_Simple` surface semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct L2Simple;

/// Wrapped SO(2) angular difference metric. Inputs are assumed to be in
/// `[-pi, pi]`, matching the original header's documented precondition.
#[derive(Clone, Copy, Debug, Default)]
pub struct SO2;

/// SO(3) metric implemented as L2-simple over the provided representation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SO3;

impl<F: Real, D: KdTreeDataset<F>> DistanceMetric<F, D> for L1 {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        worst_dist: Option<F>,
    ) -> F {
        let mut result = F::zero();
        let check_worst = worst_dist.filter(|w| *w > F::zero());
        for dim in 0..size {
            result = result + (query[dim] - dataset.kdtree_get_pt(b_idx, dim)).abs();
            if let Some(worst) = check_worst {
                if result > worst {
                    return result;
                }
            }
        }
        result
    }

    #[inline]
    fn accum_dist(&self, a: F, b: F, _dim: usize) -> F {
        (a - b).abs()
    }
}

impl<F: Real, D: KdTreeDataset<F>> DistanceMetric<F, D> for L2 {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        worst_dist: Option<F>,
    ) -> F {
        let mut result = F::zero();
        let check_worst = worst_dist.filter(|w| *w > F::zero());
        for dim in 0..size {
            let diff = query[dim] - dataset.kdtree_get_pt(b_idx, dim);
            result = result + diff * diff;
            if let Some(worst) = check_worst {
                if result > worst {
                    return result;
                }
            }
        }
        result
    }

    #[inline]
    fn accum_dist(&self, a: F, b: F, _dim: usize) -> F {
        let diff = a - b;
        diff * diff
    }
}

impl<F: Real, D: KdTreeDataset<F>> DistanceMetric<F, D> for L2Simple {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        _worst_dist: Option<F>,
    ) -> F {
        let mut result = F::zero();
        for dim in 0..size {
            let diff = query[dim] - dataset.kdtree_get_pt(b_idx, dim);
            result = result + diff * diff;
        }
        result
    }

    #[inline]
    fn accum_dist(&self, a: F, b: F, _dim: usize) -> F {
        let diff = a - b;
        diff * diff
    }
}

impl<F: Real, D: KdTreeDataset<F>> DistanceMetric<F, D> for SO2 {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        _worst_dist: Option<F>,
    ) -> F {
        <SO2 as DistanceMetric<F, D>>::accum_dist(
            self,
            query[size - 1],
            dataset.kdtree_get_pt(b_idx, size - 1),
            size - 1,
        )
    }

    #[inline]
    fn accum_dist(&self, a: F, b: F, _dim: usize) -> F {
        let pi = F::from_f64(std::f64::consts::PI);
        let two_pi = F::from_f64(2.0 * std::f64::consts::PI);
        let mut result = b - a;
        if result > pi {
            result = result - two_pi;
        } else if result < -pi {
            result = result + two_pi;
        }
        result
    }
}

impl<F: Real, D: KdTreeDataset<F>> DistanceMetric<F, D> for SO3 {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        worst_dist: Option<F>,
    ) -> F {
        <L2Simple as DistanceMetric<F, D>>::eval_metric(
            &L2Simple, dataset, query, b_idx, size, worst_dist,
        )
    }

    #[inline]
    fn accum_dist(&self, a: F, b: F, dim: usize) -> F {
        <L2Simple as DistanceMetric<F, D>>::accum_dist(&L2Simple, a, b, dim)
    }
}
