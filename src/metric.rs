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

impl<F: Real + 'static, D: KdTreeDataset<F>> DistanceMetric<F, D> for L2 {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        worst_dist: Option<F>,
    ) -> F {
        // SIMD fast path for f32/f64 when the feature is enabled
        #[cfg(feature = "simd")]
        {
            use std::any::TypeId;

            if TypeId::of::<F>() == TypeId::of::<f32>() {
                // SAFETY: TypeId check guarantees the cast is valid
                let q32 = unsafe { &*(query as *const [F] as *const [f32]) };
                if let Some(pt) = dataset.kdtree_get_point(b_idx) {
                    let p32 = unsafe { &*(pt as *const [F] as *const [f32]) };
                    let dist = simd::l2_squared_f32(q32, p32, worst_dist.map(|w| w.to_f64() as f32));
                    return F::from_f64(dist as f64);
                } else {
                    // gather fallback (still vectorized in the helper if we wanted, but scalar gather here)
                    let mut result = 0.0f32;
                    let check = worst_dist.map(|w| w.to_f64() as f32).filter(|w| *w > 0.0);
                    for dim in 0..size {
                        let diff = q32[dim] - dataset.kdtree_get_pt(b_idx, dim).to_f64() as f32;
                        result += diff * diff;
                        if let Some(w) = check {
                            if result > w {
                                return F::from_f64(result as f64);
                            }
                        }
                    }
                    return F::from_f64(result as f64);
                }
            }

            if TypeId::of::<F>() == TypeId::of::<f64>() {
                let q64 = unsafe { &*(query as *const [F] as *const [f64]) };
                if let Some(pt) = dataset.kdtree_get_point(b_idx) {
                    let p64 = unsafe { &*(pt as *const [F] as *const [f64]) };
                    let dist = simd::l2_squared_f64(q64, p64, worst_dist.map(|w| w.to_f64()));
                    return F::from_f64(dist);
                } else {
                    let mut result = 0.0f64;
                    let check = worst_dist.map(|w| w.to_f64()).filter(|w| *w > 0.0);
                    for dim in 0..size {
                        let diff = q64[dim] - dataset.kdtree_get_pt(b_idx, dim).to_f64();
                        result += diff * diff;
                        if let Some(w) = check {
                            if result > w {
                                return F::from_f64(result);
                            }
                        }
                    }
                    return F::from_f64(result);
                }
            }
        }

        // Scalar fallback (used for all other Real types and when simd feature is disabled)
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

impl<F: Real + 'static, D: KdTreeDataset<F>> DistanceMetric<F, D> for L2Simple {
    #[inline]
    fn eval_metric(
        &self,
        dataset: &D,
        query: &[F],
        b_idx: usize,
        size: usize,
        _worst_dist: Option<F>,
    ) -> F {
        // SIMD fast path for f32/f64 when the feature is enabled
        #[cfg(feature = "simd")]
        {
            use std::any::TypeId;

            if TypeId::of::<F>() == TypeId::of::<f32>() {
                let q32 = unsafe { &*(query as *const [F] as *const [f32]) };
                if let Some(pt) = dataset.kdtree_get_point(b_idx) {
                    let p32 = unsafe { &*(pt as *const [F] as *const [f32]) };
                    let dist = simd::l2_simple_squared_f32(q32, p32);
                    return F::from_f64(dist as f64);
                } else {
                    let mut result = 0.0f32;
                    for dim in 0..size {
                        let diff = q32[dim] - dataset.kdtree_get_pt(b_idx, dim).to_f64() as f32;
                        result += diff * diff;
                    }
                    return F::from_f64(result as f64);
                }
            }

            if TypeId::of::<F>() == TypeId::of::<f64>() {
                let q64 = unsafe { &*(query as *const [F] as *const [f64]) };
                if let Some(pt) = dataset.kdtree_get_point(b_idx) {
                    let p64 = unsafe { &*(pt as *const [F] as *const [f64]) };
                    let dist = simd::l2_simple_squared_f64(q64, p64);
                    return F::from_f64(dist);
                } else {
                    let mut result = 0.0f64;
                    for dim in 0..size {
                        let diff = q64[dim] - dataset.kdtree_get_pt(b_idx, dim).to_f64();
                        result += diff * diff;
                    }
                    return F::from_f64(result);
                }
            }
        }

        // Scalar fallback
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

impl<F: Real + 'static, D: KdTreeDataset<F>> DistanceMetric<F, D> for SO3 {
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

// =============================================================================
// SIMD-accelerated specializations (enabled with `cargo +nightly ... --features simd`)
// =============================================================================

#[cfg(feature = "simd")]
mod simd {
    use std::simd::prelude::*;

    /// SIMD L2 (squared Euclidean) with early abort support for f32.
    #[inline]
    pub fn l2_squared_f32(
        query: &[f32],
        point: &[f32],
        worst_dist: Option<f32>,
    ) -> f32 {
        let len = query.len().min(point.len());
        let mut sum = f32x8::splat(0.0);
        let mut i = 0;

        let check = worst_dist.filter(|w| *w > 0.0);

        // Process 8 elements at a time
        while i + 8 <= len {
            let q = f32x8::from_slice(&query[i..i + 8]);
            let p = f32x8::from_slice(&point[i..i + 8]);
            let diff = q - p;
            sum += diff * diff;

            if let Some(w) = check {
                let partial = sum.reduce_sum();
                if partial > w {
                    return partial;
                }
            }
            i += 8;
        }

        let mut result = sum.reduce_sum();

        // Tail handling + early exit
        for j in i..len {
            let diff = query[j] - point[j];
            result += diff * diff;
            if let Some(w) = check {
                if result > w {
                    return result;
                }
            }
        }
        result
    }

    /// SIMD L2 (squared Euclidean) with early abort support for f64.
    #[inline]
    pub fn l2_squared_f64(
        query: &[f64],
        point: &[f64],
        worst_dist: Option<f64>,
    ) -> f64 {
        let len = query.len().min(point.len());
        let mut sum = f64x4::splat(0.0);
        let mut i = 0;

        let check = worst_dist.filter(|w| *w > 0.0);

        while i + 4 <= len {
            let q = f64x4::from_slice(&query[i..i + 4]);
            let p = f64x4::from_slice(&point[i..i + 4]);
            let diff = q - p;
            sum += diff * diff;

            if let Some(w) = check {
                let partial = sum.reduce_sum();
                if partial > w {
                    return partial;
                }
            }
            i += 4;
        }

        let mut result = sum.reduce_sum();

        for j in i..len {
            let diff = query[j] - point[j];
            result += diff * diff;
            if let Some(w) = check {
                if result > w {
                    return result;
                }
            }
        }
        result
    }

    /// SIMD L2Simple (no early abort) for f32 – slightly simpler hot path.
    #[inline]
    pub fn l2_simple_squared_f32(query: &[f32], point: &[f32]) -> f32 {
        let len = query.len().min(point.len());
        let mut sum = f32x8::splat(0.0);
        let mut i = 0;

        while i + 8 <= len {
            let q = f32x8::from_slice(&query[i..i + 8]);
            let p = f32x8::from_slice(&point[i..i + 8]);
            let diff = q - p;
            sum += diff * diff;
            i += 8;
        }

        let mut result = sum.reduce_sum();
        for j in i..len {
            let diff = query[j] - point[j];
            result += diff * diff;
        }
        result
    }

    /// SIMD L2Simple (no early abort) for f64.
    #[inline]
    pub fn l2_simple_squared_f64(query: &[f64], point: &[f64]) -> f64 {
        let len = query.len().min(point.len());
        let mut sum = f64x4::splat(0.0);
        let mut i = 0;

        while i + 4 <= len {
            let q = f64x4::from_slice(&query[i..i + 4]);
            let p = f64x4::from_slice(&point[i..i + 4]);
            let diff = q - p;
            sum += diff * diff;
            i += 4;
        }

        let mut result = sum.reduce_sum();
        for j in i..len {
            let diff = query[j] - point[j];
            result += diff * diff;
        }
        result
    }
}

