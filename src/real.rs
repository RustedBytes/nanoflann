use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Floating-point scalar supported by the KD-tree.
///
/// The original C++ header is template-based and can be instantiated with many
/// arithmetic types. This Rust port intentionally supports the floating-point
/// cases most commonly used for geometry and computer-vision point clouds.
pub trait Real:
    Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn max_value() -> Self;
    fn abs(self) -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn max_value() -> Self {
        f64::MAX
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl Real for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn max_value() -> Self {
        f32::MAX
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}
