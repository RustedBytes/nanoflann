# nanoflann

[![Crates.io Version](https://img.shields.io/crates/v/nanoflann)](https://crates.io/crates/nanoflann)

A safe Rust KD-tree crate ported from the [nanoflann.hpp](https://github.com/jlblancoc/nanoflann) project.

This is not a line-by-line syntax translation. It keeps the behaviorally important pieces of the C++ API while replacing raw pointers, pooled allocation, output buffers, and template-heavy result sets with safe Rust ownership and `Result`-based errors.

## Implemented

- Static KD-tree over an adaptor-style dataset trait.
- `PointCloud` and flat `MatrixDataset` adaptors.
- L1, squared L2, L2-simple, SO2, and SO3 metrics.
- KNN, radius search, radius-limited KNN, and inclusive bounding-box search.
- Dynamic logarithmic multi-tree wrapper with lazy deletion and reactivation.
- Portable save/load of the index structure. As in nanoflann, data points are not serialized.
- Integration tests comparing searches with brute force and checking edge semantics.

## Example

```rust
use nanoflann::{KdTree, KdTreeParams, L2, PointCloud};

fn main() -> nanoflann::Result<()> {
    let cloud = PointCloud::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 2.0],
    ])?;

    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default())?;
    let nearest = tree.knn_search(&[0.9, 0.1], 1)?;
    assert_eq!(nearest[0].index, 1);
    Ok(())
}
```

## Validation

```bash
cargo test
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

## Benchmarking

```bash
cargo bench
```

See [BENCHMARKS.md](BENCHMARKS.md) for details on measured workloads and instructions for comparing nanoflann **0.2.0** vs **0.3.0**.

## Notes

The original C++ library supports arbitrary template instantiations. This Rust port supports `f32` and `f64`, which are the common point-cloud and computer-vision cases. L2 distances and L2 radii are squared, matching nanoflann.
