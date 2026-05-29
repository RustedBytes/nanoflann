# Benchmarking nanoflann

This crate uses [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical benchmarking.

## Running the benchmarks

```bash
cargo bench
```

Results (including HTML reports) are written to `target/criterion/`.

You can also run a specific benchmark group:

```bash
cargo bench --bench kdtree_bench construction
cargo bench --bench kdtree_bench knn_search
cargo bench --bench kdtree_bench radius_search
cargo bench --bench kdtree_bench dynamic_construction
```

## What is benchmarked

The default benchmark suite (`benches/kdtree_bench.rs`) measures:

- **construction** — Building a static `KdTree` over random point clouds (various sizes × dimensions).
- **knn_search** — `k`-nearest neighbor queries (k=1, 10, 100) on pre-built trees.
- **radius_search** — Radius search queries using the L2 (squared) metric.
- **dynamic_construction** — Construction of `DynamicKdTree` (logarithmic multi-tree structure).

All benchmarks use:
- The `L2` (squared Euclidean) metric — the most common real-world case.
- `f64` coordinates.
- Reproducible random data via seeded RNG.
- `PointCloud` dataset adaptor (the simplest and most frequently used).

## Comparing nanoflann 0.2.0 vs 0.3.0

The 0.3.0 release contains significant internal changes (new node representation, improved traversal, and multiple micro-optimizations — see commit history around "change data structure for crate" and "enhance speed").

### Recommended comparison workflow

1. **Benchmark current version (0.3.0 development)**

   ```bash
   cargo bench
   ```

   Note the numbers (or keep the `target/criterion` directory).

2. **Benchmark version 0.2.0**

   Option A — Using git (when 0.2.0 tag exists or you know the commit):

   ```bash
   git fetch --tags
   git checkout v0.2.0   # or the last commit before the 0.3 data structure changes
   cargo bench
   ```

   Option B — Using the published crate (most reliable for exact 0.2.0):

   Create a small throw-away project:

   ```bash
   mkdir /tmp/nanoflann-0.2-bench
   cd /tmp/nanoflann-0.2-bench

   cat > Cargo.toml << 'EOF'
   [package]
   name = "nanoflann-0.2-bench"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   nanoflann = "0.2"
   criterion = { version = "0.5", features = ["html_reports"] }
   rand = "0.8"

   [[bench]]
   name = "compare"
   harness = false
   EOF

   mkdir -p benches
   # Copy or adapt the benchmark code from the 0.3.0 version (adjusting for any API differences)
   # Then run:
   cargo bench
   ```

3. **Compare the results**

   - Criterion produces excellent side-by-side statistical reports.
   - Look at mean times, throughput, and the "change" detection.
   - The most interesting numbers are usually:
     - Construction time for 50k–100k points in 2D/3D
     - knn_search (k=10) latency on 100k-point trees

## Customizing the benchmark

You can extend `benches/kdtree_bench.rs` with:
- Different metrics (`L1`, `SO3`, etc.)
- `MatrixDataset` adaptor
- Larger datasets (be careful with memory)
- Approximate search (`SearchParameters { eps: 0.1, .. }`)
- `find_within_box` benchmarks

Pull requests that add meaningful new benchmark scenarios are welcome.

## Continuous benchmarking (optional)

If you want to track performance over time, consider integrating:
- `cargo-criterion`
- GitHub Actions + `bencher` or `github-action-benchmark`

The current setup already produces machine-readable JSON in `target/criterion/`.

## Experimental SIMD acceleration (`simd` feature)

Starting with version 0.3, nanoflann includes an optional **SIMD-accelerated** path for the most common distance metrics (`L2` and `L2Simple`) using Rust's `std::simd` (portable SIMD).

### Requirements

- Nightly Rust toolchain (because `portable_simd` is still unstable)
- The `simd` feature

### Enabling and running SIMD benchmarks

```bash
# Scalar (baseline)
cargo bench

# SIMD version (much faster on modern CPUs for medium+ dimensions)
cargo +nightly bench --features simd
```

When the `simd` feature is enabled, the benchmark binary also includes an extra group:

- `knn_search_simd` — same workloads as `knn_search` but compiled with vectorized distance calculations (targets 8–32 dimensions where SIMD gains are largest).

### How the optimization works

- A new optional method `kdtree_get_point(idx) -> Option<&[F]>` was added to `KdTreeDataset` (implemented efficiently for `PointCloud` and row-major `MatrixDataset`).
- Inside `L2` / `L2Simple`, when `feature = "simd"` is active and the scalar type is `f32` or `f64`, the library dispatches to hand-written SIMD kernels (`f32x8` / `f64x4`) with chunked processing + early exit for `L2`.
- Non-f32/f64 `Real` types and datasets that don't provide contiguous points fall back to the original scalar implementation.

### Expected speedups

Typical gains (highly dependent on dimension, CPU, and data layout):

- 2–3× in 8–16 dimensions
- 3–5×+ in 32+ dimensions (when using contiguous storage)

Gains are smaller in very low dimensions (2–4) due to overhead.
