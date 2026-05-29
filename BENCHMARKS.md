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