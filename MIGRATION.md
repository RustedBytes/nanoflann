# Migration report: `nanoflann.hpp` to Rust

## Original functionality

The uploaded C++ header is a header-only KD-tree library with:

- Dataset adaptor methods: `kdtree_get_point_count()`, `kdtree_get_pt(idx, dim)`, and optional `kdtree_get_bbox()`.
- Result set classes for KNN, radius, and radius-limited KNN queries.
- Metric adaptors for L1, squared L2, L2-simple, SO2, and SO3.
- A static `KDTreeSingleIndexAdaptor` with build, KNN, radius, radius-limited KNN, and bounding-box search.
- A dynamic KD-tree that stores points across logarithmically sized static subtrees, uses lazy deletion, and reactivates removed points instead of inserting duplicates.
- An Eigen matrix adaptor and binary index save/load helpers.

## Rust design decisions

- C++ template adaptors became the trait `KdTreeDataset<F>`.
- Distance functors became the trait `DistanceMetric<F, D>` plus zero-sized metric structs: `L1`, `L2`, `L2Simple`, `SO2`, and `SO3`.
- Raw output pointer pairs became owned `Vec<ResultItem<F>>` return values.
- Exceptions became `Result<T, KdTreeError>`.
- The C++ node union and pooled allocator became a safe recursive Rust enum: `Node::Leaf` and `Node::Split` inside `Box`.
- The `NANOFLANN_FIRST_MATCH` macro became a runtime parameter: `KdTreeParams::first_match`.
- `KDTreeEigenMatrixAdaptor` became `MatrixDataset`, which interprets a flat row-major matrix as either row-points or column-points.

## Important semantic mappings

| C++ | Rust |
| --- | --- |
| `KDTreeSingleIndexAdaptor` | `KdTree<'a, F, D, M>` |
| `KDTreeSingleIndexDynamicAdaptor` | `DynamicKdTree<'a, F, D, M>` |
| `ResultItem<IndexType, DistanceType>` | `ResultItem<F> { index: usize, distance: F }` |
| `KDTreeSingleIndexAdaptorParams` | `KdTreeParams` |
| `SearchParameters` | `SearchParameters<F>` |
| `Interval { low, high }` | `Interval<F> { low, high }` |
| `metric_L2` / `L2_Adaptor` | `L2`, returning squared distance |
| `radiusSearch(radius)` | `radius_search(radius)`, strict `distance < radius` |
| `findWithinBox()` | `find_within_box()`, inclusive bounds |

## Numerical and indexing behavior

- L2 and SO3 distances are squared.
- Radius arguments for L2 are therefore squared radii, as in nanoflann.
- Radius membership is strict: `distance < radius`.
- Box membership is inclusive.
- Query slices may contain more dimensions than the tree; only the first `dim` values are used, matching pointer-style C++ semantics.
- `SO2` preserves the original signed wrapped difference behavior and assumes input angles are already in `[-pi, pi]`.

## Safety and ownership changes

- No `unsafe` is used.
- The pooled allocator was intentionally removed. Rust ownership frees nodes automatically when the tree is dropped or rebuilt.
- Output parameters were replaced with return values.
- Dynamic insertion is checked: new points must be inserted contiguously unless reactivating a removed point. This avoids silent corruption from non-sequential point IDs.
- `leaf_max_size == 0` is rejected because it can lead to invalid splits or nontermination in the original recursive algorithm.

## Save/load behavior

The Rust `save_index()` and `load_index()` methods use a portable little-endian semantic format instead of writing raw struct bytes and pointer values. As in the C++ header, the dataset itself is not serialized, so loading must be done into a tree attached to the same point data.

## Testing strategy

Tests cover:

- L1, L2, L2-simple, and SO2 metric semantics.
- KNN results against brute-force squared L2 search.
- Strict radius behavior and sorted radius output.
- Radius-limited KNN.
- Inclusive bounding-box search.
- Delayed build via `skip_initial_build`.
- Save/load round trip preserving query results.
- Row-point and column-point matrix adaptors.
- Dynamic insertion, lazy removal, and reactivation without duplicates.

## Performance considerations

- Tree traversal and split logic follow the nanoflann algorithm closely.
- Nodes use `Box<Node<F>>`; this is safe and simple, though not as allocation-dense as the C++ pooled allocator.
- The original C++ unrolled distance loops were simplified into idiomatic Rust loops; LLVM should optimize common small dimensions well, but SIMD/pool allocation could be added later if benchmarks require it.
- Multithreaded build parameters are retained for API familiarity but the current build is single-threaded.

## Known limitations and assumptions

- The scalar type is limited to `f32` and `f64` via the crate's `Real` trait. The C++ header can be instantiated with more numeric types.
- Custom result-set callbacks are not exposed as a public stable API; the provided wrappers cover KNN, RKNN, radius, and box search.
- The binary index format is not compatible with nanoflann's raw C++ save files. It is intentionally portable and pointer-free.
- Dynamic additions are designed for datasets that already contain or can expose newly appended points. For mutation while indexed, use interior mutability such as `Rc<RefCell<PointCloud<_>>>`, as shown in the tests.
