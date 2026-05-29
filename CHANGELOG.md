# Changelog

## [0.5.0] - 2026-05-29

### Added
- Exposed zero-allocation search methods `knn_search_into`, `radius_search_into`, and `rknn_search_into` for `DynamicKdTree`.
- Exposed zero-allocation search methods `radius_search_into` and `rknn_search_into` for `KdTree`.
- Added a `clear(&mut self)` method to the `ResultSet` trait (with a default no-op implementation) and overridden implementations for `KnnResultSet`, `RknnResultSet`, `RadiusResultSet`, and `SmallKnnResultSet` to allow reusing pre-allocated result set buffers across queries.
- Added a `DatasetTooLarge` error variant to `KdTreeError` returned when point counts exceed the maximum capacity of 32-bit indexing.

### Changed
- Changed internal index array type (`v_acc`) in `KdTree` from `usize` to `u32`. This reduces index memory footprint by 50% on 64-bit systems, matching the native `u32` limits of Leaf node offsets and improving cache performance.
- Increased the query-time local stack buffer threshold from `32` to `256` dimensions across KD-tree split/bounding-box computations and query distance tracking, avoiding heap allocation of `Vec` for all typical high-dimensional queries.
- Marked KD-tree getters, helpers, and critical query paths (`find_neighbors_set`, `search_level`, `is_active`) with `#[inline]` to allow better compiler optimization, cross-crate monomorphization, and loop vectorization.
