use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nanoflann::{KdTree, KdTreeParams, L2, PointCloud};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate a reproducible random point cloud.
fn generate_random_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f64>()).collect())
        .collect()
}

/// Generate a random query point.
fn generate_random_query(dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.gen::<f64>()).collect()
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    // Representative sizes and dimensions for KD-tree workloads
    let sizes = [1_000usize, 10_000, 50_000, 100_000];
    let dims = [2usize, 3, 10];

    for &size in &sizes {
        for &dim in &dims {
            let points = generate_random_points(size, dim, 42);
            let cloud = PointCloud::new(points).expect("valid point cloud");

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::new("KdTree", format!("{size}x{dim}D")),
                &(size, dim),
                |b, &(_size, dim)| {
                    b.iter(|| {
                        let params = KdTreeParams::default();
                        let tree = KdTree::new(dim, &cloud, L2, params).expect("build ok");
                        black_box(tree)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");

    let sizes = [10_000usize, 100_000];
    let dims = [2usize, 3, 10];
    let ks = [1usize, 10, 100];

    for &size in &sizes {
        for &dim in &dims {
            let points = generate_random_points(size, dim, 123);
            let cloud = PointCloud::new(points).expect("valid point cloud");
            let params = KdTreeParams::default();
            let tree = KdTree::new(dim, &cloud, L2, params).expect("build ok");

            for &k in &ks {
                let query = generate_random_query(dim, 999);

                group.throughput(Throughput::Elements(1));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("k={k}"),
                        format!("{size}x{dim}D"),
                    ),
                    &query,
                    |b, q| {
                        b.iter(|| {
                            let results = tree.knn_search(black_box(q), k).expect("search ok");
                            black_box(results)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_radius_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("radius_search");

    // For unit-cube random data, these radii give interesting result set sizes
    let radii = [0.1f64, 0.3, 0.7];

    let size = 50_000usize;
    let dim = 3usize;

    let points = generate_random_points(size, dim, 777);
    let cloud = PointCloud::new(points).expect("valid point cloud");
    let params = KdTreeParams::default();
    let tree = KdTree::new(dim, &cloud, L2, params).expect("build ok");

    for &radius in &radii {
        let query = generate_random_query(dim, 555);

        // Note: radius is squared distance for L2 metric
        let squared_radius = radius * radius;

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("radius", format!("r={radius}")),
            &query,
            |b, q| {
                b.iter(|| {
                    let results = tree
                        .radius_search(black_box(q), squared_radius)
                        .expect("search ok");
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_dynamic_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_construction");

    let sizes = [1_000usize, 10_000, 50_000];
    let dim = 3usize;

    for &size in &sizes {
        let points = generate_random_points(size, dim, 2024);
        let cloud = PointCloud::new(points).expect("valid point cloud");

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("DynamicKdTree", format!("{size}x{dim}D")),
            &size,
            |b, _| {
                b.iter(|| {
                    // Dynamic tree with a generous maximum point count
                    let params = KdTreeParams::default();
                    let dtree = nanoflann::DynamicKdTree::new(
                        dim,
                        &cloud,
                        L2,
                        params,
                        size * 2,
                    )
                    .expect("dynamic build ok");
                    black_box(dtree)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_knn_search,
    bench_radius_search,
    bench_dynamic_construction
);
criterion_main!(benches);