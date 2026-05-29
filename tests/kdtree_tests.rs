use nanoflann::{
    DistanceMetric, DynamicKdTree, Interval, KdTree, KdTreeParams, L2Simple, MatrixDataset,
    MatrixLayout, PointCloud, SearchParameters, L1, L2, SO2,
};
use std::cell::RefCell;
use std::rc::Rc;

const EPS: f64 = 1e-9;

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() <= EPS, "left={a}, right={b}");
}

fn sample_cloud() -> PointCloud<f64> {
    PointCloud::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 2.0],
        vec![2.0, 2.0],
        vec![-1.0, 1.0],
        vec![3.0, -1.0],
        vec![1.0, 1.0],
    ])
    .unwrap()
}

fn brute_l2(points: &[Vec<f64>], query: &[f64]) -> Vec<(usize, f64)> {
    let mut out: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            let dist: f64 = point
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (idx, dist)
        })
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out
}

#[test]
fn distance_metrics_match_nanoflann_semantics() {
    let cloud = PointCloud::new(vec![vec![2.0, -1.0, 4.0]]).unwrap();
    let query = [1.0, 2.0, 3.0];

    let l1 = L1.eval_metric(&cloud, &query, 0, 3, None);
    let l2 = L2.eval_metric(&cloud, &query, 0, 3, None);
    let l2_simple = L2Simple.eval_metric(&cloud, &query, 0, 3, None);

    assert_close(l1, 5.0);
    assert_close(l2, 11.0);
    assert_close(l2_simple, 11.0);
}

#[test]
fn so2_wraps_angular_difference_like_cpp() {
    let cloud = PointCloud::new(vec![vec![-3.0]]).unwrap();
    let query = [3.0];
    let distance = SO2.eval_metric(&cloud, &query, 0, 1, None);
    assert_close(distance, -3.0 - 3.0 + 2.0 * std::f64::consts::PI);
}

#[test]
fn knn_search_matches_brute_force_l2() {
    let cloud = sample_cloud();
    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default()).unwrap();
    let query = [0.9, 0.2];

    let got = tree.knn_search(&query, 3).unwrap();
    let expected = brute_l2(cloud.points(), &query);

    assert_eq!(got.len(), 3);
    for (result, expected) in got.iter().zip(expected.iter().take(3)) {
        assert_eq!(result.index, expected.0);
        assert_close(result.distance, expected.1);
    }
}

#[test]
fn radius_search_uses_strict_radius_and_sorted_results() {
    let cloud = sample_cloud();
    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default()).unwrap();

    let strict = tree.radius_search(&[0.0, 0.0], 2.0).unwrap();
    let strict_indices: Vec<_> = strict.iter().map(|item| item.index).collect();
    assert_eq!(strict_indices, vec![0, 1]);

    let wider = tree
        .radius_search_with_params(
            &[0.0, 0.0],
            2.01,
            SearchParameters {
                eps: 0.0,
                sorted: true,
            },
        )
        .unwrap();
    let mut wider_indices: Vec<_> = wider.iter().map(|item| item.index).collect();
    wider_indices.sort_unstable();
    assert_eq!(wider_indices, vec![0, 1, 4, 6]);
    assert!(wider.windows(2).all(|w| w[0].distance <= w[1].distance));
}

#[test]
fn rknn_search_applies_radius_and_capacity() {
    let cloud = sample_cloud();
    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default()).unwrap();

    let got = tree.rknn_search(&[0.9, 0.2], 5, 1.0).unwrap();
    let got_indices: Vec<_> = got.iter().map(|item| item.index).collect();
    assert_eq!(got_indices, vec![1, 6, 0]);
}

#[test]
fn find_within_box_is_inclusive() {
    let cloud = sample_cloud();
    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default()).unwrap();
    let mut got = tree
        .find_within_box(&[Interval::new(0.0, 1.0), Interval::new(0.0, 1.0)])
        .unwrap();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 6]);
}

#[test]
fn skip_initial_build_reports_unbuilt_until_build_index() {
    let cloud = sample_cloud();
    let params = KdTreeParams {
        skip_initial_build: true,
        ..Default::default()
    };
    let mut tree = KdTree::new(2, &cloud, L2, params).unwrap();

    assert!(tree.knn_search(&[0.0, 0.0], 1).is_err());
    tree.build_index().unwrap();
    let got = tree.knn_search(&[0.0, 0.0], 1).unwrap();
    assert_eq!(got[0].index, 0);
}

#[test]
fn save_and_load_round_trip_preserves_queries() {
    let cloud = sample_cloud();
    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default()).unwrap();
    let expected = tree.knn_search(&[0.9, 0.2], 4).unwrap();

    let mut bytes = Vec::new();
    tree.save_index(&mut bytes).unwrap();

    let params = KdTreeParams {
        skip_initial_build: true,
        ..Default::default()
    };
    let mut loaded = KdTree::new(2, &cloud, L2, params).unwrap();
    let mut slice = bytes.as_slice();
    loaded.load_index(&mut slice).unwrap();

    let got = loaded.knn_search(&[0.9, 0.2], 4).unwrap();
    assert_eq!(got.len(), expected.len());
    for (got, expected) in got.iter().zip(expected.iter()) {
        assert_eq!(got.index, expected.index);
        assert_close(got.distance, expected.distance);
    }
}

#[test]
fn matrix_dataset_supports_row_and_column_points() {
    // Matrix: [[0, 0], [1, 0], [0, 2]] stored row-major.
    let row_points = MatrixDataset::new(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
        3,
        2,
        MatrixLayout::RowMajorPoints,
    )
    .unwrap();
    let row_tree = KdTree::new(
        row_points.point_dim(),
        &row_points,
        L2,
        KdTreeParams::default(),
    )
    .unwrap();
    assert_eq!(row_tree.knn_search(&[0.8, 0.1], 1).unwrap()[0].index, 1);

    // Same logical points as columns in matrix [[0, 1, 0], [0, 0, 2]].
    let col_points = MatrixDataset::new(
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        2,
        3,
        MatrixLayout::ColumnMajorPoints,
    )
    .unwrap();
    let col_tree = KdTree::new(
        col_points.point_dim(),
        &col_points,
        L2,
        KdTreeParams::default(),
    )
    .unwrap();
    assert_eq!(col_tree.knn_search(&[0.8, 0.1], 1).unwrap()[0].index, 1);
}

#[test]
fn dynamic_tree_handles_insert_remove_and_reactivation_without_duplicates() {
    let cloud = Rc::new(RefCell::new(PointCloud::empty(2)));
    let mut dynamic = DynamicKdTree::new(2, &cloud, L2, KdTreeParams::default(), 16).unwrap();

    cloud.borrow_mut().push(vec![0.0, 0.0]).unwrap();
    dynamic.add_points(0, 0).unwrap();
    cloud.borrow_mut().push(vec![10.0, 0.0]).unwrap();
    dynamic.add_points(1, 1).unwrap();
    cloud.borrow_mut().push(vec![1.0, 0.0]).unwrap();
    dynamic.add_points(2, 2).unwrap();

    assert_eq!(dynamic.knn_search(&[0.9, 0.0], 1).unwrap()[0].index, 2);

    dynamic.remove_point(2);
    assert_eq!(dynamic.knn_search(&[0.9, 0.0], 1).unwrap()[0].index, 0);

    dynamic.add_points(2, 2).unwrap();
    assert_eq!(dynamic.knn_search(&[0.9, 0.0], 1).unwrap()[0].index, 2);

    let near_reactivated = dynamic.radius_search(&[1.0, 0.0], 0.001).unwrap();
    assert_eq!(near_reactivated.len(), 1);
    assert_eq!(near_reactivated[0].index, 2);
}
