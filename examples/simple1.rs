use nanoflann_rs::{KdTree, KdTreeParams, PointCloud, L2};

fn main() -> nanoflann_rs::Result<()> {
    let cloud = PointCloud::new(vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 2.0]])?;

    let tree = KdTree::new(2, &cloud, L2, KdTreeParams::default())?;
    let nearest = tree.knn_search(&[0.9, 0.1], 1)?;
    assert_eq!(nearest[0].index, 1);
    Ok(())
}
