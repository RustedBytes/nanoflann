use nanoflann::{
    KdTree, KdTreeParams, MatrixDataset, MatrixLayout, SearchParameters, SmallKnnResultSet, L2,
};

#[no_mangle]
#[inline(never)]
pub fn asm_query_knn_f64_l2(tree: &KdTree<f64, MatrixDataset<f64>, L2>, q: &[f64; 3]) -> usize {
    let mut result: SmallKnnResultSet<f64, 8> = SmallKnnResultSet::new();
    tree.knn_search_into(
        std::hint::black_box(q),
        &mut result,
        SearchParameters::default(),
    )
    .unwrap();
    std::hint::black_box(result.as_slice()[0].index)
}

fn main() {
    let n = 4096;
    let mut data = Vec::with_capacity(n * 3);

    for i in 0..n {
        let x = i as f64;
        data.push(x.sin());
        data.push(x.cos());
        data.push((x * 0.01).sin());
    }

    let dataset = MatrixDataset::new(data, n, 3, MatrixLayout::RowMajorPoints).unwrap();
    let tree = KdTree::new(3, &dataset, L2, KdTreeParams::default()).unwrap();

    let q = [0.1, 0.2, 0.3];
    let idx = asm_query_knn_f64_l2(&tree, &q);
    println!("{idx}");
}
