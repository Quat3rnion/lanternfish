//! Locate-then-edit linear algebra. Vectors are columns; a list of vectors
//! stores a matrix in column-major order. Use F64 for the small solves.

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(x: &mut [f64], y: &[f64], s: f64) {
    for (x, y) in x.iter_mut().zip(y) {
        *x += y * s;
    }
}

/// Each context group contributes its mean with equal weight.
pub(super) fn mean_key(keys: &[Vec<f32>], groups: &[usize]) -> Vec<f64> {
    debug_assert_eq!(groups.iter().sum::<usize>(), keys.len());
    let mut mean = vec![0.0; keys[0].len()];
    let mut start = 0;
    for &count in groups {
        let weight = 1.0 / (groups.len() * count) as f64;
        for key in &keys[start..start + count] {
            for (out, &value) in mean.iter_mut().zip(key) {
                *out += value as f64 * weight;
            }
        }
        start += count;
    }
    mean
}

struct Gram {
    lower: Vec<Vec<f64>>,
}

impl Gram {
    // Cholesky of I + X^T X / ridge, which is strictly positive definite.
    fn new(x: &[Vec<f64>], ridge: f64) -> Self {
        let mut lower: Vec<Vec<f64>> = Vec::with_capacity(x.len());
        for (i, a) in x.iter().enumerate() {
            let mut row = Vec::with_capacity(i + 1);
            for (j, b) in x[..=i].iter().enumerate() {
                let g = dot(a, b) / ridge;
                if i == j {
                    row.push((1.0 + g - dot(&row, &row)).sqrt());
                } else {
                    row.push((g - dot(&row, &lower[j])) / lower[j][j]);
                }
            }
            lower.push(row);
        }
        Self { lower }
    }

    fn solve(&self, mut b: Vec<f64>) -> Vec<f64> {
        for (i, row) in self.lower.iter().enumerate() {
            b[i] = (b[i] - dot(row, &b[..i])) / row[i];
        }
        for (i, row) in self.lower.iter().enumerate().rev() {
            b[i] /= row[i];
            for j in 0..i {
                b[j] -= row[j] * b[i];
            }
        }
        b
    }
}

// (ridge I + XX^T)^-1 k using Woodbury, without a width x width matrix.
fn whiten(keys: &[Vec<f64>], x: &[Vec<f64>], ridge: f64) -> Vec<Vec<f64>> {
    let gram = Gram::new(x, ridge);
    keys.iter()
        .map(|k| {
            let mut out = k.clone();
            let coefficients = gram.solve(x.iter().map(|v| dot(v, k) / ridge).collect());
            for (v, s) in x.iter().zip(coefficients) {
                axpy(&mut out, v, -s);
            }
            out.iter_mut().for_each(|v| *v /= ridge);
            out
        })
        .collect()
}

pub(super) fn rome(
    averaged: Vec<f64>,
    lookup: &[f64],
    moment: &[Vec<f64>],
    ridge: f64,
) -> Vec<Vec<f64>> {
    let mut out = whiten(&[averaged], moment, ridge);
    let denom = dot(&out[0], lookup);
    out[0].iter_mut().for_each(|v| *v /= denom);
    out
}

pub(super) fn memit(
    keys: &[Vec<f64>],
    moment: &[Vec<f64>],
    weight: f64,
    ridge: f64,
) -> Vec<Vec<f64>> {
    let scale = weight.sqrt();
    let x: Vec<Vec<f64>> = moment
        .iter()
        .map(|v| v.iter().map(|t| t * scale).collect())
        .chain(keys.iter().cloned())
        .collect();
    whiten(keys, &x, ridge)
}

pub(super) fn alphaedit(keys: &[Vec<f64>], preserved: &[Vec<f64>], ridge: f64) -> Vec<Vec<f64>> {
    // The range of the second moment equals the span of its sampled keys.
    // Two-pass modified Gram-Schmidt builds its numerical orthonormal basis.
    fn orthogonalize(v: &mut [f64], basis: &[Vec<f64>]) {
        for _ in 0..2 {
            for b in basis {
                let s = dot(v, b);
                axpy(v, b, -s);
            }
        }
    }
    let mut basis = Vec::new();
    for row in preserved {
        let mut v = row.clone();
        orthogonalize(&mut v, &basis);
        let len = dot(&v, &v).sqrt();
        if len > 1e-6 * dot(row, row).sqrt() {
            v.iter_mut().for_each(|x| *x /= len);
            basis.push(v);
        }
    }
    let projected: Vec<_> = keys
        .iter()
        .cloned()
        .map(|mut k| {
            orthogonalize(&mut k, &basis);
            k
        })
        .collect();
    // Solve within range(P). First batch has no previous edit history term.
    whiten(&projected, &projected, ridge)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} differs from {b}");
    }

    #[test]
    fn rome_maps_canonical_key_to_target() {
        // C = diag(4,1). Whiten mean [2,1] to [.5,1], then normalize by
        // dot against the unaveraged key [1,2].
        let k = [1.0, 2.0];
        let a = rome(vec![2.0, 1.0], &k, &[vec![3.0f64.sqrt(), 0.0]], 1.0);
        near(a[0][0], 0.2);
        near(a[0][1], 0.4);
        let wk = [-1.0, 3.0];
        let target = [5.0, 7.0];
        for (value, y) in wk.into_iter().zip(target) {
            near(value + (y - value) * dot(&a[0], &k), y);
        }
    }

    #[test]
    fn memit_matches_dense_batch_normal_equation() {
        // I + C + KK^T = [[4,1],[1,2]] with K=[[1,1],[0,1]].
        let a = memit(
            &[vec![1.0, 0.0], vec![1.0, 1.0]],
            &[vec![1.0, 0.0]],
            1.0,
            1.0,
        );
        near(a[0][0], 2.0 / 7.0);
        near(a[0][1], -1.0 / 7.0);
        near(a[1][0], 1.0 / 7.0);
        near(a[1][1], 3.0 / 7.0);
    }

    #[test]
    fn batch_key_means_match_reference_context_group_weighting() {
        let facts = [
            [
                vec![8.0, 4.0],
                vec![1.0, 0.0],
                vec![2.0, 2.0],
                vec![3.0, 4.0],
            ],
            [
                vec![0.0, 3.0],
                vec![3.0, 1.0],
                vec![6.0, 3.0],
                vec![9.0, 5.0],
            ],
        ];
        // The reference gives the canonical group and the three-template
        // group equal weight, not the four templates equal weight.
        let means: Vec<_> = facts.iter().map(|keys| mean_key(keys, &[1, 3])).collect();
        assert_eq!(means, [vec![5.0, 3.0], vec![3.0, 3.0]]);
    }

    #[test]
    fn alphaedit_leaves_preserved_outputs_unchanged() {
        let old = [3.0, 5.0, 7.0];
        let preserve = [1.0, 2.0, 0.0];
        let edit = [1.0, 2.0, 2.0];
        let a = alphaedit(
            &[edit.to_vec()],
            &[preserve.to_vec(), vec![2.0, 4.0, 0.0]],
            1.0,
        );
        let new: Vec<_> = old.iter().zip(&a[0]).map(|(w, a)| w + 5.0 * a).collect();
        near(dot(&new, &preserve), 13.0);
        // P k = [0,0,2] and (I+Pkk^T P)^-1 Pk=[0,0,2/5].
        near(dot(&new, &edit), dot(&old, &edit) + 4.0);
    }
}
