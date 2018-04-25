
use super::*;



/// A vector slice.
/// Holds a pointer to possibly non-continuous GPU memory.
pub struct CuSubMatrix<'a> {
    pub(crate) parent: PhantomData<&'a CuMatrixOp>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) leading_dimension: usize,
    pub(crate) ptr: *const f32,
}
impl_CuMatrixOp_fragmented!(CuSubMatrix<'a>, 'a);


/// A mutable vector slice.
/// Holds a pointer to possibly non-continuous GPU memory.
pub struct CuSubMatrixMut<'a> {
    pub(crate) parent: PhantomData<&'a CuMatrixOpMut>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) leading_dimension: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuMatrixOp_fragmented!(CuSubMatrixMut<'a>, 'a);
impl_CuMatrixOpMut_fragmented!(CuSubMatrixMut<'a>, 'a);




#[cfg(test)]
mod tests {
    use super::{CuMatrix, CuMatrixOp, CuMatrixOpMut};
    use cuda::*;

    #[test]
    fn getters() {
        let initial_rows = 4;
        let initial_cols = 8;
        let mut matrix = CuMatrix::new(initial_rows, initial_cols, 0.0);

        {
            let slice = matrix.slice(2, 1, 2, 7);
            assert_eq!(slice.rows(), 2);
            assert_eq!(slice.cols(), 7);
            assert_eq!(slice.len(), 14);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }

        {
            let slice = matrix.slice_mut(1, 3, 2, 2);
            assert_eq!(slice.rows(), 2);
            assert_eq!(slice.cols(), 2);
            assert_eq!(slice.len(), 4);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }

        {
            let slice = matrix.slice_col(1, 7);
            assert_eq!(slice.rows(), initial_rows);
            assert_eq!(slice.cols(), 7);
            assert_eq!(slice.len(), initial_rows*7);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }

        {
            let slice = matrix.slice_col_mut(3, 2);
            assert_eq!(slice.rows(), initial_rows);
            assert_eq!(slice.cols(), 2);
            assert_eq!(slice.len(), initial_rows*2);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }
    }

    #[test]
    fn init() {
        let value = -1.254;
        let mut matrix = CuMatrix::new(2, 3, 0.0);
        matrix.slice_mut(0, 1, 1, 2).init(value, &DEFAULT_STREAM);

        let output = &mut[0.0; 6];
        matrix.clone_to_host(output);

        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], value);
        assert_eq!(output[3], 0.0);
        assert_eq!(output[4], value);
        assert_eq!(output[5], 0.0);
    }

}