

use std::{ptr, mem::size_of};
use super::*;



/// A GPU-allocated vector.
/// Holds a pointer to continuous GPU memory.
pub struct CuVector {
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl Drop for CuVector {
    fn drop(&mut self) { cuda_free(self.ptr); }
}
impl_CuPackedDataMut!(CuVector);
impl_CuVectorOpMut!(CuVector);


impl CuVector {

    /// Returns a new GPU-allocated vector from a length and an initial value.
    pub fn new(len: usize, init_value: f32) -> CuVector {
        let mut data = ptr::null_mut();
        cuda_malloc(&mut data, len*size_of::<f32>());
        unsafe { VectorKernel_init(data as *mut f32, len as i32, init_value, DEFAULT_STREAM.stream) }
        CuVector { len, ptr: (data as *mut f32) }
    }

    /// Returns a new GPU-allocated copy of 'data'.
    pub fn from_data(data: &[f32]) -> CuVector {
        let mut output = {
            let len = data.len();
            let mut data = ptr::null_mut();
            cuda_malloc(&mut data, len*size_of::<f32>());
            CuVector { len, ptr: (data as *mut f32) }
        };
        output.clone_from_host(data);
        output
    }

}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init() {
        let value0 = -0.23254;
        let value1 = 1.1852;
        let mut vector = super::CuVector::new(5, 0.0);

        vector.init(value0, &DEFAULT_STREAM);
        vector.slice_mut(1, 3).init(value1, &DEFAULT_STREAM);

        let mut output = vec![0.0; 5];
        vector.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value1);
        assert_equals_float(output[2], value1);
        assert_equals_float(output[3], value1);
        assert_equals_float(output[4], value0);
    }
    #[test]
    fn add_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut vector0 = super::CuVector::new(5, 0.0);
        let mut vector1 = super::CuVector::new(2, 0.0);

        vector0.init(value0, &DEFAULT_STREAM);
        vector1.init(value1, &DEFAULT_STREAM);
        vector0.slice_mut(2, 2).add(&vector1, &DEFAULT_STREAM);

        let mut output = vec![0.0; 5];
        vector0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0+value1);
        assert_equals_float(output[4], value0);
    }
}
