

use ffi::cuda_ffi::*;



pub struct Cuda {

}


impl Cuda {
    pub fn synchronize() {
        cuda_synchronize();
    }
}