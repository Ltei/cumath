
use super::*;
use super::ffi::*;
use std::ptr;
use std::os::raw::c_void;
use std::fmt::{self, Debug};
use CuDataType;






pub struct CuActivationDescriptor {
    data: *mut _ActivationDescriptorStruct,
}

impl Drop for CuActivationDescriptor {
    fn drop(&mut self) {
        cudnn_destroy_activation_descriptor(self.data)
    }
}

impl CuActivationDescriptor {

    pub fn new(mode: CudnnActivationMode, coef: f64) -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, mode, CudnnNanPropagation::Propagate, coef);
        CuActivationDescriptor { data }
    }
    pub fn sigmoid() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Sigmoid, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }
    pub fn relu() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Relu, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }
    pub fn tanh() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Tanh, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }
    pub fn clipped_relu(threshold: f64) -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::ClippedRelu, CudnnNanPropagation::Propagate, threshold);
        CuActivationDescriptor { data }
    }
    pub fn elu(alpha: f64) -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Elu, CudnnNanPropagation::Propagate, alpha);
        CuActivationDescriptor { data }
    }
    /*pub fn identity() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Identity, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }*/

    pub fn get_info(&self) -> CuActivationDescriptorInfo {
        let mut mode = CudnnActivationMode::Elu;
        let mut relu_nan_opt = CudnnNanPropagation::NotPropagate;
        let mut coef = -9999.0;
        cudnn_get_activation_descriptor(
            self.data,
            &mut mode,
            &mut relu_nan_opt,
            &mut coef,
        );
        CuActivationDescriptorInfo { mode, relu_nan_opt, coef }
    }

    pub fn forward<T: CuDataType>(&self, cudnn: &Cudnn, input: &CuTensorDeref<T>, input_scale: T, output: &mut CuTensorDeref<T>, output_scale: T) {
        cudnn_activation_forward(cudnn.handle, self.data,
                                 &input_scale as *const T as *const c_void, input.descriptor.data, input.data as *const c_void,
                                 &output_scale as *const T as *const c_void, output.descriptor.data, output.data as *mut c_void)
    }
    pub fn forward_inplace<T: CuDataType>(&self, cudnn: &Cudnn, vector: &mut CuTensorDeref<T>, input_scale: T, output_scale: T) {
        cudnn_activation_forward(cudnn.handle, self.data,
                                 &input_scale as *const T as *const c_void, vector.descriptor.data, vector.data as *const c_void,
                                 &output_scale as *const T as *const c_void, vector.descriptor.data, vector.data as *mut c_void)
    }
    pub fn backward<T: CuDataType>(&self, cudnn: &Cudnn, alpha: T, beta: T,
                                   input: &CuTensorDeref<T>, d_input: &CuTensorDeref<T>,
                                   output: &CuTensorDeref<T>, d_output: &mut CuTensorDeref<T>) {
        cudnn_activation_backward(cudnn.handle, self.data,
                                  (&alpha) as *const T as *const c_void, input.descriptor.data, input.data as *const c_void,
                                  d_input.descriptor.data, d_input.data as *const c_void, output.descriptor.data, output.data as *mut c_void,
                                  (&beta) as *const T as *const c_void, d_output.descriptor.data, d_output.data as *mut c_void)
    }
    pub fn backward_inplace<T: CuDataType>(&self, cudnn: &Cudnn, alpha: T, beta: T,
                                           input: &CuTensorDeref<T>,
                                           output: &CuTensorDeref<T>,
                                           signal: &mut CuTensorDeref<T>) {
        cudnn_activation_backward(cudnn.handle, self.data,
                                  (&alpha) as *const T as *const c_void, input.descriptor.data, input.data as *const c_void,
                                  signal.descriptor.data, signal.data as *const c_void, output.descriptor.data, output.data as *mut c_void,
                                  (&beta) as *const T as *const c_void, signal.descriptor.data, signal.data as *mut c_void)
    }

}


pub struct CuActivationDescriptorInfo {
    pub mode: CudnnActivationMode,
    pub relu_nan_opt: CudnnNanPropagation,
    pub coef: f64,
}

impl Debug for CuActivationDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mode:{:?}, ReluNanOpt:{:?}, coef:{}", self.mode, self.relu_nan_opt, self.coef)
    }
}



#[cfg(test)]
mod tests {

    use super::*;
    use nn::*;
    use ::*;

    fn test_forward(name: &str, activation: CuActivationDescriptor) {
        let cudnn = Cudnn::new();

        let tensor_descriptor = CuTensorDescriptor::<f32>::new(&[2, 2, 1]);
        let mut input = CuVector::<f32>::from_host_data(&[-0.75, -0.5, 0.0, 1.0]);
        let mut output = CuVector::<f32>::zero(tensor_descriptor.data_len());
        let mut d_input = CuVector::<f32>::new(1.0, tensor_descriptor.data_len());
        let mut d_output = CuVector::<f32>::zero(tensor_descriptor.data_len());
        let mut gradient = CuVector::<f32>::zero(tensor_descriptor.data_len());

        activation.forward(&cudnn, &tensor_descriptor.link(&input), 1.0, &mut tensor_descriptor.link_mut(&mut output), 0.0);

        println!("{} 1 : Input[{:?}] Output[{:?}]", name, input, output);

        activation.backward(&cudnn, 1.0, 0.0,
                            &tensor_descriptor.link(&input),
                            &tensor_descriptor.link(&d_input),
                            &tensor_descriptor.link(&output),
                            &mut tensor_descriptor.link_mut(&mut d_output));

        println!("{} 2 : Input[{:?}] Output[{:?}] DInput[{:?}] DOutput[{:?}]", name, input, output, d_input, d_output);

        activation.forward_inplace(&cudnn, &mut tensor_descriptor.link_mut(&mut input), 1.0, 0.0);
    }

    #[test]
    fn sigmoid_forward() {
        test_forward("sigmoid", CuActivationDescriptor::sigmoid());
    }

    #[test]
    fn relu_forward() {
        test_forward("relu", CuActivationDescriptor::relu());
    }

    #[test]
    fn tanh_forward() {
        test_forward("tanh", CuActivationDescriptor::tanh());
    }

    #[test]
    fn clipped_relu_forward() {
        test_forward("clippedRelu", CuActivationDescriptor::clipped_relu(0.5));
    }

    #[test]
    fn elu_forward() {
        test_forward("elu", CuActivationDescriptor::elu(0.5));
    }

    /*#[test]
    fn identity_forward() {
        test_forward("identity", CuActivationDescriptor::identity());
    }*/

    #[test]
    #[ignore]
    fn sigmoid_forward_benchmarch() {
        use std::time::Instant;

        let mut vector = CuVector::<f32>::zero(500);

        let t0 = Instant::now();
        for i in 0..10000 {
            vector.fast_sigmoid(&DEFAULT_STREAM)
        }
        let dt = t0.elapsed();
        println!("Finished in {}.{}", dt.as_secs(), dt.subsec_nanos());


        let cudnn = Cudnn::new();
        let activation = CuActivationDescriptor::sigmoid();
        let tensor_descriptor = CuTensorDescriptor::<f32>::new(&[2, 3, 4, 5]);
        let mut vector = CuVector::<f32>::zero(tensor_descriptor.data_len());
        println!("Input = {:?}", vector);
        let t0 = Instant::now();
        for i in 0..10000 {
            activation.forward_inplace(&cudnn, &mut tensor_descriptor.link_mut(&mut vector), 1.0, 0.0);
        }
        let dt = t0.elapsed();
        println!("Finished in {}.{}", dt.as_secs(), dt.subsec_nanos());
    }

}