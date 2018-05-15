#![allow(dead_code)]



mod cudnn;
mod tensor_descriptor;
mod activation_descriptor;
mod convolution_descriptor;
mod reduce_tensor_descriptor;
mod filter_descriptor;
mod rnn_descriptor;
mod dropout_descriptor;

pub use self::cudnn::*;
pub use self::tensor_descriptor::*;
pub use self::activation_descriptor::*;
pub use self::convolution_descriptor::*;
pub use self::reduce_tensor_descriptor::*;
pub use self::filter_descriptor::*;
pub use self::rnn_descriptor::*;
pub use self::dropout_descriptor::*;


#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnDataType {
    Float = 0,
    Double = 1,
    Half = 2,
    Int8 = 3,
    Int32 = 4,
    Int8x4 = 5,
    Uint8 = 6,
    Uint8x4 = 7,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnNanPropagation {
    NotPropagate = 0,
    Propagate = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnStatus {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 2,
    BadParam = 3,
    InternalError = 4,
    InvalidValue = 5,
    ArchMismatch = 6,
    MappingError = 7,
    ExecutionFailed = 8,
    NotSupported = 9,
    LicenseError = 10,
    RuntimePrerequisiteMissing = 11,
    RuntimeInProgress = 12,
    RuntimeFPOverflow = 13,
}
impl CudnnStatus {
    fn assert_success(&self) {
        assert_eq!(self, &CudnnStatus::Success);
    }
    fn get_error_str(&self) -> Option<&'static str> {
        match *self {
            CudnnStatus::Success => None,
            CudnnStatus::NotInitialized => Some("NotInitialized"),
            CudnnStatus::AllocFailed => Some("AllocFailed"),
            CudnnStatus::BadParam => Some("BadParam"),
            CudnnStatus::InvalidValue => Some("InvalidValue"),
            CudnnStatus::ArchMismatch => Some("ArchMismatch"),
            CudnnStatus::MappingError => Some("MappingError"),
            CudnnStatus::ExecutionFailed => Some("ExecutionFailed"),
            CudnnStatus::InternalError => Some("InternalError"),
            CudnnStatus::NotSupported => Some("NotSupported"),
            CudnnStatus::LicenseError => Some("LicenseError"),
            CudnnStatus::RuntimePrerequisiteMissing => Some("RuntimePrerequisiteMissing"),
            CudnnStatus::RuntimeInProgress => Some("RuntimeInProgress"),
            CudnnStatus::RuntimeFPOverflow => Some("RuntimeFPOverflow"),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnReduceTensorOp {
    Add = 0,
    Mul = 1,
    Min = 2,
    Max = 3,
    Amax = 4,
    Avg = 5,
    Norm1 = 6,
    Norm2 = 7,
    MulNoZeros = 8,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnReduceTensorIndices {
    NoIndices = 0,
    FlattenedIndices = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnIndicesType {
    Indices32bit = 0,
    Indices64bit = 1,
    Indices16bit = 2,
    Indices8bit = 3,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnConvolutionMode {
    Convolution = 0,
    CrossCorrelation = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnActivationMode {
    Sigmoid = 0,
    Relu = 1,
    Tanh = 2,
    ClippedRelu = 3,
    Elu = 4,
    //Identity = 5, Doesn't work, but it is useless anyway
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnConvolutionFwdAlgo {
    ImplicitGemm = 0,
    ImplicitPrecompGemm = 1,
    Gemm = 2,
    Direct = 3,
    Fft = 4,
    FftTiling = 5,
    Winograd = 6,
    WinogradNonfused = 7,
    Count = 8,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnMathType {
    Default = 0,
    TensorOp = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnTensorFormat {
    Nchw = 0,
    //Nhwc = 1,      Shouldn't be useful
    //NchwVectC = 2,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnRNNMode {
    Relu = 0,
    Tanh = 1,
    Lstm = 2,
    Gru = 3,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnDirectionMode {
    Unidirectional = 0,
    Bidirectional = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnRNNInputMode {
    LinearInput = 0,
    SkipInput = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnRNNAlgo {
    Standard = 0,
    PersistStatic = 1,
    PersistDynamic = 2,
    Count = 3,
}

