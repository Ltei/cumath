#![allow(dead_code)]

use libc::c_void;



#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[repr(C)]
pub enum CudaError {
    Success                      =      0,
    MissingConfiguration         =      1,
    MemoryAllocation             =      2,
    InitializationError          =      3,
    LaunchFailure                =      4,
    PriorLaunchFailure           =      5,
    LaunchTimeout                =      6,
    LaunchOutOfResources         =      7,
    InvalidDeviceFunction        =      8,
    InvalidConfiguration         =      9,
    InvalidDevice                =     10,
    InvalidValue                 =     11,
    InvalidPitchValue            =     12,
    InvalidSymbol                =     13,
    MapBufferObjectFailed        =     14,
    UnmapBufferObjectFailed      =     15,
    InvalidHostPointer           =     16,
    InvalidDevicePointer         =     17,
    InvalidTexture               =     18,
    InvalidTextureBinding        =     19,
    InvalidChannelDescriptor     =     20,
    InvalidMemcpyDirection       =     21,
    AddressOfConstant            =     22,
    TextureFetchFailed           =     23,
    TextureNotBound              =     24,
    SynchronizationError         =     25,
    InvalidFilterSetting         =     26,
    InvalidNormSetting           =     27,
    MixedDeviceExecution         =     28,
    CudartUnloading              =     29,
    Unknown                      =     30,
    NotYetImplemented            =     31,
    MemoryValueTooLarge          =     32,
    InvalidResourceHandle        =     33,
    NotReady                     =     34,
    InsufficientDriver           =     35,
    SetOnActiveProcess           =     36,
    InvalidSurface               =     37,
    NoDevice                     =     38,
    ECCUncorrectable             =     39,
    SharedObjectSymbolNotFound   =     40,
    SharedObjectInitFailed       =     41,
    UnsupportedLimit             =     42,
    DuplicateVariableName        =     43,
    DuplicateTextureName         =     44,
    DuplicateSurfaceName         =     45,
    DevicesUnavailable           =     46,
    InvalidKernelImage           =     47,
    NoKernelImageForDevice       =     48,
    IncompatibleDriverContext    =     49,
    PeerAccessAlreadyEnabled     =     50,
    PeerAccessNotEnabled         =     51,
    DeviceAlreadyInUse           =     54,
    ProfilerDisabled             =     55,
    ProfilerNotInitialized       =     56,
    ProfilerAlreadyStarted       =     57,
    ProfilerAlreadyStopped       =     58,
    Assert                       =     59,
    TooManyPeers                 =     60,
    HostMemoryAlreadyRegistered  =     61,
    HostMemoryNotRegistered      =     62,
    OperatingSystem              =     63,
    PeerAccessUnsupported        =     64,
    LaunchMaxDepthExceeded       =     65,
    LaunchFileScopedTex          =     66,
    LaunchFileScopedSurf         =     67,
    SyncDepthExceeded            =     68,
    LaunchPendingCountExceeded   =     69,
    NotPermitted                 =     70,
    NotSupported                 =     71,
    HardwareStackError           =     72,
    IllegalInstruction           =     73,
    MisalignedAddress            =     74,
    InvalidAddressSpace          =     75,
    InvalidPc                    =     76,
    IllegalAddress               =     77,
    InvalidPtx                   =     78,
    InvalidGraphicsContext       =     79,
    StartupFailure               =   0x7f,
    ApiFailureBase               =  10000,
}
impl CudaError {
    pub fn assert_success(&self) {
        assert_eq!(self, &CudaError::Success);
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[repr(C)]
pub enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4
}


extern {
    pub fn cudaMalloc(devPtr: *mut*mut c_void, size: usize) -> CudaError;

    pub fn cudaFree(dev_ptr: *mut c_void) -> CudaError;

    pub fn cudaMemcpy(dst: *mut c_void,
                      src: *const c_void,
                      count: usize,
                      kind: CudaMemcpyKind) -> CudaError;

    pub fn cudaMemcpy2D(dst: *mut c_void,
                        dpitch: usize,
                        src: *const c_void,
                        spitch: usize,
                        width: usize,
                        height: usize,
                        kind: CudaMemcpyKind) -> CudaError;

    pub fn cudaDeviceSynchronize() -> CudaError;
}