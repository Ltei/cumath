

#[macro_use]
mod debug;

#[macro_use]
mod vector_op;

#[macro_use]
mod vector_op_mut_i32;

#[macro_use]
mod vector_op_mut_f32;

#[macro_use]
mod matrix_op;

#[macro_use]
mod matrix_op_mut_i32;

#[macro_use]
mod matrix_op_mut_f32;


macro_rules! impl_immutable_vector_holder {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl_Debug_vector!($name $(,$lifetimes)*);
        impl_CuVectorOp!($name $(,$lifetimes)*);
    };
}
macro_rules! impl_mutable_vector_holder {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl_Debug_vector!($name $(,$lifetimes)*);
        impl_CuVectorOp!($name $(,$lifetimes)*);
        impl_CuVectorOpMut_i32!($name $(,$lifetimes)*);
        impl_CuVectorOpMut_f32!($name $(,$lifetimes)*);
    };
}

macro_rules! impl_immutable_packed_matrix_holder {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl_Debug_matrix_packed!($name $(,$lifetimes)*);
        impl_CuMatrixOp_packed!($name $(,$lifetimes)*);
    };
}
macro_rules! impl_mutable_packed_matrix_holder {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl_Debug_matrix_packed!($name $(,$lifetimes)*);
        impl_CuMatrixOp_packed!($name $(,$lifetimes)*);
        impl_CuMatrixOpMut_packed_i32!($name $(,$lifetimes)*);
        impl_CuMatrixOpMut_packed_f32!($name $(,$lifetimes)*);
    };
}

macro_rules! impl_immutable_fragmented_matrix_holder {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl_Debug_matrix_fragmented!($name $(,$lifetimes)*);
        impl_CuMatrixOp_fragmented!($name $(,$lifetimes)*);
    };
}
macro_rules! impl_mutable_fragmented_matrix_holder {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl_Debug_matrix_fragmented!($name $(,$lifetimes)*);
        impl_CuMatrixOp_fragmented!($name $(,$lifetimes)*);
        impl_CuMatrixOpMut_fragmented_i32!($name $(,$lifetimes)*);
        impl_CuMatrixOpMut_fragmented_f32!($name $(,$lifetimes)*);
    };
}