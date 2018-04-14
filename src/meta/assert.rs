

/*pub(crate) fn assert_equals_float(a: f32, b: f32) {
    let d = a-b;
    if d < -0.000001 || d > 0.000001 {
        panic!("{} != {}", a, b);
    }
}*/
pub(crate) fn assert_eq_usize(a: usize, a_name: &str, b: usize, b_name: &str) {
    if a != b {
        panic!("{} ({}) != {} ({})", a_name, a, b_name, b);
    }
}
pub(crate) fn assert_infeq_usize(inf: usize, inf_name: &str, sup: usize, sup_name: &str) {
    if inf > sup {
        panic!("{} ({}) > {} ({})", inf_name, inf, sup_name, sup);
    }
}