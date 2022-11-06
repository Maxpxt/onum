use crate::ufunc::{BuiltInType::*, Identity, UFuncMetadata};
use cstr::cstr;
use num_traits::Float;
use std::os::raw::{c_double, c_float};

pub static mut CIS: UFuncMetadata<'static, 1, 1, 4> = ufunc_metadata!(
    cstr!(b"cis"),
    cstr!(b"Cosine plus i sine"),
    Identity::None,
    {
        cis_real::<c_float> => (Float) -> (CFloat),
        cis_real::<c_double> => (Double) -> (CDouble),
        cis_complex::<c_float> => (CFloat) -> (CFloat),
        cis_complex::<c_double> => (CDouble) -> (CDouble),
    }
);

ufunc_1_1_fn! {
    cis_real<T> where { T: Float } {
        <T, [T; 2]>|p| {
            let (sin, cos) = p.sin_cos();
            [cos, sin]
        }
    }
}

ufunc_1_1_fn! {
    cis_complex<T> where { T: Float } {
        <[T; 2], [T; 2]>|[a, b]| {
            let (sin, cos) = a.sin_cos();
            let amplitude = (-b).exp();
            [amplitude * cos, amplitude * sin]
        }
    }
}
