use crate::ufunc::{BuiltInType::*, Identity, UFuncMetadata};
use cstr::cstr;
use std::{
    ops,
    os::raw::{
        c_double, c_float, c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong,
        c_ushort,
    },
};

pub static mut ABCONJ: UFuncMetadata<'static, 2, 1, 18> = ufunc_metadata!(
    cstr!(b"abconj"),
    cstr!(b"``a`` times the complex conjugate of ``b``"),
    Identity::None,
    {
        abconj_real::<i8> => (Byte, Byte) -> (Byte),
        abconj_real::<u8> => (UByte, UByte) -> (UByte),
        abconj_real::<c_short> => (Short, Short) -> (Short),
        abconj_real::<c_ushort> => (UShort, UShort) -> (UShort),
        abconj_real::<c_int> => (Byte, Byte) -> (Int),
        abconj_real::<c_uint> => (UByte, UByte) -> (UInt),
        abconj_real::<c_long> => (Long, Long) -> (Long),
        abconj_real::<c_ulong> => (ULong, ULong) -> (ULong),
        abconj_real::<c_longlong> => (LongLong, LongLong) -> (LongLong),
        abconj_real::<c_ulonglong> => (ULongLong, ULongLong) -> (ULongLong),
        abconj_real::<c_float> => (Float, Float) -> (Float),
        abconj_real::<c_double> => (Double, Double) -> (Double),
        abconj_complex::<c_float> => (CFloat, CFloat) -> (Float),
        abconj_complex::<c_double> => (CDouble, CDouble) -> (Double),
        abconj_real_complex::<c_float> => (Float, CFloat) -> (Float),
        abconj_real_complex::<c_double> => (Double, CDouble) -> (Double),
        abconj_complex_real::<c_float> => (CFloat, Float) -> (Float),
        abconj_complex_real::<c_double> => (CDouble, Double) -> (Double),
    }
);

ufunc_2_1_fn! {
    abconj_real<T> where { T: ops::Mul<Output = T> + Copy } {
        <T, T, T>|a, b| a * b
    }
}

ufunc_2_1_fn! {
    abconj_complex<T> where {
        T: ops::Mul<Output = T> + ops::Add<Output = T> + ops::Sub<Output = T> + Copy,
    } {
        <[T; 2], [T; 2], [T; 2]>|[ar, ai], [br, bi]| [ar * br + ai * bi, ai * br - ar * bi]
    }
}

ufunc_2_1_fn! {
    abconj_real_complex<T> where {
        T: ops::Mul<Output = T> + ops::Add<Output = T> + ops::Neg<Output = T> + Copy,
    } {
        <T, [T; 2], [T; 2]>|ar, [br, bi]| [ar * br, -(ar * bi)]
    }
}

ufunc_2_1_fn! {
    abconj_complex_real<T> where {
        T: ops::Mul<Output = T> + ops::Add<Output = T> + ops::Neg<Output = T> + Copy,
    } {
        <[T; 2], T, [T; 2]>|[ar, ai], br| [ar * br, ai * br]
    }
}
