use crate::ufunc::{BuiltInType::*, Identity, UFuncMetadata};
use cstr::cstr;
use std::{
    ops,
    os::raw::{
        c_double, c_float, c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong,
        c_ushort,
    },
};

pub static mut ABS2: UFuncMetadata<'static, 1, 1, 14> = ufunc_metadata!(
    cstr!(b"abs2"),
    cstr!(b"Absolute value squared"),
    Identity::None,
    {
        abs2_real::<i8> => (Byte) -> (Byte),
        abs2_real::<u8> => (UByte) -> (UByte),
        abs2_real::<c_short> => (Short) -> (Short),
        abs2_real::<c_ushort> => (UShort) -> (UShort),
        abs2_real::<c_int> => (Byte) -> (Int),
        abs2_real::<c_uint> => (UByte) -> (UInt),
        abs2_real::<c_long> => (Long) -> (Long),
        abs2_real::<c_ulong> => (ULong) -> (ULong),
        abs2_real::<c_longlong> => (LongLong) -> (LongLong),
        abs2_real::<c_ulonglong> => (ULongLong) -> (ULongLong),
        abs2_real::<c_float> => (Float) -> (Float),
        abs2_real::<c_double> => (Double) -> (Double),
        abs2_complex::<c_float> => (CFloat) -> (Float),
        abs2_complex::<c_double> => (CDouble) -> (Double),
    }
);

ufunc_1_1_fn! {
    abs2_real<T> where { T: ops::Mul<Output = T> + Copy } {
        <T, T>|x| x * x
    }
}

ufunc_1_1_fn! {
    abs2_complex<T> where { T: ops::Mul<Output = T> + ops::Add<Output = T> + Copy } {
        <[T; 2], T>|[x, y]| x * x + y * y
    }
}
