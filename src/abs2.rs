use crate::{
    ufunc::{BuiltInType::*, Identity, UFuncMetadata},
    utils::{is_aligned, is_contiguous},
};
use core::slice;
use cstr::cstr;
use numpy::npyffi::npy_intp;
use std::{
    ops,
    os::raw::{
        c_char, c_double, c_float, c_int, c_long, c_longlong, c_short, c_uint, c_ulong,
        c_ulonglong, c_ushort, c_void,
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

pub unsafe extern "C" fn abs2_real<T>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) where
    T: ops::Mul<Output = T> + Copy,
{
    let length = *dimensions as usize;
    let [input_step, output_step] = *(steps as *const [isize; 2]);

    let [mut input, mut output] = *(args as *mut [*mut T; 2]);
    if is_aligned(input) && is_aligned(output) {
        if is_contiguous::<T>(input_step) && is_contiguous::<T>(output_step) {
            let input = slice::from_raw_parts(input, length);
            let output = slice::from_raw_parts_mut(output, length);
            input
                .iter()
                .zip(output.iter_mut())
                .for_each(|(&x, out)| *out = x * x)
        } else {
            for _ in 0..length {
                let x = input.read();
                output.write(x * x);
                input = (input as *mut u8).offset(input_step) as _;
                output = (output as *mut u8).offset(output_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            let x = input.read_unaligned();
            output.write_unaligned(x * x);
            input = (input as *mut u8).offset(input_step) as _;
            output = (output as *mut u8).offset(output_step) as _;
        }
    }
}

pub unsafe extern "C" fn abs2_complex<T>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) where
    T: ops::Mul<Output = T> + ops::Add<Output = T> + Copy,
{
    let length = *dimensions as usize;
    let [input_step, output_step] = *(steps as *const [isize; 2]);

    let mut input = *args as *mut [T; 2];
    let mut output = *args.offset(1) as *mut T;
    if is_aligned(input) && is_aligned(output) {
        if is_contiguous::<[T; 2]>(input_step) && is_contiguous::<T>(output_step) {
            let input = slice::from_raw_parts(input, length);
            let output = slice::from_raw_parts_mut(output, length);
            input
                .iter()
                .zip(output.iter_mut())
                .for_each(|(&[x, y], out)| *out = x * x + y * y)
        } else {
            for _ in 0..length {
                let [x, y] = input.read();
                output.write(x * x + y * y);
                input = (input as *mut u8).offset(input_step) as _;
                output = (output as *mut u8).offset(output_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            let [x, y] = input.read_unaligned();
            output.write_unaligned(x * x + y * y);
            input = (input as *mut u8).offset(input_step) as _;
            output = (output as *mut u8).offset(output_step) as _;
        }
    }
}
