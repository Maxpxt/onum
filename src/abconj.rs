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

pub unsafe extern "C" fn abconj_real<T>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) where
    T: ops::Mul<Output = T> + Copy,
{
    #[repr(C)]
    struct Args<T> {
        a_ptr: *mut T,
        b_ptr: *mut T,
        output: *mut T,
    }

    let Args {
        mut a_ptr,
        mut b_ptr,
        mut output,
    } = *(args as *mut Args<T>);
    let length = *dimensions as usize;
    let [a_step, b_step, output_step] = *(steps as *const [isize; 3]);

    if is_aligned(a_ptr) && is_aligned(b_ptr) && is_aligned(output) {
        if is_contiguous::<T>(a_step)
            && is_contiguous::<T>(b_step)
            && is_contiguous::<T>(output_step)
        {
            let a = slice::from_raw_parts(a_ptr, length);
            let b = slice::from_raw_parts(b_ptr, length);
            let output = slice::from_raw_parts_mut(output, length);
            a.iter()
                .zip(b.iter())
                .zip(output.iter_mut())
                .for_each(|((&a, &b), out)| *out = a * b)
        } else {
            for _ in 0..length {
                let a = a_ptr.read();
                let b = b_ptr.read();
                output.write(a * b);
                a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
                b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
                output = (output as *mut u8).offset(output_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            let a = a_ptr.read_unaligned();
            let b = b_ptr.read_unaligned();
            output.write_unaligned(a * b);
            a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
            b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
            output = (output as *mut u8).offset(output_step) as _;
        }
    }
}

pub unsafe extern "C" fn abconj_complex<T>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) where
    T: ops::Mul<Output = T> + ops::Add<Output = T> + ops::Sub<Output = T> + Copy,
{
    #[repr(C)]
    struct Args<T> {
        a_ptr: *mut [T; 2],
        b_ptr: *mut [T; 2],
        output: *mut [T; 2],
    }

    let Args {
        mut a_ptr,
        mut b_ptr,
        mut output,
    } = *(args as *mut Args<T>);
    let length = *dimensions as usize;
    let [a_step, b_step, output_step] = *(steps as *const [isize; 3]);

    if is_aligned(a_ptr) && is_aligned(b_ptr) && is_aligned(output) {
        if is_contiguous::<[T; 2]>(a_step)
            && is_contiguous::<[T; 2]>(b_step)
            && is_contiguous::<[T; 2]>(output_step)
        {
            let a = slice::from_raw_parts(a_ptr, length);
            let b = slice::from_raw_parts(b_ptr, length);
            let output = slice::from_raw_parts_mut(output, length);
            a.iter().zip(b.iter()).zip(output.iter_mut()).for_each(
                |((&[ar, ai], &[br, bi]), out)| *out = [ar * br + ai * bi, ai * br - ar * bi],
            )
        } else {
            for _ in 0..length {
                let [ar, ai] = a_ptr.read();
                let [br, bi] = b_ptr.read();
                output.write([ar * br + ai * bi, ai * br - ar * bi]);
                a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
                b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
                output = (output as *mut u8).offset(output_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            let [ar, ai] = a_ptr.read_unaligned();
            let [br, bi] = b_ptr.read_unaligned();
            output.write_unaligned([ar * br + ai * bi, ai * br - ar * bi]);
            a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
            b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
            output = (output as *mut u8).offset(output_step) as _;
        }
    }
}

pub unsafe extern "C" fn abconj_real_complex<T>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) where
    T: ops::Mul<Output = T> + ops::Add<Output = T> + ops::Neg<Output = T> + Copy,
{
    #[repr(C)]
    struct Args<T> {
        a_ptr: *mut T,
        b_ptr: *mut [T; 2],
        output: *mut [T; 2],
    }

    let Args {
        mut a_ptr,
        mut b_ptr,
        mut output,
    } = *(args as *mut Args<T>);
    let length = *dimensions as usize;
    let [a_step, b_step, output_step] = *(steps as *const [isize; 3]);

    if is_aligned(a_ptr) && is_aligned(b_ptr) && is_aligned(output) {
        if is_contiguous::<T>(a_step)
            && is_contiguous::<[T; 2]>(b_step)
            && is_contiguous::<[T; 2]>(output_step)
        {
            let a = slice::from_raw_parts(a_ptr, length);
            let b = slice::from_raw_parts(b_ptr, length);
            let output = slice::from_raw_parts_mut(output, length);
            a.iter()
                .zip(b.iter())
                .zip(output.iter_mut())
                .for_each(|((&ar, &[br, bi]), out)| *out = [ar * br, -(ar * bi)])
        } else {
            for _ in 0..length {
                let ar = a_ptr.read();
                let [br, bi] = b_ptr.read();
                output.write([ar * br, -(ar * bi)]);
                a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
                b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
                output = (output as *mut u8).offset(output_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            let ar = a_ptr.read_unaligned();
            let [br, bi] = b_ptr.read_unaligned();
            output.write_unaligned([ar * br, -(ar * bi)]);
            a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
            b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
            output = (output as *mut u8).offset(output_step) as _;
        }
    }
}

pub unsafe extern "C" fn abconj_complex_real<T>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) where
    T: ops::Mul<Output = T> + ops::Add<Output = T> + ops::Neg<Output = T> + Copy,
{
    #[repr(C)]
    struct Args<T> {
        a_ptr: *mut [T; 2],
        b_ptr: *mut T,
        output: *mut [T; 2],
    }

    let Args {
        mut a_ptr,
        mut b_ptr,
        mut output,
    } = *(args as *mut Args<T>);
    let length = *dimensions as usize;
    let [a_step, b_step, output_step] = *(steps as *const [isize; 3]);

    if is_aligned(a_ptr) && is_aligned(b_ptr) && is_aligned(output) {
        if is_contiguous::<[T; 2]>(a_step)
            && is_contiguous::<[T; 2]>(b_step)
            && is_contiguous::<[T; 2]>(output_step)
        {
            let a = slice::from_raw_parts(a_ptr, length);
            let b = slice::from_raw_parts(b_ptr, length);
            let output = slice::from_raw_parts_mut(output, length);
            a.iter()
                .zip(b.iter())
                .zip(output.iter_mut())
                .for_each(|((&[ar, ai], &br), out)| *out = [ar * br, ai * br])
        } else {
            for _ in 0..length {
                let [ar, ai] = a_ptr.read();
                let br = b_ptr.read();
                output.write([ar * br, ai * br]);
                a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
                b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
                output = (output as *mut u8).offset(output_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            let [ar, ai] = a_ptr.read_unaligned();
            let br = b_ptr.read_unaligned();
            output.write_unaligned([ar * br, ai * br]);
            a_ptr = (a_ptr as *mut u8).offset(a_step) as _;
            b_ptr = (b_ptr as *mut u8).offset(b_step) as _;
            output = (output as *mut u8).offset(output_step) as _;
        }
    }
}
