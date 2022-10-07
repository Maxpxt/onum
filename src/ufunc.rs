use numpy::npyffi::npy_intp;
use numpy::npyffi::{PyUFuncGenericFunction, PY_UFUNC_API};
use pyo3::{PyObject, Python};
use std::slice;
use std::{
    ffi::CStr,
    os::raw::{c_char, c_void},
    ptr,
};

use crate::utils::{is_aligned, is_contiguous};

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Identity {
    Zero,
    One,
    MinusOne,
    None,
}

#[allow(dead_code)]
#[repr(i8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BuiltInType {
    Bool = 0,
    Byte = 1,
    UByte = 2,
    Short = 3,
    UShort = 4,
    Int = 5,
    UInt = 6,
    Long = 7,
    ULong = 8,
    LongLong = 9,
    ULongLong = 10,
    Float = 11,
    Double = 12,
    LongDouble = 13,
    CFloat = 14,
    CDouble = 15,
    CLongDouble = 16,
    Object = 17,
    String = 18,
    Unicode = 19,
    Void = 20,
    Datetime = 21,
    Timedelta = 22,
    Half = 23,
}

#[repr(C)]
pub struct Signature<const N_IN: usize, const N_OUT: usize> {
    pub input: [BuiltInType; N_IN],
    pub output: [BuiltInType; N_OUT],
}

pub struct UFuncMetadata<'a, const N_IN: usize, const N_OUT: usize, const N_FUNCS: usize> {
    pub functions: [PyUFuncGenericFunction; N_FUNCS],
    pub data: Option<[*mut c_void; N_FUNCS]>,
    pub signatures: [Signature<N_IN, N_OUT>; N_FUNCS],
    pub identity: Identity,
    pub name: &'a CStr,
    pub doc: &'a CStr,
}

unsafe impl<'a, const N_IN: usize, const N_OUT: usize, const N_FUNCS: usize> Sync
    for UFuncMetadata<'a, N_IN, N_OUT, N_FUNCS>
{
}

pub unsafe fn ufunc<const N_IN: usize, const N_OUT: usize, const N_FUNCS: usize>(
    py: Python,
    metadata: &mut UFuncMetadata<N_IN, N_OUT, N_FUNCS>,
) -> PyObject {
    PyObject::from_owned_ptr(
        py,
        PY_UFUNC_API.PyUFunc_FromFuncAndData(
            py,
            metadata.functions.as_mut_ptr(),
            metadata
                .data
                .as_mut()
                .map_or(ptr::null_mut(), |data| data.as_mut_ptr()),
            metadata.signatures.as_mut_ptr() as *mut _,
            N_FUNCS.try_into().unwrap(),
            N_IN.try_into().unwrap(),
            N_OUT.try_into().unwrap(),
            match metadata.identity {
                Identity::Zero => 0,
                Identity::One => 1,
                Identity::MinusOne => 2,
                Identity::None => -1,
            },
            metadata.name.as_ptr(),
            metadata.doc.as_ptr(),
            0,
        ),
    )
}

macro_rules! ufunc_metadata {
    (
        $name:expr, $doc:expr, $identity:expr, {
            $(
                $f:expr => ($($input:expr),+ $(,)?) -> ($($output:expr),+ $(,)?)
            ),* $(,)?
        }
    ) => {
        $crate::ufunc::UFuncMetadata {
            functions: [
                $(Some($f),)*
            ],
            data: ::core::option::Option::None,
            signatures: [
                $(
                    $crate::ufunc::Signature {
                        input: [$($input),+],
                        output: [$($output),+],
                    },
                )*
            ],
            identity: $identity,
            name: $name,
            doc: $doc,
        }
    };
}

#[inline(always)]
pub unsafe fn ufunc_1_1_fn<I1: Copy, O1>(
    op: impl Fn(I1) -> O1,
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    #[repr(C)]
    struct Args<I1, O1> {
        i1_ptr: *mut I1,
        o1_ptr: *mut O1,
    }

    let Args {
        mut i1_ptr,
        mut o1_ptr,
    } = *(args as *mut Args<I1, O1>);
    let length = *dimensions as usize;
    let [i1_step, o1_step] = *(steps as *const [isize; 2]);

    if is_aligned(i1_ptr) && is_aligned(o1_ptr) {
        if is_contiguous::<I1>(i1_step) && is_contiguous::<O1>(o1_step) {
            let i1_slice = slice::from_raw_parts(i1_ptr, length);
            let o1_slice = slice::from_raw_parts_mut(o1_ptr, length);
            i1_slice
                .iter()
                .zip(o1_slice.iter_mut())
                .for_each(|(&i1, o1)| *o1 = op(i1))
        } else {
            for _ in 0..length {
                o1_ptr.write(op(i1_ptr.read()));
                i1_ptr = (i1_ptr as *mut u8).offset(i1_step) as _;
                o1_ptr = (o1_ptr as *mut u8).offset(o1_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            o1_ptr.write_unaligned(op(i1_ptr.read_unaligned()));
            i1_ptr = (i1_ptr as *mut u8).offset(i1_step) as _;
            o1_ptr = (o1_ptr as *mut u8).offset(o1_step) as _;
        }
    }
}

macro_rules! ufunc_1_1_fn {
    (
        $vis:vis $name:ident <$($type_params:ident),* $(,)?>
        $(where { $($bound:tt)+ })?
        {
            <$i1:ty, $o1:ty> $op:expr
        }
    ) => {
        $vis unsafe extern "C" fn $name<$($type_params),*>(
            args: *mut *mut ::std::os::raw::c_char,
            dimensions: *mut ::numpy::npyffi::npy_intp,
            steps: *mut ::numpy::npyffi::npy_intp,
            data: *mut ::std::os::raw::c_void,
        ) $(where $($bound)+)?
        {
            $crate::ufunc::ufunc_1_1_fn::<$i1, $o1>($op, args, dimensions, steps, data)
        }
    };
}

#[inline(always)]
pub unsafe fn ufunc_2_1_fn<I1: Copy, I2: Copy, O1>(
    op: impl Fn(I1, I2) -> O1,
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    #[repr(C)]
    struct Args<I1, I2, O1> {
        i1_ptr: *mut I1,
        i2_ptr: *mut I2,
        o1_ptr: *mut O1,
    }

    let Args {
        mut i1_ptr,
        mut i2_ptr,
        mut o1_ptr,
    } = *(args as *mut Args<I1, I2, O1>);
    let length = *dimensions as usize;
    let [i1_step, i2_step, o1_step] = *(steps as *const [isize; 3]);

    if is_aligned(i1_ptr) && is_aligned(i2_ptr) && is_aligned(o1_ptr) {
        if is_contiguous::<I1>(i1_step)
            && is_contiguous::<I2>(i2_step)
            && is_contiguous::<O1>(o1_step)
        {
            let i1_slice = slice::from_raw_parts(i1_ptr, length);
            let i2_slice = slice::from_raw_parts(i2_ptr, length);
            let o1_slice = slice::from_raw_parts_mut(o1_ptr, length);
            i1_slice
                .iter()
                .zip(i2_slice.iter())
                .zip(o1_slice.iter_mut())
                .for_each(|((&i1, &i2), o1)| *o1 = op(i1, i2))
        } else {
            for _ in 0..length {
                o1_ptr.write(op(i1_ptr.read(), i2_ptr.read()));
                i1_ptr = (i1_ptr as *mut u8).offset(i1_step) as _;
                i2_ptr = (i2_ptr as *mut u8).offset(i2_step) as _;
                o1_ptr = (o1_ptr as *mut u8).offset(o1_step) as _;
            }
        }
    } else {
        for _ in 0..length {
            o1_ptr.write_unaligned(op(i1_ptr.read_unaligned(), i2_ptr.read_unaligned()));
            i1_ptr = (i1_ptr as *mut u8).offset(i1_step) as _;
            i2_ptr = (i2_ptr as *mut u8).offset(i2_step) as _;
            o1_ptr = (o1_ptr as *mut u8).offset(o1_step) as _;
        }
    }
}

macro_rules! ufunc_2_1_fn {
    (
        $vis:vis $name:ident <$($type_params:ident),* $(,)?>
        $(where { $($bound:tt)+ })?
        {
            <$i1:ty, $i2:ty, $o1:ty> $op:expr
        }
    ) => {
        $vis unsafe extern "C" fn $name<$($type_params),*>(
            args: *mut *mut ::std::os::raw::c_char,
            dimensions: *mut ::numpy::npyffi::npy_intp,
            steps: *mut ::numpy::npyffi::npy_intp,
            data: *mut ::std::os::raw::c_void,
        ) $(where $($bound)+)?
        {
            $crate::ufunc::ufunc_2_1_fn::<$i1, $i2, $o1>($op, args, dimensions, steps, data)
        }
    };
}
