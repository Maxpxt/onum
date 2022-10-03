use numpy::npyffi::{PyUFuncGenericFunction, PY_UFUNC_API};
use pyo3::{PyObject, Python};
use std::{ffi::CStr, os::raw::c_void, ptr};

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
