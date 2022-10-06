use std::mem;

#[inline]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    ptr as usize % mem::align_of::<T>() == 0
}

#[inline]
pub fn is_contiguous<T>(byte_stride: isize) -> bool {
    byte_stride as usize == mem::size_of::<T>()
}
