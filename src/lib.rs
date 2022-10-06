mod utils;
#[macro_use]
mod ufunc;
mod abconj;
mod abs2;

use crate::{abconj::ABCONJ, abs2::ABS2, ufunc::ufunc};
use pyo3::prelude::*;

#[pymodule]
fn onum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("abs2", unsafe { ufunc(py, &mut ABS2) })?;
    m.add("abconj", unsafe { ufunc(py, &mut ABCONJ) })?;
    Ok(())
}
