mod utils;
#[macro_use]
mod ufunc;
mod ufuncs;

use crate::{
    ufunc::ufunc,
    ufuncs::{ABCONJ, ABS2, CIS},
};
use pyo3::prelude::*;

#[pymodule]
fn onum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("abs2", unsafe { ufunc(py, &mut ABS2) })?;
    m.add("abconj", unsafe { ufunc(py, &mut ABCONJ) })?;
    m.add("cis", unsafe { ufunc(py, &mut CIS) })?;
    Ok(())
}
