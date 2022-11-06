mod utils;
#[macro_use]
mod ufunc;
mod abconj;
mod abs2;
mod cis;

use crate::{abconj::ABCONJ, abs2::ABS2, cis::CIS, ufunc::ufunc};
use pyo3::prelude::*;

#[pymodule]
fn onum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("abs2", unsafe { ufunc(py, &mut ABS2) })?;
    m.add("abconj", unsafe { ufunc(py, &mut ABCONJ) })?;
    m.add("cis", unsafe { ufunc(py, &mut CIS) })?;
    Ok(())
}
