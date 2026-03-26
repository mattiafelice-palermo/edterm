use memmap2::MmapOptions;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::fs::File;

const ENX_VERSION: i32 = 5;
const XDR_INT: i32 = 0;
const XDR_FLOAT: i32 = 1;
const XDR_DOUBLE: i32 = 2;
const XDR_INT64: i32 = 3;
const XDR_CHAR: i32 = 4;
const XDR_STRING: i32 = 5;

#[derive(Clone, Debug)]
struct SubBlock {
    nr: i32,
    dtype: i32,
}

#[derive(Clone, Debug)]
struct Block {
    sub: Vec<SubBlock>,
}

#[derive(Clone, Debug)]
struct FrameHeader {
    file_version: i32,
    t: f64,
    nsum: i32,
    nre: i32,
    blocks: Vec<Block>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn ensure(&self, n: usize) -> Result<(), String> {
        if self.pos + n > self.buf.len() {
            return Err("Unexpected EOF".to_string());
        }
        Ok(())
    }

    fn set_pos(&mut self, new_pos: usize) -> Result<(), String> {
        if new_pos > self.buf.len() {
            return Err("Unexpected EOF".to_string());
        }
        self.pos = new_pos;
        Ok(())
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        self.ensure(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(i32::from_be_bytes(arr))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        self.ensure(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_be_bytes(arr))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        self.ensure(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(i64::from_be_bytes(arr))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        self.ensure(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(f32::from_bits(u32::from_be_bytes(arr)))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        self.ensure(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(f64::from_bits(u64::from_be_bytes(arr)))
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u32()? as usize;
        let padded = (len + 3) & !3;
        self.ensure(padded)?;
        let s = &self.buf[self.pos..self.pos + len];
        self.pos += padded;
        Ok(String::from_utf8_lossy(s).to_string())
    }
}

fn read_real(cur: &mut Cursor<'_>, gmx_double: bool) -> Result<f64, String> {
    if gmx_double {
        cur.read_f64()
    } else {
        Ok(cur.read_f32()? as f64)
    }
}

fn skip_real(cur: &mut Cursor<'_>, gmx_double: bool, n: usize) -> Result<(), String> {
    let sz = if gmx_double { 8 } else { 4 };
    cur.set_pos(cur.pos + sz * n)
}

fn skip_subblock(cur: &mut Cursor<'_>, sub: &SubBlock) -> Result<(), String> {
    let nr = usize::try_from(sub.nr).map_err(|_| "Negative sub-block size".to_string())?;
    match sub.dtype {
        XDR_INT | XDR_FLOAT | XDR_CHAR => cur.set_pos(cur.pos + 4 * nr),
        XDR_DOUBLE | XDR_INT64 => cur.set_pos(cur.pos + 8 * nr),
        XDR_STRING => {
            for _ in 0..nr {
                let len = cur.read_u32()? as usize;
                let padded = (len + 3) & !3;
                cur.set_pos(cur.pos + padded)?;
            }
            Ok(())
        }
        _ => Err("Unsupported XDR sub-block type".to_string()),
    }
}

fn parse_names(cur: &mut Cursor<'_>) -> Result<(Vec<String>, HashMap<String, String>), String> {
    let magic = cur.read_i32()?;
    if magic > 0 {
        return Err("Old EDR format is not supported in rust fast path".to_string());
    }
    if magic != -55555 {
        return Err("Energy names magic mismatch".to_string());
    }

    let file_version = cur.read_i32()?;
    if file_version > ENX_VERSION {
        return Err("EDR file version too new".to_string());
    }
    let nre = cur.read_i32()?;
    if nre < 0 {
        return Err("Invalid nre in EDR names header".to_string());
    }

    let mut names = Vec::with_capacity((nre as usize) + 1);
    let mut units: HashMap<String, String> = HashMap::with_capacity((nre as usize) + 1);
    names.push("Time".to_string());
    units.insert("Time".to_string(), "ps".to_string());

    for _ in 0..nre {
        let name = cur.read_string()?;
        let unit = if file_version >= 2 {
            cur.read_string()?
        } else {
            "kJ/mol".to_string()
        };
        names.push(name.clone());
        units.insert(name, unit);
    }
    Ok((names, units))
}

fn parse_frame_header(cur: &mut Cursor<'_>, nre_expected: usize) -> Result<(FrameHeader, bool), String> {
    let base = cur.pos;
    cur.ensure(8)?;

    let marker = i32::from_be_bytes(
        cur.buf
            .get(base + 4..base + 8)
            .ok_or_else(|| "Unexpected EOF".to_string())?
            .try_into()
            .map_err(|_| "Unexpected EOF".to_string())?,
    );
    let gmx_double = marker != -7777777;

    let first_real = read_real(cur, gmx_double)?;
    if first_real > -1e-10 {
        return Err("Old-format frame header not supported in rust fast path".to_string());
    }
    let hdr_magic = cur.read_i32()?;
    if hdr_magic != -7777777 {
        return Err("Frame magic mismatch".to_string());
    }

    let file_version = cur.read_i32()?;
    if file_version > ENX_VERSION {
        return Err("EDR frame version too new".to_string());
    }
    let t = cur.read_f64()?;
    let _step = cur.read_i64()?;
    let nsum = cur.read_i32()?;
    if file_version >= 3 {
        let _ = cur.read_i64()?;
    }
    if file_version >= 5 {
        let _ = cur.read_f64()?;
    }

    let nre = cur.read_i32()?;
    if nre < 0 {
        return Err("Negative nre in frame".to_string());
    }
    if nre as usize != nre_expected {
        return Err("Frame nre does not match names header".to_string());
    }
    let ndisre = cur.read_i32()?;
    let mut nblock = cur.read_i32()?;
    if nblock < 0 {
        return Err("Negative nblock in frame".to_string());
    }
    if ndisre != 0 {
        if file_version >= 4 {
            return Err("Unexpected ndisre with file version >=4".to_string());
        }
        nblock += 1;
    }

    let dtreal = if gmx_double { XDR_DOUBLE } else { XDR_FLOAT };
    let mut blocks = Vec::with_capacity(nblock as usize);
    let mut startb = 0;
    if ndisre > 0 {
        blocks.push(Block {
            sub: vec![
                SubBlock {
                    nr: ndisre,
                    dtype: dtreal,
                },
                SubBlock {
                    nr: ndisre,
                    dtype: dtreal,
                },
            ],
        });
        startb = 1;
    }

    for _ in startb..nblock {
        if file_version < 4 {
            let nrint = cur.read_i32()?;
            blocks.push(Block {
                sub: vec![SubBlock {
                    nr: nrint,
                    dtype: dtreal,
                }],
            });
        } else {
            let _id = cur.read_i32()?;
            let nsub = cur.read_i32()?;
            if nsub < 0 {
                return Err("Negative nsub in frame block".to_string());
            }
            let mut subs = Vec::with_capacity(nsub as usize);
            for _ in 0..nsub {
                let dtype = cur.read_i32()?;
                let nr = cur.read_i32()?;
                subs.push(SubBlock { nr, dtype });
            }
            blocks.push(Block { sub: subs });
        }
    }

    let _e_size = cur.read_i32()?;
    let _ = cur.read_i32()?;
    let _ = cur.read_i32()?;

    Ok((
        FrameHeader {
            file_version,
            t,
            nsum,
            nre,
            blocks,
        },
        gmx_double,
    ))
}

fn report_progress(
    py: Python<'_>,
    callback: Option<&PyObject>,
    bytes_read: usize,
    total_bytes: usize,
    records_read: usize,
) -> PyResult<()> {
    if let Some(cb) = callback {
        cb.call1(py, (bytes_read, total_bytes, records_read))?;
    }
    Ok(())
}

fn read_edr_impl(
    py: Python<'_>,
    path: &str,
    frame_stride: usize,
    progress_callback: Option<&PyObject>,
    progress_stride: usize,
) -> Result<(Vec<Vec<f64>>, Vec<String>, Vec<f64>), String> {
    let file = File::open(path).map_err(|e| format!("Failed opening file: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| format!("Failed mmap: {e}"))? };
    let buf: &[u8] = &mmap;
    let mut cur = Cursor::new(buf);

    let (all_names, _units) = parse_names(&mut cur)?;
    let nre_expected = all_names.len().saturating_sub(1);

    let mut all_energies: Vec<Vec<f64>> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    let mut frame_count: usize = 0;
    let total_bytes = buf.len();

    if report_progress(py, progress_callback, cur.pos, total_bytes, 0).is_err() {
        return Err("Progress callback failed".to_string());
    }

    loop {
        if cur.pos >= cur.buf.len() {
            break;
        }

        let (hdr, gmx_double) = match parse_frame_header(&mut cur, nre_expected) {
            Ok(v) => v,
            Err(e) => {
                if e == "Unexpected EOF" {
                    break;
                }
                return Err(e);
            }
        };

        let keep_frame = frame_count % frame_stride == 0;
        if keep_frame {
            let mut row = Vec::with_capacity((hdr.nre as usize) + 1);
            row.push(hdr.t);
            for _ in 0..hdr.nre {
                let e = read_real(&mut cur, gmx_double)?;
                row.push(e);
                if hdr.file_version == 1 || hdr.nsum > 0 {
                    skip_real(&mut cur, gmx_double, if hdr.file_version == 1 { 3 } else { 2 })?;
                }
            }
            all_energies.push(row);
            times.push(hdr.t);
        } else {
            for _ in 0..hdr.nre {
                skip_real(&mut cur, gmx_double, 1)?;
                if hdr.file_version == 1 || hdr.nsum > 0 {
                    skip_real(&mut cur, gmx_double, if hdr.file_version == 1 { 3 } else { 2 })?;
                }
            }
        }

        for block in &hdr.blocks {
            for sub in &block.sub {
                skip_subblock(&mut cur, sub)?;
            }
        }

        frame_count += 1;
        if progress_callback.is_some() && (frame_count == 1 || frame_count % progress_stride == 0) {
            if report_progress(py, progress_callback, cur.pos, total_bytes, frame_count).is_err() {
                return Err("Progress callback failed".to_string());
            }
        }
    }

    if report_progress(py, progress_callback, total_bytes, total_bytes, frame_count).is_err() {
        return Err("Progress callback failed".to_string());
    }
    Ok((all_energies, all_names, times))
}

fn read_edr_packed_impl(
    py: Python<'_>,
    path: &str,
    frame_stride: usize,
    progress_callback: Option<&PyObject>,
    progress_stride: usize,
) -> Result<(Vec<f64>, usize, usize, Vec<String>), String> {
    let file = File::open(path).map_err(|e| format!("Failed opening file: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| format!("Failed mmap: {e}"))? };
    let buf: &[u8] = &mmap;
    let mut cur = Cursor::new(buf);

    let (all_names, _units) = parse_names(&mut cur)?;
    let ncols = all_names.len();
    let nre_expected = ncols.saturating_sub(1);

    let mut flat: Vec<f64> = Vec::new();
    let mut frame_count: usize = 0;
    let mut kept_rows: usize = 0;
    let total_bytes = buf.len();

    if report_progress(py, progress_callback, cur.pos, total_bytes, 0).is_err() {
        return Err("Progress callback failed".to_string());
    }

    loop {
        if cur.pos >= cur.buf.len() {
            break;
        }

        let (hdr, gmx_double) = match parse_frame_header(&mut cur, nre_expected) {
            Ok(v) => v,
            Err(e) => {
                if e == "Unexpected EOF" {
                    break;
                }
                return Err(e);
            }
        };

        let keep_frame = frame_count % frame_stride == 0;
        if keep_frame {
            flat.push(hdr.t);
            for _ in 0..hdr.nre {
                let e = read_real(&mut cur, gmx_double)?;
                flat.push(e);
                if hdr.file_version == 1 || hdr.nsum > 0 {
                    skip_real(&mut cur, gmx_double, if hdr.file_version == 1 { 3 } else { 2 })?;
                }
            }
            kept_rows += 1;
        } else {
            for _ in 0..hdr.nre {
                skip_real(&mut cur, gmx_double, 1)?;
                if hdr.file_version == 1 || hdr.nsum > 0 {
                    skip_real(&mut cur, gmx_double, if hdr.file_version == 1 { 3 } else { 2 })?;
                }
            }
        }

        for block in &hdr.blocks {
            for sub in &block.sub {
                skip_subblock(&mut cur, sub)?;
            }
        }

        frame_count += 1;
        if progress_callback.is_some() && (frame_count == 1 || frame_count % progress_stride == 0) {
            if report_progress(py, progress_callback, cur.pos, total_bytes, frame_count).is_err() {
                return Err("Progress callback failed".to_string());
            }
        }
    }

    if report_progress(py, progress_callback, total_bytes, total_bytes, frame_count).is_err() {
        return Err("Progress callback failed".to_string());
    }
    Ok((flat, kept_rows, ncols, all_names))
}

fn get_unit_dictionary_impl(path: &str) -> Result<HashMap<String, String>, String> {
    let file = File::open(path).map_err(|e| format!("Failed opening file: {e}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| format!("Failed mmap: {e}"))? };
    let mut cur = Cursor::new(&mmap);
    let (_names, units) = parse_names(&mut cur)?;
    Ok(units)
}

#[pyfunction]
#[pyo3(signature = (path, frame_stride=1, progress_callback=None, progress_stride=1000))]
fn read_edr(
    py: Python<'_>,
    path: String,
    frame_stride: usize,
    progress_callback: Option<PyObject>,
    progress_stride: usize,
) -> PyResult<PyObject> {
    let frame_stride = frame_stride.max(1);
    let progress_stride = progress_stride.max(1);
    match read_edr_impl(
        py,
        &path,
        frame_stride,
        progress_callback.as_ref(),
        progress_stride,
    ) {
        Ok((all_energies, all_names, times)) => {
            let energies_py = PyList::new_bound(py, all_energies);
            let names_py = PyList::new_bound(py, all_names);
            let times_py = PyList::new_bound(py, times);
            Ok((energies_py, names_py, times_py).into_py(py))
        }
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (path, frame_stride=1, progress_callback=None, progress_stride=1000))]
fn read_edr_packed(
    py: Python<'_>,
    path: String,
    frame_stride: usize,
    progress_callback: Option<PyObject>,
    progress_stride: usize,
) -> PyResult<PyObject> {
    let frame_stride = frame_stride.max(1);
    let progress_stride = progress_stride.max(1);
    match read_edr_packed_impl(
        py,
        &path,
        frame_stride,
        progress_callback.as_ref(),
        progress_stride,
    ) {
        Ok((flat, nrows, ncols, names)) => {
            let byte_len = flat.len() * std::mem::size_of::<f64>();
            let raw: &[u8] = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, byte_len) };
            let payload = PyBytes::new_bound(py, raw);
            let names_py = PyList::new_bound(py, names);
            Ok((payload, nrows, ncols, names_py).into_py(py))
        }
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
fn get_unit_dictionary(py: Python<'_>, path: String) -> PyResult<PyObject> {
    match get_unit_dictionary_impl(&path) {
        Ok(units) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in units {
                dict.set_item(k, v)?;
            }
            Ok(dict.into_py(py))
        }
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pymodule]
fn edterm_rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_edr, m)?)?;
    m.add_function(wrap_pyfunction!(read_edr_packed, m)?)?;
    m.add_function(wrap_pyfunction!(get_unit_dictionary, m)?)?;
    Ok(())
}
