//! The OpenCL runtime: device selection, buffers, and kernel launches. The only module with unsafe
//! code; every unsafe call here passes buffers whose lengths match the host slices.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Display;
use std::ptr;
use std::sync::{Mutex, OnceLock, PoisonError};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_double, cl_ulong};
use stanmath_rs_core::{Error, Result};

const F: &str = "opencl";

struct Runtime {
    name: String,
    context: Context,
    queue: CommandQueue,
    /// Built programs, one per kernel, by kernel name.
    programs: Mutex<HashMap<&'static str, Program>>,
}

fn fail(function: &'static str, e: impl Display) -> Error {
    Error::Backend {
        function,
        message: e.to_string(),
    }
}

/// The index in the environment variable `var`, zero when unset.
fn index(var: &str) -> std::result::Result<usize, String> {
    match std::env::var(var) {
        Err(_) => Ok(0),
        Ok(v) => v
            .trim()
            .parse()
            .map_err(|_| format!("{var} must be a nonnegative integer, not {v:?}")),
    }
}

/// Opens the device that `OPENCL_PLATFORM_ID` and `OPENCL_DEVICE_ID` select, as Stan does.
fn open() -> std::result::Result<Runtime, String> {
    let platforms = get_platforms().map_err(|e| e.to_string())?;
    let p = index("OPENCL_PLATFORM_ID")?;
    let platform = platforms
        .get(p)
        .ok_or_else(|| format!("no OpenCL platform {p}; found {}", platforms.len()))?;
    let devices = platform
        .get_devices(CL_DEVICE_TYPE_ALL)
        .map_err(|e| e.to_string())?;
    let d = index("OPENCL_DEVICE_ID")?;
    let device = Device::new(*devices.get(d).ok_or_else(|| {
        format!(
            "no device {d} on OpenCL platform {p}; found {}",
            devices.len()
        )
    })?);
    let name = device.name().map_err(|e| e.to_string())?;
    let extensions = device.extensions().map_err(|e| e.to_string())?;
    if !extensions.split_whitespace().any(|e| e == "cl_khr_fp64") {
        return Err(format!("{name} has no double precision (cl_khr_fp64)"));
    }
    let context = Context::from_device(&device).map_err(|e| e.to_string())?;
    let queue = CommandQueue::create_default(&context, 0).map_err(|e| e.to_string())?;
    Ok(Runtime {
        name,
        context,
        queue,
        programs: Mutex::new(HashMap::new()),
    })
}

fn runtime() -> Result<&'static Runtime> {
    static RUNTIME: OnceLock<std::result::Result<Runtime, String>> = OnceLock::new();
    RUNTIME.get_or_init(open).as_ref().map_err(|m| fail(F, m))
}

/// The name of the OpenCL device in use.
pub(crate) fn device_name() -> Result<String> {
    runtime().map(|r| r.name.clone())
}

/// A device buffer of `len` doubles; `None` when empty, which OpenCL does not allow.
pub(crate) struct DeviceBuffer {
    pub(crate) len: usize,
    buffer: Option<Buffer<cl_double>>,
}

impl DeviceBuffer {
    /// An uninitialized buffer of `len` doubles.
    pub(crate) fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(DeviceBuffer { len, buffer: None });
        }
        let r = runtime()?;
        let buffer = unsafe {
            Buffer::<cl_double>::create(&r.context, CL_MEM_READ_WRITE, len, ptr::null_mut())
        }
        .map_err(|e| fail(F, e))?;
        Ok(DeviceBuffer {
            len,
            buffer: Some(buffer),
        })
    }

    /// A buffer holding a copy of `data`.
    pub(crate) fn from_slice(data: &[f64]) -> Result<Self> {
        let mut b = DeviceBuffer::new(data.len())?;
        if let Some(buffer) = &mut b.buffer {
            let r = runtime()?;
            unsafe {
                r.queue
                    .enqueue_write_buffer(buffer, CL_BLOCKING, 0, data, &[])
            }
            .map_err(|e| fail(F, e))?;
        }
        Ok(b)
    }

    /// The buffer's contents.
    pub(crate) fn to_vec(&self) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.len];
        if let Some(buffer) = &self.buffer {
            let r = runtime()?;
            unsafe {
                r.queue
                    .enqueue_read_buffer(buffer, CL_BLOCKING, 0, &mut out, &[])
            }
            .map_err(|e| fail(F, e))?;
        }
        Ok(out)
    }
}

/// A kernel argument.
pub(crate) enum Arg<'a> {
    Buffer(&'a DeviceBuffer),
    Size(usize),
    Real(f64),
}

/// Runs kernel `kernel` of the program `program` over the index space `global` and waits for it.
/// The program is built from `source()` on first use. Does nothing when the index space is empty.
pub(crate) fn launch(
    function: &'static str,
    program: &'static str,
    kernel: &str,
    source: impl FnOnce() -> String,
    args: &[Arg<'_>],
    global: &[usize],
) -> Result<()> {
    if global.contains(&0) {
        return Ok(());
    }
    let r = runtime()?;
    let kernel = {
        let mut programs = r.programs.lock().unwrap_or_else(PoisonError::into_inner);
        let built = match programs.entry(program) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(
                Program::create_and_build_from_source(&r.context, &source(), "").map_err(
                    |log| {
                        fail(
                            function,
                            format!("program {program} failed to build: {log}"),
                        )
                    },
                )?,
            ),
        };
        Kernel::create(built, kernel).map_err(|e| fail(function, e))?
    };
    let sizes: Vec<cl_ulong> = args
        .iter()
        .map(|a| match a {
            Arg::Size(s) => *s as cl_ulong,
            Arg::Buffer(_) | Arg::Real(_) => 0,
        })
        .collect();
    let reals: Vec<f64> = args
        .iter()
        .map(|a| match a {
            Arg::Real(v) => *v,
            _ => 0.0,
        })
        .collect();
    let mut exec = ExecuteKernel::new(&kernel);
    for ((a, size), real) in args.iter().zip(&sizes).zip(&reals) {
        match a {
            Arg::Buffer(b) => match &b.buffer {
                Some(buffer) => unsafe { exec.set_arg(buffer) },
                None => return Err(fail(function, "an empty buffer passed to a kernel")),
            },
            Arg::Size(_) => unsafe { exec.set_arg(size) },
            Arg::Real(_) => unsafe { exec.set_arg(real) },
        };
    }
    let event = unsafe {
        exec.set_global_work_sizes(global)
            .enqueue_nd_range(&r.queue)
    }
    .map_err(|e| fail(function, e))?;
    event.wait().map_err(|e| fail(function, e))
}
