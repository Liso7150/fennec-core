//! Self-reported memory and CPU time. The Python side of the benchmark reads
//! these with `psutil` from inside the measured process, so the Rust side has to
//! answer the same question about itself for the numbers to mean the same thing.

use fennec_pyjson::{Map, Value};

/// `(working set, peak working set, CPU ms)`, all `None` where the host has
/// no equivalent of `GetProcessMemoryInfo`.
#[cfg(windows)]
fn sample() -> (Option<i64>, Option<i64>, Option<i64>) {
    use std::ffi::c_void;

    #[repr(C)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    extern "system" {
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, cb: u32) -> i32;
        fn GetProcessTimes(process: *mut c_void, creation: *mut u64, exit: *mut u64,
                           kernel: *mut u64, user: *mut u64) -> i32;
    }

    // GetCurrentProcess() is the -1 pseudo-handle, without the call.
    let self_process = -1isize as *mut c_void;
    let mut counters = unsafe { std::mem::zeroed::<Counters>() };
    counters.cb = std::mem::size_of::<Counters>() as u32;
    let memory_ok =
        unsafe { K32GetProcessMemoryInfo(self_process, &mut counters, counters.cb) } != 0;
    // All four FILETIME slots must be writable: the kernel does not tolerate a
    // null here, however little it has to say about a process still running.
    let mut creation = 0u64;
    let mut exit = 0u64;
    let mut kernel = 0u64;
    let mut user = 0u64;
    let times_ok = unsafe {
        GetProcessTimes(self_process, &mut creation, &mut exit, &mut kernel, &mut user)
    } != 0;
    (
        memory_ok.then(|| counters.working_set_size as i64),
        memory_ok.then(|| counters.peak_working_set_size as i64),
        // 100 ns ticks, and the process has not exited, so kernel time is 0.
        times_ok.then(|| ((user + kernel) as f64 / 10_000.0) as i64),
    )
}

#[cfg(not(windows))]
fn sample() -> (Option<i64>, Option<i64>, Option<i64>) {
    (None, None, None)
}

pub fn report() -> Value {
    let (rss, peak, cpu_ms) = sample();
    let number = |value: Option<i64>| match value {
        Some(bytes) => Value::from(bytes),
        None => Value::Null,
    };
    let mut out = Map::new();
    out.insert("rss_bytes".to_string(), number(rss));
    out.insert("peak_rss_bytes".to_string(), number(peak));
    out.insert("cpu_ms".to_string(), number(cpu_ms));
    Value::Object(out)
}
