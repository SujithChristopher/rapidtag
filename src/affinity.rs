//! Pin the rayon worker pool to the "fast" CPU cores on heterogeneous
//! (big.LITTLE) ARM SoCs.
//!
//! On chips like the QCS6490 (4x A55 @1.9GHz + 3x A78 @2.4 + 1x A78 @2.7),
//! rayon's default pool spans all cores and the scheduler freely places scale
//! tasks on the little cores. Every detect call then waits on its slowest
//! task, so one A55-placed task caps the whole batch (measured: 68 fps
//! unpinned vs 234 fps pinned to the big cores, single 1280x800 frame).
//!
//! Core selection, in order:
//!   * `RAPIDTAG_CORES=4-7` / `4,5,6,7` — explicit core list
//!   * `RAPIDTAG_CORES=all` (or unparseable) — disable pinning entirely
//!   * otherwise: autodetect from `/sys/.../cpuinfo_max_freq`; the fast set is
//!     every core clocked above the slowest tier. Homogeneous CPUs (all one
//!     frequency) and non-Linux hosts get no pinning — pool as before.
//!
//! Worker threads are pinned to the *set* (not one core each) so the kernel
//! can still balance within it. The calling (Python) thread is never pinned;
//! capture/IO threads keep the little cores.

use std::fs;

/// Parse "4-7" or "4,5,6" or "3" into a core list.
fn parse_core_list(s: &str) -> Option<Vec<usize>> {
    let mut cores = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let (a, b) = (a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?);
            if a > b {
                return None;
            }
            cores.extend(a..=b);
        } else {
            cores.push(part.parse::<usize>().ok()?);
        }
    }
    cores.sort_unstable();
    cores.dedup();
    if cores.is_empty() {
        None
    } else {
        Some(cores)
    }
}

/// Max frequency per online core from sysfs; empty when unavailable.
fn cpu_max_freqs() -> Vec<(usize, u64)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return out;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        let Some(idx) = name
            .strip_prefix("cpu")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let p = e.path().join("cpufreq/cpuinfo_max_freq");
        if let Ok(s) = fs::read_to_string(p) {
            if let Ok(f) = s.trim().parse::<u64>() {
                out.push((idx, f));
            }
        }
    }
    out.sort_unstable();
    out
}

/// The core set detect workers should run on, or None for "don't pin".
fn fast_cores() -> Option<Vec<usize>> {
    if let Ok(v) = std::env::var("RAPIDTAG_CORES") {
        let v = v.trim();
        if v.eq_ignore_ascii_case("all") {
            return None;
        }
        return parse_core_list(v); // unparseable -> None -> no pinning
    }
    let freqs = cpu_max_freqs();
    if freqs.len() < 2 {
        return None;
    }
    let min = freqs.iter().map(|&(_, f)| f).min()?;
    let fast: Vec<usize> = freqs
        .iter()
        .filter(|&&(_, f)| f > min)
        .map(|&(c, _)| c)
        .collect();
    // homogeneous (fast empty) or degenerate (a single fast core): don't pin
    if fast.len() >= 2 && fast.len() < freqs.len() {
        Some(fast)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cores: &[usize]) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cores {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        // tid 0 = current thread; failure just leaves default affinity
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cores: &[usize]) {}

/// Build the global rayon pool with fast-core affinity. Call once at module
/// init, before any rayon use. No-ops (keeping rayon's defaults) when pinning
/// is disabled or the pool was already built.
pub fn init_pool() {
    let Some(cores) = fast_cores() else {
        return;
    };
    let mut builder = rayon::ThreadPoolBuilder::new();
    // RAYON_NUM_THREADS still wins when the user sets it (builder default
    // honors it); otherwise size the pool to the fast set.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        builder = builder.num_threads(cores.len());
    }
    let _ = builder
        .start_handler(move |_| pin_current_thread(&cores))
        .build_global();
}

#[cfg(test)]
mod tests {
    use super::parse_core_list;

    #[test]
    fn parses_ranges_and_lists() {
        assert_eq!(parse_core_list("4-7"), Some(vec![4, 5, 6, 7]));
        assert_eq!(parse_core_list("4,5,6,7"), Some(vec![4, 5, 6, 7]));
        assert_eq!(parse_core_list("7,4-5"), Some(vec![4, 5, 7]));
        assert_eq!(parse_core_list("3"), Some(vec![3]));
        assert_eq!(parse_core_list(""), None);
        assert_eq!(parse_core_list("7-4"), None);
        assert_eq!(parse_core_list("x"), None);
    }
}
