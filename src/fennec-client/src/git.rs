//! Subprocess helpers the native shells need and no Python module provides:
//! a console-less spawn on Windows, `which`, and the git facts behind a
//! repository's branch/changed indicators.
use std::path::PathBuf;

/// Never flash a console window for a helper process (Windows CREATE_NO_WINDOW).
pub fn quiet(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let names: Vec<String> = if cfg!(windows) {
            vec![format!("{name}.exe"), format!("{name}.cmd"), name.to_string()]
        } else {
            vec![name.to_string()]
        };
        for cand in names {
            let p = dir.join(cand);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

pub fn git_out(dir: &str, args: &[&str]) -> Option<String> {
    let mut cmd = std::process::Command::new("git");
    cmd.arg("-C").arg(dir).args(args);
    let out = quiet(&mut cmd).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null())
        .output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[derive(Clone, Default)]
pub struct RepoContext {
    pub branch: String,
    pub changed: usize,
    pub is_repo: bool,
}

impl RepoContext {
    /// The footer text: `clean` or `N changed`.
    pub fn changed_label(&self) -> String {
        if !self.is_repo {
            return "not a git repo".into();
        }
        if self.changed == 0 {
            "clean".into()
        } else {
            format!("{} changed", self.changed)
        }
    }
}

pub fn repo_context(dir: &str) -> RepoContext {
    let head = git_out(dir, &["rev-parse", "--short", "HEAD"]);
    if head.is_none() {
        return RepoContext::default();
    }
    let branch = git_out(dir, &["branch", "--show-current"])
        .filter(|s| !s.is_empty())
        .or(head)
        .unwrap_or_default();
    let changed = git_out(dir, &["status", "--porcelain", "--", "."])
        .map(|s| if s.trim().is_empty() { 0 } else { s.lines().count() })
        .unwrap_or(0);
    RepoContext { branch, changed, is_repo: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_repo_paths_report_honestly() {
        let ctx = repo_context(if cfg!(windows) { "Z:/definitely-not-here" } else { "/definitely-not-here" });
        assert!(!ctx.is_repo);
        assert_eq!(ctx.changed_label(), "not a git repo");
    }

    #[test]
    fn which_finds_a_tool_that_is_on_path() {
        assert!(which("git").is_some() || which("cmd").is_some());
        assert!(which("definitely-not-a-real-binary-9f3a").is_none());
    }
}
