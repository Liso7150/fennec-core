use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MAX_READ_CHARS: usize = 24_000;
pub const MAX_RESULT_CHARS: usize = 7_000;
pub const MAX_DIRECTORY_ENTRIES: usize = 500;
pub const MAX_GREP_HITS: usize = 200;

#[derive(Debug)]
pub enum ToolError {
    InvalidPath,
    OutsideWorkspace,
    RestrictedPath,
    Io(std::io::Error),
    Regex(regex::Error),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPath => write!(f, "invalid path"),
            Self::OutsideWorkspace => write!(f, "path escapes the configured workspace"),
            Self::RestrictedPath => write!(f, "credential or reserved path is not accessible"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Regex(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ToolError {}

impl From<std::io::Error> for ToolError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<regex::Error> for ToolError {
    fn from(error: regex::Error) -> Self {
        Self::Regex(error)
    }
}

#[derive(Clone, Debug)]
pub struct WorkspaceTools {
    root: PathBuf,
}

impl WorkspaceTools {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, ToolError> {
        let root = fs::canonicalize(root)?;
        if !root.is_dir() {
            return Err(ToolError::InvalidPath);
        }
        Ok(Self { root })
    }

    pub fn from_env() -> Result<Self, ToolError> {
        let root = std::env::var_os("FENNEC_WORKSPACE").ok_or(ToolError::InvalidPath)?;
        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list_dir(&self, path: &str) -> Result<Value, ToolError> {
        let resolved = self.safe_path(path)?;
        if !resolved.is_dir() {
            return Ok(json!({ "ok": false, "error": format!("not a directory: {path}") }));
        }
        let mut entries = Vec::new();
        for entry in fs::read_dir(&resolved)?.take(2_000) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            let child = match self.safe_path(&join_relative(&resolved, &self.root, &name)) {
                Ok(child) => child,
                Err(_) => continue,
            };
            let metadata = match fs::metadata(&child) {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            entries.push(DirectoryEntry {
                name: if metadata.is_dir() { format!("{name}/") } else { name },
                size: metadata.is_file().then_some(metadata.len()),
            });
            if entries.len() >= MAX_DIRECTORY_ENTRIES {
                break;
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(json!({
            "ok": true,
            "path": relative_display(&self.root, &resolved),
            "entries": entries,
        }))
    }

    pub fn read_file(&self, path: &str, numbered: bool) -> Result<Value, ToolError> {
        let resolved = self.safe_path(path)?;
        if !resolved.is_file() {
            return Ok(json!({ "ok": false, "error": format!("not a file: {path}") }));
        }
        let bytes = fs::metadata(&resolved)?.len();
        let mut limited = Vec::new();
        fs::File::open(&resolved)?
            .take((MAX_READ_CHARS + 1) as u64)
            .read_to_end(&mut limited)?;
        let text = String::from_utf8_lossy(&limited).into_owned();
        let lines: Vec<&str> = text.lines().collect();
        let content = if numbered {
            lines
                .iter()
                .take(600)
                .enumerate()
                .map(|(index, line)| format!("{}\t{}", index + 1, line))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            truncate(&text, MAX_READ_CHARS)
        };
        Ok(json!({
            "ok": true,
            "path": path,
            "lines": lines.len(),
            "bytes": bytes,
            "content": content,
        }))
    }

    /// One file's raw bytes — at most `limit` of them — plus its size on disk.
    /// `read_file` decodes to text, which is what the loop's context needs and
    /// what turns a PNG into replacement characters; a pane that paints pixels
    /// needs the bytes whole. This stays out of `call` so no agent tool can
    /// drop a megabyte of base64 into its own prompt.
    pub fn read_bytes(&self, path: &str, limit: u64) -> Result<(Vec<u8>, u64), ToolError> {
        let resolved = self.safe_path(path)?;
        let metadata = fs::metadata(&resolved)?;
        if !metadata.is_file() {
            return Err(ToolError::InvalidPath);
        }
        let mut bytes = Vec::new();
        fs::File::open(&resolved)?
            .take(limit)
            .read_to_end(&mut bytes)?;
        Ok((bytes, metadata.len()))
    }

    pub fn grep(&self, pattern: &str, path: &str) -> Result<Value, ToolError> {
        let regex = Regex::new(pattern)?;
        let root = self.safe_path(path)?;
        let mut hits = Vec::new();
        for file in self.walk_files(&root, 2_000) {
            let file = match file {
                Ok(file) => file,
                Err(_) => continue,
            };
            let text = match fs::read(&file) {
                Ok(bytes) => String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_READ_CHARS)]).into_owned(),
                Err(_) => continue,
            };
            for (line_number, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    hits.push(json!({
                        "file": relative_display(&self.root, &file),
                        "line": line_number + 1,
                        "text": truncate(line.trim_end(), 200),
                    }));
                    if hits.len() >= MAX_GREP_HITS {
                        break;
                    }
                }
            }
            if hits.len() >= MAX_GREP_HITS {
                break;
            }
        }
        Ok(json!({ "ok": true, "pattern": pattern, "count": hits.len(), "hits": hits }))
    }

    pub fn call(&self, name: &str, arguments: &Value) -> Result<Value, ToolError> {
        match name {
            "fennec_list_dir" | "list_dir" => self.list_dir(
                arguments.get("path").and_then(Value::as_str).unwrap_or("."),
            ),
            "fennec_read_file" | "read_file" => self.read_file(
                arguments.get("path").and_then(Value::as_str).ok_or(ToolError::InvalidPath)?,
                arguments.get("numbered").and_then(Value::as_bool).unwrap_or(true),
            ),
            "fennec_grep" | "grep" => self.grep(
                arguments.get("pattern").and_then(Value::as_str).ok_or(ToolError::InvalidPath)?,
                arguments.get("path").and_then(Value::as_str).unwrap_or("."),
            ),
            _ => Err(ToolError::InvalidPath),
        }
    }

    fn safe_path(&self, relative: &str) -> Result<PathBuf, ToolError> {
        if relative.contains('\0') {
            return Err(ToolError::InvalidPath);
        }
        let candidate = self.root.join(relative);
        let parent = if candidate == self.root {
            candidate.as_path()
        } else {
            candidate.parent().unwrap_or(&self.root)
        };
        let resolved_parent = fs::canonicalize(parent).map_err(ToolError::Io)?;
        if !is_inside(&self.root, &resolved_parent) {
            return Err(ToolError::OutsideWorkspace);
        }
        let resolved = fs::canonicalize(candidate).map_err(ToolError::Io)?;
        if !is_inside(&self.root, &resolved) {
            return Err(ToolError::OutsideWorkspace);
        }
        let relative = resolved.strip_prefix(&self.root).map_err(|_| ToolError::OutsideWorkspace)?;
        if relative.components().any(|component| restricted_component(&component.as_os_str().to_string_lossy())) {
            return Err(ToolError::RestrictedPath);
        }
        Ok(resolved)
    }

    fn walk_files(&self, root: &Path, limit: usize) -> Vec<Result<PathBuf, ToolError>> {
        let mut output = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(directory) = stack.pop() {
            let entries = match fs::read_dir(&directory) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(_) => continue,
                };
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if restricted_component(&name) || matches!(name.as_str(), "__pycache__" | "node_modules" | ".venv") {
                    continue;
                }
                let resolved = match self.safe_path(&join_relative(&path, &self.root, "")) {
                    Ok(resolved) => resolved,
                    Err(_) => continue,
                };
                if resolved.is_dir() {
                    stack.push(resolved);
                } else if resolved.is_file()
                    && !SKIP_EXTENSIONS.iter().any(|suffix| name.to_ascii_lowercase().ends_with(suffix))
                    && fs::metadata(&resolved).map(|m| m.len() <= 400_000).unwrap_or(false)
                {
                    output.push(Ok(resolved));
                    if output.len() >= limit {
                        return output;
                    }
                }
            }
        }
        output
    }
}

#[derive(Serialize)]
struct DirectoryEntry {
    name: String,
    size: Option<u64>,
}

const SKIP_EXTENSIONS: [&str; 10] = [
    ".pt", ".png", ".jpg", ".jpeg", ".gif", ".pyc", ".safetensors", ".exe", ".dll", ".zip",
];

fn join_relative(path: &Path, root: &Path, child: &str) -> String {
    let base = path.strip_prefix(root).unwrap_or(path);
    let mut value = base.to_string_lossy().into_owned();
    if !child.is_empty() {
        if !value.is_empty() {
            value.push(std::path::MAIN_SEPARATOR);
        }
        value.push_str(child);
    }
    value
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|value| {
            let text = value.to_string_lossy();
            if text.is_empty() { ".".to_string() } else { text.into_owned() }
        })
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

fn is_inside(root: &Path, path: &Path) -> bool {
    #[cfg(windows)]
    {
        let root_components: Vec<String> = root
            .components()
            .map(|component| component.as_os_str().to_string_lossy().to_ascii_lowercase())
            .collect();
        let path_components: Vec<String> = path
            .components()
            .map(|component| component.as_os_str().to_string_lossy().to_ascii_lowercase())
            .collect();
        path_components.starts_with(&root_components)
    }
    #[cfg(not(windows))]
    {
        path.starts_with(root)
    }
}

fn restricted_component(component: &str) -> bool {
    let name = component.trim_end_matches([' ', '.']).to_ascii_lowercase();
    let denied = matches!(
        name.as_str(),
        ".git" | ".ssh" | ".aws" | ".azure" | ".kube" | ".gnupg" | ".docker" | ".npmrc"
            | ".pypirc" | ".netrc" | "credentials" | "credentials.json" | "authorized_keys"
            | "known_hosts" | "id_rsa" | "id_ed25519" | "id_dsa" | "id_ecdsa" | ".env"
    );
    denied
        || name.starts_with(".env.")
        || [".pem", ".key", ".p12", ".pfx"].iter().any(|suffix| name.ends_with(suffix))
        || name.split(['.', '_', '-']).any(|part| matches!(part, "token" | "tokens" | "secret" | "secrets" | "credential" | "credentials"))
        || component.contains(':')
        || (component.ends_with([' ', '.']) && component != ".")
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_string()
    } else {
        let kept: String = text.chars().take(limit).collect();
        format!("{kept}\n…[truncated {} chars]", text.chars().count().saturating_sub(limit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn blocks_escape_and_credentials() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("ok.txt"), "hello").unwrap();
        fs::write(temp.path().join(".env"), "secret").unwrap();
        let tools = WorkspaceTools::new(temp.path()).unwrap();
        assert!(matches!(tools.read_file("../ok.txt", true), Err(ToolError::OutsideWorkspace)));
        assert!(matches!(tools.read_file(".env", true), Err(ToolError::RestrictedPath)));
    }

    #[test]
    fn reads_and_greps_with_bounds() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("main.rs"), "alpha\nbeta alpha\n").unwrap();
        let tools = WorkspaceTools::new(temp.path()).unwrap();
        assert_eq!(tools.read_file("main.rs", true).unwrap()["lines"], 2);
        assert_eq!(tools.grep("alpha", ".").unwrap()["count"], 2);
    }

    #[test]
    fn reads_bytes_whole_capped_and_jailed() {
        let temp = tempfile::tempdir().unwrap();
        // A PNG header is the case that matters: no text decoder hands these
        // bytes back unchanged, which is why the pane does not use `read_file`.
        let png: [u8; 12] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 3];
        fs::write(temp.path().join("shot.png"), png).unwrap();
        fs::write(temp.path().join(".env"), "secret").unwrap();
        let tools = WorkspaceTools::new(temp.path()).unwrap();
        let (bytes, size) = tools.read_bytes("shot.png", 8).unwrap();
        assert_eq!(bytes.as_slice(), &png[..8]);
        assert_eq!(size, png.len() as u64, "the cap limits the read, not the reported size");
        assert!(matches!(tools.read_bytes(".env", 8), Err(ToolError::RestrictedPath)));
        assert!(matches!(tools.read_bytes("../shot.png", 8), Err(ToolError::OutsideWorkspace)));
        assert!(matches!(tools.read_bytes(".", 8), Err(ToolError::InvalidPath)));
    }
}
