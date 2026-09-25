use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// How a supported file is turned into text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Extractor {
    Pdf,
    Office,
    PlainText,
}

/// The single extension table behind both [`extract_text`] and
/// [`supported_extension`]: a file the indexer accepts is always extractable
/// (and the other way round), because neither has its own list to drift from.
fn extractor_for(ext: &str) -> Option<Extractor> {
    match ext {
        "pdf" => Some(Extractor::Pdf),
        "docx" | "xlsx" | "pptx" => Some(Extractor::Office),
        "txt" | "md" | "rs" | "py" | "js" | "ts" | "go" | "java" | "c" | "cpp" | "h" | "json"
        | "yaml" | "yml" | "toml" | "xml" | "csv" | "html" | "css" => Some(Extractor::PlainText),
        _ => None,
    }
}

/// Lower-cased file extension, or `""` when there is none (or it is not
/// valid UTF-8) — so the error message and the table lookup see one shape.
fn normalized_extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

async fn read_text_with_encoding_detection(path: &Path) -> Result<String> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    if bytes.is_empty() {
        return Ok(String::new());
    }

    if let Ok(text) = std::str::from_utf8(&bytes) {
        return Ok(text.to_string());
    }

    let mut detector = chardetng::EncodingDetector::new(chardetng::Iso2022JpDetection::Deny);
    detector.feed(&bytes, true);
    let encoding = detector.guess(None, chardetng::Utf8Detection::Allow);

    let (cow, _, had_errors) = encoding.decode(&bytes);
    if !had_errors {
        return Ok(cow.into_owned());
    }

    let (cow, _, had_errors) = encoding_rs::UTF_8.decode(&bytes);
    if !had_errors {
        return Ok(cow.into_owned());
    }

    Err(anyhow::anyhow!(
        "Could not detect encoding for {}",
        path.display()
    ))
}

pub async fn extract_text(path: &Path) -> Result<String> {
    let ext = normalized_extension(path);

    match extractor_for(&ext) {
        Some(Extractor::Pdf) => unpdf::extract_text(path).context("extracting PDF"),
        Some(Extractor::Office) => undoc::extract_text(path).context("extracting Office document"),
        Some(Extractor::PlainText) => read_text_with_encoding_detection(path).await,
        None => Err(anyhow::anyhow!("Unsupported file format: {}", ext)),
    }
}

pub fn supported_extension(path: &Path) -> bool {
    extractor_for(&normalized_extension(path)).is_some()
}

/// Every supported file below `root`, in the order a depth-first walk reaches
/// them. Unreadable directories are skipped silently — the per-file indexing
/// step reports real failures — and a `root` that is itself a supported file
/// is the only entry.
pub async fn walk_supported_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(current) = stack.pop() {
        if current.is_dir() {
            if let Ok(mut entries) = tokio::fs::read_dir(&current).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    stack.push(entry.path());
                }
            }
        } else if current.is_file() && supported_extension(&current) {
            files.push(current);
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_extract_txt() {
        let mut f = NamedTempFile::with_suffix(".txt").unwrap();
        writeln!(f, "Hello, world!").unwrap();
        let text = extract_text(f.path()).await.unwrap();
        assert_eq!(text.trim(), "Hello, world!");
    }

    #[test]
    fn test_supported_extension() {
        assert!(supported_extension(Path::new("test.pdf")));
        assert!(supported_extension(Path::new("test.docx")));
        assert!(supported_extension(Path::new("test.xlsx")));
        assert!(supported_extension(Path::new("test.pptx")));
        assert!(supported_extension(Path::new("test.txt")));
        assert!(supported_extension(Path::new("test.rs")));
        assert!(!supported_extension(Path::new("test.exe")));
        assert!(!supported_extension(Path::new("test.bin")));
    }

    #[tokio::test]
    async fn test_unsupported_format() {
        let f = NamedTempFile::with_suffix(".xyz").unwrap();
        assert!(extract_text(f.path()).await.is_err());
    }

    #[tokio::test]
    async fn test_unsupported_without_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("README");
        tokio::fs::write(&path, b"hi").await.unwrap();
        let err = extract_text(&path).await.unwrap_err();
        assert!(
            err.to_string().contains("Unsupported file format"),
            "got: {err}"
        );
    }

    /// An empty file must short-circuit before any encoding detection.
    #[tokio::test]
    async fn test_extract_empty_file_is_empty_string() {
        let f = NamedTempFile::with_suffix(".txt").unwrap();
        assert_eq!(extract_text(f.path()).await.unwrap(), "");
    }

    /// Non-UTF-8 input goes through `chardetng`; a UTF-16LE BOM is the
    /// one signal the detector can never miss.
    #[tokio::test]
    async fn test_extract_non_utf8_uses_encoding_detection() {
        let mut f = NamedTempFile::with_suffix(".txt").unwrap();
        let mut bytes = vec![0xFF, 0xFE]; // UTF-16LE BOM
        for unit in "Hello, café!".encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        f.write_all(&bytes).unwrap();

        let text = extract_text(f.path()).await.unwrap();
        assert_eq!(text, "Hello, café!");
    }

    /// Plain valid UTF-8 takes the fast path (`str::from_utf8`) and is
    /// returned verbatim, BOM and all.
    #[tokio::test]
    async fn test_extract_valid_utf8_is_verbatim() {
        let mut f = NamedTempFile::with_suffix(".txt").unwrap();
        write!(f, "héllo ✨").unwrap();
        assert_eq!(extract_text(f.path()).await.unwrap(), "héllo ✨");
    }

    /// Case-insensitive and no-extension paths of the extension probe.
    #[test]
    fn test_supported_extension_case_and_missing() {
        assert!(supported_extension(Path::new("ARCHIVE.MD")));
        assert!(!supported_extension(Path::new("Makefile")));
        assert!(!supported_extension(Path::new("archive.unknown")));
    }

    /// Walk: nested directories are traversed depth-first, unsupported
    /// files are ignored, and a plain file root yields itself.
    #[tokio::test]
    async fn test_walk_supported_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        tokio::fs::create_dir_all(root.join("sub/deep"))
            .await
            .unwrap();
        tokio::fs::write(root.join("top.md"), b"a").await.unwrap();
        tokio::fs::write(root.join("sub/nested.rs"), b"b")
            .await
            .unwrap();
        tokio::fs::write(root.join("sub/deep/deep.txt"), b"c")
            .await
            .unwrap();
        tokio::fs::write(root.join("sub/skip.bin"), b"d")
            .await
            .unwrap();
        tokio::fs::write(root.join("Makefile"), b"e").await.unwrap();

        let mut found = walk_supported_files(root).await;
        found.sort();
        let expected = vec![
            root.join("sub/deep/deep.txt"),
            root.join("sub/nested.rs"),
            root.join("top.md"),
        ];
        assert_eq!(found, expected);

        // A supported file passed directly is walked as itself.
        assert_eq!(
            walk_supported_files(&root.join("top.md")).await,
            vec![root.join("top.md")]
        );
        // A directory with nothing supported below it yields no files.
        let empty = root.join("empty");
        tokio::fs::create_dir(&empty).await.unwrap();
        assert!(walk_supported_files(&empty).await.is_empty());
    }

    /// Bytes the detector claims but cannot decode (a UTF-16LE BOM with a
    /// trailing odd byte) must surface as the encoding error, not as text
    /// full of replacement characters.
    #[tokio::test]
    async fn test_extract_undecodable_bytes_errors() {
        let mut f = NamedTempFile::with_suffix(".txt").unwrap();
        let mut bytes = vec![0xFF, 0xFE]; // UTF-16LE BOM
        for unit in "Hello".encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        bytes.push(0x41); // dangling byte: neither decoder can finish it
        f.write_all(&bytes).unwrap();

        let err = extract_text(f.path()).await.unwrap_err();
        assert!(
            err.to_string().contains("Could not detect encoding"),
            "got: {err}"
        );
    }

    /// A directory the walk cannot read contributes nothing and never fails
    /// the walk. Root ignores the mode, so only assert when the chmod stuck.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_walk_skips_unreadable_directories() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let locked = root.join("locked");
        tokio::fs::create_dir(&locked).await.unwrap();
        tokio::fs::write(root.join("open.md"), b"a").await.unwrap();
        tokio::fs::write(locked.join("hidden.txt"), b"b")
            .await
            .unwrap();

        let mode = std::fs::Permissions::from_mode(0o000);
        std::fs::set_permissions(&locked, mode).unwrap();
        let readable = std::fs::read_dir(&locked).is_ok();

        let mut found = walk_supported_files(root).await;
        found.sort();

        // Restore before asserting so tempdir cleanup can unlink the entry.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

        if readable {
            // Running as root the mode is ignored: the walk sees everything.
            assert_eq!(found.len(), 2, "root can still read the directory");
            return;
        }
        assert_eq!(found, vec![root.join("open.md")]);
        assert!(
            !found.iter().any(|p| p.starts_with(&locked)),
            "an unreadable directory must contribute no files"
        );
    }

    /// The extension table is the single source of truth for *both*
    /// [`extract_text`] and [`supported_extension`]; a file the indexer
    /// accepts must always be extractable (and vice versa).
    #[tokio::test]
    async fn test_every_supported_extension_roundtrips() {
        for ext in [
            "txt", "md", "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "json", "yaml",
            "yml", "toml", "xml", "csv", "html", "css",
        ] {
            let mut f = NamedTempFile::with_suffix(format!(".{ext}")).unwrap();
            f.write_all(b"body").unwrap();
            assert!(supported_extension(f.path()), "supported: {ext}");
            assert_eq!(
                extract_text(f.path()).await.unwrap(),
                "body",
                "extract: {ext}"
            );
        }
    }
}
