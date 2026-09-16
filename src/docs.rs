use anyhow::{Context, Result};
use std::path::Path;

pub async fn extract_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.to_lowercase().as_str() {
        "pdf" => unpdf::extract_text(path).context("extracting PDF"),
        "docx" | "xlsx" | "pptx" => undoc::extract_text(path).context("extracting Office document"),
        "txt" | "md" | "rs" | "py" | "js" | "ts" | "go" | "java" | "c" | "cpp" | "h" | "json"
        | "yaml" | "yml" | "toml" | "xml" | "csv" | "html" | "css" => {
            tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("reading {}", path.display()))
        }
        _ => Err(anyhow::anyhow!("Unsupported file format: {}", ext)),
    }
}

pub fn supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            matches!(
                ext.to_lowercase().as_str(),
                "pdf"
                    | "docx"
                    | "xlsx"
                    | "pptx"
                    | "txt"
                    | "md"
                    | "rs"
                    | "py"
                    | "js"
                    | "ts"
                    | "go"
                    | "java"
                    | "c"
                    | "cpp"
                    | "h"
                    | "json"
                    | "yaml"
                    | "yml"
                    | "toml"
                    | "xml"
                    | "csv"
                    | "html"
                    | "css"
            )
        })
        .unwrap_or(false)
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
}
