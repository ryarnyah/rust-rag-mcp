mod fixtures;

use rust_rag_mcp::chunker::Chunker;
use rust_rag_mcp::docs;
use tempfile::tempdir;

const TEST_FIXTURES_DIR: &str = "../../test-fixtures";

#[test]
fn test_extract_sample_pdf() {
    let pdf_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.pdf");
    if !pdf_path.exists() {
        eprintln!("Skipping - sample.pdf not found");
        return;
    }
    let text = docs::extract_text(&pdf_path).unwrap();
    assert!(!text.is_empty(), "Should extract text from real PDF");
    assert!(
        text.len() > 100,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[test]
fn test_extract_sample_docx() {
    let docx_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.docx");
    if !docx_path.exists() {
        eprintln!("Skipping - sample.docx not found");
        return;
    }
    let text = docs::extract_text(&docx_path).unwrap();
    assert!(!text.is_empty(), "Should extract text from real DOCX");
    assert!(
        text.len() > 50,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[test]
fn test_extract_sample_xlsx() {
    let xlsx_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.xlsx");
    if !xlsx_path.exists() {
        eprintln!("Skipping - sample.xlsx not found");
        return;
    }
    let text = docs::extract_text(&xlsx_path).unwrap();
    assert!(!text.is_empty(), "Should extract text from real XLSX");
    assert!(
        text.len() > 10,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[test]
fn test_extract_sample_ppt() {
    let ppt_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.pptx");
    if !ppt_path.exists() {
        eprintln!("Skipping - sample.pptx not found");
        return;
    }
    let text = docs::extract_text(&ppt_path).unwrap();
    assert!(!text.is_empty(), "Should extract text from real PPT");
    assert!(
        text.len() > 10,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[test]
fn test_extract_txt_file() {
    let dir = tempdir().unwrap();
    let txt_path = dir.path().join("test.txt");
    let sample = fixtures::sample_text();
    std::fs::write(&txt_path, &sample).unwrap();

    let text = docs::extract_text(&txt_path).unwrap();
    assert!(text.contains("Rust is a systems programming language"));
    assert!(text.contains("tokio runtime"));
}

#[test]
fn test_chunker_with_page_sections() {
    let chunker = Chunker::new(10, 2);
    let text = fixtures::sample_text_with_sections();
    let chunks = chunker.chunk_file(&text, "ml-doc.pdf");

    assert!(!chunks.is_empty());
    assert!(chunks.iter().any(|c| c.source.contains("page-1")));
    assert!(chunks.iter().any(|c| c.source.contains("page-2")));
    assert!(chunks.iter().any(|c| c.source.contains("page-3")));
}

#[test]
fn test_chunker_preserves_content() {
    let chunker = Chunker::new(20, 5);
    let text = "The quick brown fox jumps over the lazy dog. A second sentence for testing.";
    let chunks = chunker.chunk_text(text, "test.txt");

    let combined: String = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(combined.contains("quick brown fox"));
    assert!(combined.contains("second sentence"));
}

#[test]
fn test_supported_extensions() {
    assert!(docs::supported_extension(std::path::Path::new("doc.pdf")));
    assert!(docs::supported_extension(std::path::Path::new("doc.docx")));
    assert!(docs::supported_extension(std::path::Path::new("doc.xlsx")));
    assert!(docs::supported_extension(std::path::Path::new("doc.pptx")));
    assert!(docs::supported_extension(std::path::Path::new(
        "readme.txt"
    )));
    assert!(docs::supported_extension(std::path::Path::new("code.rs")));
    assert!(docs::supported_extension(std::path::Path::new("data.json")));
    assert!(!docs::supported_extension(std::path::Path::new(
        "image.png"
    )));
    assert!(!docs::supported_extension(std::path::Path::new(
        "binary.exe"
    )));
}

#[test]
fn test_extract_unsupported_returns_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.xyz");
    std::fs::write(&path, "data").unwrap();
    assert!(docs::extract_text(&path).is_err());
}

#[test]
fn test_chunker_empty_and_whitespace() {
    let chunker = Chunker::default();
    assert!(chunker.chunk_text("", "src.txt").is_empty());
    assert!(chunker.chunk_text("  \n\t  ", "src.txt").is_empty());
}

#[test]
fn test_chunker_single_word() {
    let chunker = Chunker::new(100, 10);
    let chunks = chunker.chunk_text("hello", "test.txt");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "hello");
    assert_eq!(chunks[0].source, "test.txt");
}

#[test]
fn test_chunker_offset_accuracy() {
    let chunker = Chunker::new(5, 0);
    let text = "alpha bravo charlie delta echo";
    let chunks = chunker.chunk_text(text, "test.txt");

    for chunk in &chunks {
        let reconstructed = &text[chunk.start_offset..chunk.end_offset];
        assert_eq!(
            reconstructed, chunk.text,
            "offset mismatch for chunk {}",
            chunk.chunk_index
        );
    }
}
