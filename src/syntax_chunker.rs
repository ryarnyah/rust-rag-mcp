use crate::chunker::Chunker;
use crate::DocumentChunk;
use anyhow::Result;
use tree_sitter::{Language, Node, Parser};

#[derive(Default)]
pub struct SyntaxChunker {
    word_chunker: Chunker,
}

impl SyntaxChunker {
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            word_chunker: Chunker::new(max_chunk_size, overlap),
        }
    }

    pub fn chunk_text(&self, text: &str, source: &str) -> Vec<DocumentChunk> {
        let ext = source.rsplit('.').next().unwrap_or("").to_lowercase();

        let language = match language_for_extension(&ext) {
            Some(lang) => lang,
            None => return self.word_chunker.chunk_text(text, source),
        };

        match parse_and_chunk(text, source, language, self.word_chunker.max_chunk_size) {
            Ok(chunks) if !chunks.is_empty() => chunks,
            _ => self.word_chunker.chunk_text(text, source),
        }
    }
}

fn parse_and_chunk(
    text: &str,
    source: &str,
    language: Language,
    max_chunk_size: usize,
) -> Result<Vec<DocumentChunk>> {
    let mut parser = Parser::new();
    parser.set_language(&language)?;
    let tree = parser
        .parse(text, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse"))?;

    let mut chunks = Vec::new();
    let mut stack: Vec<(Node, usize)> = vec![(tree.root_node(), 0)];

    while let Some((node, depth)) = stack.pop() {
        if node.is_extra() || node.is_error() || node.is_missing() {
            continue;
        }

        let kind = node.kind();

        if is_top_level_node(kind) {
            let nt = node_text(node, text);
            let trimmed = nt.trim();
            let word_count = trimmed.split_whitespace().count();

            if word_count > 0 && word_count <= max_chunk_size {
                chunks.push(DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    text: nt,
                    source: source.to_string(),
                    chunk_index: chunks.len() as u32,
                    start_offset: node.start_byte(),
                    end_offset: node.end_byte(),
                });
                continue;
            }

            for i in (0..node.child_count()).rev() {
                if let Some(child) = node.child(i) {
                    stack.push((child, depth + 1));
                }
            }
            continue;
        }

        if depth > 0 && node.child_count() > 0 {
            for i in (0..node.child_count()).rev() {
                if let Some(child) = node.child(i) {
                    stack.push((child, depth + 1));
                }
            }
            continue;
        }

        let nt = node_text(node, text);
        let word_count = nt.split_whitespace().count();

        if word_count > 0
            && chunks
                .last()
                .is_none_or(|c: &DocumentChunk| c.end_offset < node.start_byte())
        {
            if word_count <= max_chunk_size {
                chunks.push(DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    text: nt,
                    source: source.to_string(),
                    chunk_index: chunks.len() as u32,
                    start_offset: node.start_byte(),
                    end_offset: node.end_byte(),
                });
            } else {
                for i in (0..node.child_count()).rev() {
                    if let Some(child) = node.child(i) {
                        stack.push((child, depth + 1));
                    }
                }
            }
        }
    }

    Ok(chunks)
}

fn node_text(node: Node, text: &str) -> String {
    text[node.start_byte()..node.end_byte()].to_string()
}

fn is_top_level_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "function_signature_item"
            | "struct_item"
            | "enum_item"
            | "impl_item"
            | "trait_item"
            | "type_item"
            | "mod_item"
            | "macro_definition"
            | "expression_statement"
            | "function_definition"
            | "class_definition"
            | "method_definition"
            | "decorated_definition"
            | "module"
            | "import_statement"
            | "function_declaration"
            | "class_declaration"
            | "method_declaration"
            | "interface_declaration"
            | "type_alias_declaration"
            | "export_statement"
            | "generator_function_declaration"
            | "arrow_function"
            | "function"
            | "type_definition"
            | "func_declaration"
            | "type_spec"
            | "method_spec"
            | "interface_type"
            | "compilation_unit"
            | "source_file"
            | "program"
            | "module_definition"
            | "struct_specification"
            | "interface_specification"
            | "enum_declaration"
            | "component_definition"
            | "signal"
            | "state_machine"
    )
}

pub fn language_for_extension(ext: &str) -> Option<Language> {
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_function_chunking() {
        let code = r#"
use std::io;

fn main() {
    println!("Hello, world!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.rs");
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
            assert!(chunk.start_offset <= chunk.end_offset);
        }
    }

    #[test]
    fn test_python_function_chunking() {
        let code = r#"
def hello():
    print("Hello, world!")

def add(a, b):
    return a + b

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.py");
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_fallback_for_unknown_extension() {
        let text = "hello world this is a test";
        let chunker = SyntaxChunker::new(3, 1);
        let chunks = chunker.chunk_text(text, "test.xyz");
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_empty_code() {
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text("", "test.rs");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_large_function_falls_back() {
        let body: String = (0..1000)
            .map(|i| format!("let x_{i} = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!("fn big() {{\n{body}\n}}");
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(&code, "test.rs");
        assert!(!chunks.is_empty());
    }
}
