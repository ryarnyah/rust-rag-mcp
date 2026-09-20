use crate::chunker::Chunker;
use crate::DocumentChunk;
use std::cell::RefCell;
use tree_sitter::{Language, Node, Parser};

thread_local! {
    static PARSER_CACHE: RefCell<Option<(Language, Parser)>> = const { RefCell::new(None) };
}

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
) -> Result<Vec<DocumentChunk>, anyhow::Error> {
    let tree = PARSER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let needs_init = cache.as_ref().is_none_or(|(lang, _)| *lang != language);
        if needs_init {
            let mut parser = Parser::new();
            parser.set_language(&language)?;
            *cache = Some((language, parser));
        }
        let (_, parser) = cache.as_mut().unwrap();
        parser
            .parse(text, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse"))
    })?;

    let mut chunks: Vec<DocumentChunk> = Vec::new();
    let mut stack: Vec<Node> = vec![tree.root_node()];

    while let Some(node) = stack.pop() {
        if node.is_extra() || node.is_error() || node.is_missing() || !node.is_named() {
            continue;
        }

        let kind = node.kind();

        if is_top_level_node(kind) {
            let word_count = count_words(text, node.start_byte(), node.end_byte());

            if word_count > 0 && word_count <= max_chunk_size {
                chunks.push(DocumentChunk {
                    id: String::new(),
                    text: String::new(),
                    source: source.to_string(),
                    chunk_index: chunks.len() as u32,
                    start_offset: node.start_byte(),
                    end_offset: node.end_byte(),
                });
                continue;
            }

            push_children(&mut stack, node);
            continue;
        }

        if node.child_count() > 0 {
            push_children(&mut stack, node);
            continue;
        }

        let start = node.start_byte();
        let end = node.end_byte();
        let word_count = count_words(text, start, end);

        if word_count > 0
            && chunks
                .last()
                .is_none_or(|c: &DocumentChunk| c.end_offset < start)
        {
            chunks.push(DocumentChunk {
                id: String::new(),
                text: String::new(),
                source: source.to_string(),
                chunk_index: chunks.len() as u32,
                start_offset: start,
                end_offset: end,
            });
        }
    }

    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.chunk_index = i as u32;
        chunk.text = text[chunk.start_offset..chunk.end_offset].to_string();
    }

    Ok(chunks)
}

fn push_children<'a>(stack: &mut Vec<Node<'a>>, node: Node<'a>) {
    for i in (0..node.child_count()).rev() {
        if let Some(child) = node.child(i) {
            stack.push(child);
        }
    }
}

fn count_words(text: &str, start: usize, end: usize) -> usize {
    let slice = &text[start..end.min(text.len())];
    let mut count = 0;
    let mut in_word = false;
    for byte in slice.bytes() {
        if byte.is_ascii_whitespace() {
            in_word = false;
        } else if !in_word {
            in_word = true;
            count += 1;
        }
    }
    count
}

fn is_top_level_node(kind: &str) -> bool {
    matches!(
        kind,
        // Root
        "source_file" | "program" | "compilation_unit" | "module_definition" | "document"
        // Rust
            | "function_item" | "function_signature_item" | "struct_item"
            | "enum_item" | "impl_item" | "trait_item" | "type_item"
            | "mod_item" | "macro_definition"
        // Python
            | "decorated_definition"
        // JavaScript / TypeScript
            | "method_definition" | "type_alias_declaration" | "export_statement"
            | "generator_function_declaration" | "arrow_function"
        // Go
            | "function"
        // C / C++
            | "struct_specifier" | "enum_specifier"
            | "preproc_include" | "preproc_def" | "preproc_ifdef"
            | "preproc_ifndef" | "preproc_if" | "preproc_else"
            | "preproc_endif" | "preproc_function_def"
        // Java
            | "record_declaration" | "annotation_type_declaration"
        // C#
            | "namespace_declaration" | "struct_declaration"
        // Ruby
            | "class" | "module" | "method" | "singleton_method"
        // PHP
            | "namespace_definition" | "interface_definition"
        // Scala
            | "object_definition" | "val_definition"
        // HTML
            | "element" | "script_element" | "style_element"
        // CSS
            | "rule_set" | "media_statement" | "keyframes_statement"
        // YAML
            | "block_mapping" | "block_sequence"
        // Lua
            | "local_function_declaration" | "local_variable_declaration"
        // Zig
            | "decl"
        // Elixir
            | "def_module" | "def_function" | "defp_function"
        // Erlang
            | "function_clause"
        // HCL
            | "block"
        // Protobuf
            | "message_definition" | "service_definition" | "enum_definition"
        // CMake
            | "function_def" | "macro_def"
        // Make
            | "rule" | "variable_assignment"
        // Shared across multiple languages (deduplicated)
            | "function_definition" | "function_declaration"
            | "class_definition" | "class_declaration"
            | "interface_declaration" | "enum_declaration"
            | "method_declaration" | "constructor_declaration"
            | "type_definition" | "trait_definition"
            | "variable_declaration" | "attribute"
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
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "cs" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "rb" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        "scala" | "sc" => Some(tree_sitter_scala::LANGUAGE.into()),
        "html" | "htm" => Some(tree_sitter_html::LANGUAGE.into()),
        "css" | "scss" | "less" => Some(tree_sitter_css::LANGUAGE.into()),
        "json" => Some(tree_sitter_json::LANGUAGE.into()),
        "yaml" | "yml" => Some(tree_sitter_yaml::LANGUAGE.into()),
        "lua" => Some(tree_sitter_lua::LANGUAGE.into()),
        "zig" => Some(tree_sitter_zig::LANGUAGE.into()),
        "ex" | "exs" => Some(tree_sitter_elixir::LANGUAGE.into()),
        "erl" | "hrl" => Some(tree_sitter_erlang::LANGUAGE.into()),
        "hcl" | "tf" | "tfvars" => Some(tree_sitter_hcl::LANGUAGE.into()),
        "proto" => Some(tree_sitter_proto::LANGUAGE.into()),
        "sh" | "bash" | "zsh" => Some(tree_sitter_bash::LANGUAGE.into()),
        "cmake" => Some(tree_sitter_cmake::LANGUAGE.into()),
        "mk" | "makefile" | "Makefile" => Some(tree_sitter_make::LANGUAGE.into()),
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

    #[test]
    fn test_cpp_chunking() {
        let code = r#"
class Animal {
public:
    virtual void speak() = 0;
};

class Dog : public Animal {
    void speak() override { printf("woof\n"); }
};
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.cpp");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_java_chunking() {
        let code = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.java");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_go_chunking() {
        let code = r#"
package main

func add(a, b int) int {
    return a + b
}

type Point struct {
    X, Y float64
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.go");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_json_not_split_by_pair() {
        let code = r#"{"name": "Alice", "age": 30, "scores": [1, 2, 3]}"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.json");
        assert_eq!(
            chunks.len(),
            1,
            "JSON object should be one chunk, not split by pair"
        );
        assert_eq!(chunks[0].text, code);
    }

    #[test]
    fn test_html_chunking() {
        let code = r#"
<html>
<body>
  <h1>Title</h1>
  <p>Content</p>
</body>
</html>
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.html");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_lua_chunking() {
        let code = r#"
function greet(name)
    print("Hello, " .. name)
end
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "test.lua");
        assert!(!chunks.is_empty());
    }
}
