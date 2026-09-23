use crate::DocumentChunk;
use crate::chunker::Chunker;
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

        match parse_and_chunk(
            text,
            source,
            language,
            self.word_chunker.max_chunk_size,
            self.word_chunker.overlap,
        ) {
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
    overlap: usize,
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
    // Stack: (node, context_prefix) — context is only set for direct children
    // of an oversized node, never leaked to unrelated siblings.
    let mut stack: Vec<(Node, Option<(usize, usize)>)> = vec![(tree.root_node(), None)];

    while let Some((node, ctx)) = stack.pop() {
        if node.is_error() || node.is_missing() {
            continue;
        }

        // Comments are marked as "extra" by tree-sitter but are valuable for RAG
        if node.is_extra() {
            if is_comment_node(node.kind()) {
                let word_count = count_words(text, node.start_byte(), node.end_byte());
                if word_count >= 2 {
                    chunks.push(DocumentChunk {
                        id: String::new(),
                        text: String::new(),
                        source: source.to_string(),
                        chunk_index: chunks.len() as u32,
                        start_offset: node.start_byte(),
                        end_offset: node.end_byte(),
                    });
                }
            }
            continue;
        }

        if !node.is_named() {
            continue;
        }

        let kind = node.kind();
        let prev_end = chunks.last().map(|c| c.end_offset).unwrap_or(0);

        if is_top_level_node(kind) {
            let word_count = count_words(text, node.start_byte(), node.end_byte());

            if word_count > 0 && word_count <= max_chunk_size {
                let olap = compute_overlap_range(text, node.start_byte(), overlap);
                let (cs, ce) = merge_ranges(ctx, olap, node.start_byte(), node.end_byte(), prev_end);

                chunks.push(DocumentChunk {
                    id: String::new(),
                    text: String::new(),
                    source: source.to_string(),
                    chunk_index: chunks.len() as u32,
                    start_offset: cs,
                    end_offset: ce,
                });
                continue;
            }

            // Oversized top-level node: emit direct children with this node's
            // header as context. Do NOT propagate ctx to the stack — only the
            // immediate children of this oversized node get the header prefix.
            let header_end = find_header_end(text, node.start_byte(), node.end_byte());
            let this_ctx = (node.start_byte(), header_end);

            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.is_error() || child.is_missing() || !child.is_named() {
                        continue;
                    }
                    let cw = count_words(text, child.start_byte(), child.end_byte());
                    if cw == 0 {
                        continue;
                    }

                    let child_prev = chunks.last().map(|c| c.end_offset).unwrap_or(0);
                    if cw <= max_chunk_size {
                        let olap = compute_overlap_range(text, child.start_byte(), overlap);
                        let (cs, ce) = merge_ranges(
                            Some(this_ctx),
                            olap,
                            child.start_byte(),
                            child.end_byte(),
                            child_prev,
                        );
                        chunks.push(DocumentChunk {
                            id: String::new(),
                            text: String::new(),
                            source: source.to_string(),
                            chunk_index: chunks.len() as u32,
                            start_offset: cs,
                            end_offset: ce,
                        });
                    } else if is_top_level_node(child.kind()) {
                        // Child is a top-level node but oversized: recurse
                        // without context — it will compute its own header.
                        stack.push((child, None));
                    } else {
                        // Child is NOT a top-level node (e.g. a block with statements).
                        // AST decomposition would produce meaningless leaf tokens.
                        // Instead, split the child's text range by word count,
                        // prefixed with the parent's header as context.
                        let sub_chunks = split_by_words(
                            text,
                            child.start_byte(),
                            child.end_byte(),
                            source,
                            max_chunk_size,
                            overlap,
                            Some(this_ctx),
                            child_prev,
                        );
                        chunks.extend(sub_chunks);
                    }
                }
            }
            continue;
        }

        if node.child_count() > 0 {
            // Intermediate AST node: propagate inherited ctx to children.
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    stack.push((child, ctx));
                }
            }
            continue;
        }

        // Leaf node
        let start = node.start_byte();
        let end = node.end_byte();
        let word_count = count_words(text, start, end);

        if word_count < 2 {
            continue;
        }

        if chunks.last().is_some_and(|c| c.end_offset >= start) {
            continue;
        }

        let olap = compute_overlap_range(text, start, overlap);
        let (cs, ce) = merge_ranges(ctx, olap, start, end, prev_end);

        chunks.push(DocumentChunk {
            id: String::new(),
            text: String::new(),
            source: source.to_string(),
            chunk_index: chunks.len() as u32,
            start_offset: cs,
            end_offset: ce,
        });
    }

    merge_small_chunks(&mut chunks, text, max_chunk_size);

    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.chunk_index = i as u32;
        chunk.text = if chunk.start_offset < chunk.end_offset && chunk.end_offset <= text.len() {
            text[chunk.start_offset..chunk.end_offset].to_string()
        } else {
            String::new()
        };
    }

    Ok(chunks)
}

/// Merge optional ranges (context, overlap) with the primary node range.
/// Context must not start before prev_end (no leaking headers from siblings).
/// Overlap is allowed to start before prev_end — that's intentional repetition.
fn merge_ranges(
    ctx: Option<(usize, usize)>,
    overlap: Option<(usize, usize)>,
    node_start: usize,
    node_end: usize,
    prev_end: usize,
) -> (usize, usize) {
    let mut start = node_start;

    // Overlap can go before prev_end — that's the point of overlap
    if let Some((os, _oe)) = overlap
        && os < start
    {
        start = os;
    }

    // Context must not start before prev_end (would pull in unrelated sibling text)
    if let Some((cs, _ce)) = ctx
        && cs < start
        && cs >= prev_end
    {
        start = cs;
    }

    (start, node_end)
}

/// Compute the overlap byte range: the N words before `before_byte`.
fn compute_overlap_range(text: &str, before_byte: usize, overlap: usize) -> Option<(usize, usize)> {
    if overlap == 0 || before_byte == 0 {
        return None;
    }

    let mut words_found = 0;
    let mut in_word = false;
    let mut last_word_end = before_byte;
    let mut i = before_byte;

    while i > 0 {
        i -= 1;
        let ch = text.as_bytes()[i];
        if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
            if in_word {
                words_found += 1;
                if words_found > overlap {
                    return Some((last_word_end, before_byte));
                }
            }
            in_word = false;
        } else {
            if !in_word {
                last_word_end = i + 1;
            }
            in_word = true;
        }
    }

    if words_found > 0 {
        Some((0, before_byte))
    } else {
        None
    }
}

/// Merge adjacent chunks that are too small into a single meaningful chunk.
fn merge_small_chunks(chunks: &mut Vec<DocumentChunk>, text: &str, max_chunk_size: usize) {
    if chunks.len() <= 1 {
        return;
    }

    let min_words = 4;
    let mut merged: Vec<DocumentChunk> = Vec::new();

    for chunk in chunks.drain(..) {
        if chunk.start_offset >= chunk.end_offset {
            continue;
        }
        let word_count = count_words(text, chunk.start_offset, chunk.end_offset);

        if let Some(last) = merged.last_mut() {
            // If this chunk is tiny and adjacent to the previous one, merge them
            let gap = chunk.start_offset.saturating_sub(last.end_offset);
            let combined_words = count_words(text, last.start_offset, chunk.end_offset);

            if (word_count < min_words || count_words(text, last.start_offset, last.end_offset) < min_words)
                && gap <= 2 // allow at most 2 bytes gap (whitespace/newline)
                && combined_words <= max_chunk_size
            {
                last.end_offset = chunk.end_offset;
                continue;
            }
        }

        merged.push(chunk);
    }

    *chunks = merged;
}

/// Split a text range into word-count-based chunks, with optional context prefix.
/// Used when AST decomposition would produce meaningless leaf tokens.
#[allow(clippy::too_many_arguments)]
fn split_by_words(
    text: &str,
    base_offset: usize,
    end_offset: usize,
    source: &str,
    max_chunk_size: usize,
    overlap: usize,
    ctx: Option<(usize, usize)>,
    prev_end: usize,
) -> Vec<DocumentChunk> {
    if base_offset >= end_offset || end_offset > text.len() {
        return Vec::new();
    }
    let slice = &text[base_offset..end_offset];
    let words: Vec<&str> = slice.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    // Precompute byte start offset for each word
    let mut word_starts: Vec<usize> = Vec::with_capacity(words.len());
    let mut pos = 0;
    for w in &words {
        word_starts.push(pos);
        pos += w.len();
        while pos < slice.len() && slice.as_bytes()[pos].is_ascii_whitespace() {
            pos += 1;
        }
    }

    let mut chunks = Vec::new();
    let mut start_word = 0;

    while start_word < words.len() {
        let end_word = (start_word + max_chunk_size).min(words.len());

        let byte_pos = word_starts[start_word];
        let mut word_end_pos = word_starts[start_word];
        for w in &words[start_word..end_word] {
            word_end_pos += w.len();
            while word_end_pos < slice.len()
                && slice.as_bytes()[word_end_pos].is_ascii_whitespace()
            {
                word_end_pos += 1;
            }
        }

        let chunk_text_start = base_offset + byte_pos;
        let chunk_text_end = base_offset + word_end_pos;

        let chunk_start = ctx
            .filter(|&(cs, _)| cs < chunk_text_start && cs >= prev_end)
            .map(|(cs, _)| cs)
            .unwrap_or(chunk_text_start);

        chunks.push(DocumentChunk {
            id: String::new(),
            text: String::new(),
            source: source.to_string(),
            chunk_index: chunks.len() as u32,
            start_offset: chunk_start,
            end_offset: chunk_text_end,
        });

        if end_word >= words.len() {
            break;
        }
        start_word += max_chunk_size - overlap;
    }

    chunks
}

/// Find the end of the "header" line(s) of a top-level node.
/// For a Java class: `public class Foo {` or a method: `public int add(int a, int b) {`
/// Returns the byte offset just after the first `{` on the header line.
fn find_header_end(text: &str, start: usize, end: usize) -> usize {
    if start >= end || start >= text.len() {
        return end;
    }
    let slice = &text[start..end.min(text.len())];
    // Find the first '{' which ends the declaration header
    if let Some(pos) = slice.find('{') {
        start + pos + 1
    } else {
        // No brace found (e.g. pure declaration); return end of first line
        slice
            .find('\n')
            .map(|pos| start + pos + 1)
            .unwrap_or(end)
    }
}

fn count_words(text: &str, start: usize, end: usize) -> usize {
    if start >= end || start >= text.len() {
        return 0;
    }
    let end = end.min(text.len());
    let slice = &text.as_bytes()[start..end];
    let mut count = 0;
    let mut in_word = false;
    for &b in slice {
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            in_word = false;
        } else if !in_word {
            in_word = true;
            count += 1;
        }
    }
    count
}

fn is_comment_node(kind: &str) -> bool {
    matches!(
        kind,
        "line_comment" | "block_comment" | "comment" | "html_comment" | "doc_comment"
    )
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
            | "field_declaration" | "static_initializer" | "initializer"
            | "import_declaration" | "package_declaration"
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
package com.example.calculator;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "Calculator.java");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_java_multiple_methods() {
        let code = r#"
package com.example.order;

import java.math.BigDecimal;
import java.util.List;

public class OrderService {
    public Order createOrder(Customer customer, List<Item> items) {
        Order order = new Order(customer);
        for (Item item : items) {
            order.addItem(item);
        }
        return order;
    }

    public void cancelOrder(Order order) {
        order.setStatus(Status.CANCELLED);
        notifyCustomer(order.getCustomer());
    }

    public BigDecimal calculateTotal(Order order) {
        BigDecimal total = BigDecimal.ZERO;
        for (Item item : order.getItems()) {
            total = total.add(item.getPrice());
        }
        return total;
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "OrderService.java");
        // Each method should be its own chunk, plus the class
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let has_create = chunk_texts.iter().any(|t| t.contains("createOrder"));
        let has_cancel = chunk_texts.iter().any(|t| t.contains("cancelOrder"));
        let has_calculate = chunk_texts.iter().any(|t| t.contains("calculateTotal"));
        assert!(has_create, "Should have createOrder chunk");
        assert!(has_cancel, "Should have cancelOrder chunk");
        assert!(has_calculate, "Should have calculateTotal chunk");
    }

    #[test]
    fn test_java_inheritance() {
        let code = r#"
package com.example.repository;

import java.util.List;

public interface Repository<T> {
    T findById(Long id);
    List<T> findAll();
    void save(T entity);
    void delete(T entity);
}
"#;
        let code_impl = r#"
package com.example.repository;

import org.springframework.jdbc.core.JdbcTemplate;
import java.util.List;

public class UserRepository implements Repository<User> {
    private final JdbcTemplate jdbc;

    @Override
    public User findById(Long id) {
        return jdbc.queryForObject("SELECT * FROM users WHERE id = ?", User.class, id);
    }

    @Override
    public List<User> findAll() {
        return jdbc.query("SELECT * FROM users", User.class);
    }

    @Override
    public void save(User user) {
        jdbc.update("INSERT INTO users (name, email) VALUES (?, ?)",
            user.getName(), user.getEmail());
    }

    @Override
    public void delete(User entity) {
        jdbc.update("DELETE FROM users WHERE id = ?", entity.getId());
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);

        let chunks_if = chunker.chunk_text(code, "Repository.java");
        let texts_if: Vec<&str> = chunks_if.iter().map(|c| c.text.as_str()).collect();
        assert!(texts_if.iter().any(|t| t.contains("interface Repository")),
            "Should have interface declaration");

        let chunks_impl = chunker.chunk_text(code_impl, "UserRepository.java");
        let texts_impl: Vec<&str> = chunks_impl.iter().map(|c| c.text.as_str()).collect();
        assert!(texts_impl.iter().any(|t| t.contains("class UserRepository")),
            "Should have class declaration");
    }

    #[test]
    fn test_java_inner_class() {
        let code = r#"
package com.example.collections;

import java.util.NoSuchElementException;

public class LinkedList {
    private Node head;

    private static class Node {
        int value;
        Node next;

        Node(int value) {
            this.value = value;
        }
    }

    public void addFirst(int value) {
        Node newNode = new Node(value);
        newNode.next = head;
        head = newNode;
    }

    public int removeFirst() {
        if (head == null) throw new NoSuchElementException("Empty list");
        int val = head.value;
        head = head.next;
        return val;
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "LinkedList.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert!(chunk_texts.iter().any(|t| t.contains("class Node")),
            "Should have inner class Node chunk");
        assert!(chunk_texts.iter().any(|t| t.contains("addFirst")),
            "Should have addFirst chunk");
        assert!(chunk_texts.iter().any(|t| t.contains("removeFirst")),
            "Should have removeFirst chunk");
    }

    #[test]
    fn test_java_enum() {
        let code = r#"
package com.example.model;

public enum Status {
    ACTIVE("Active"),
    INACTIVE("Inactive"),
    DELETED("Deleted");

    private final String label;

    Status(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "Status.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert!(chunk_texts.iter().any(|t| t.contains("enum Status")),
            "Should have enum declaration chunk");
    }

    #[test]
    fn test_java_large_class_with_overlap() {
        // A large class where the class node exceeds max_chunk_size,
        // so it splits into child chunks. Verify overlap context is included.
        let methods: String = (0..20)
            .map(|i| {
                format!(
                    "    public void method_{i}(int a, int b, int c) {{\n        int result = a + b + c + {i};\n        System.out.println(result);\n    }}"
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let code = format!("package com.example;\n\npublic class ManyMethods {{\n{methods}\n}}");
        // Each method is ~17 words; max=20 so each fits, but class (~340 words) is split
        let chunker = SyntaxChunker::new(20, 4);
        let chunks = chunker.chunk_text(&code, "ManyMethods.java");
        assert!(chunks.len() > 1, "Should produce multiple chunks, got {}", chunks.len());
        // Verify that child chunks of the split class include the class header
        // as context prefix (oversized node split)
        let has_class_header = chunks.iter().any(|c| c.text.contains("class ManyMethods"));
        assert!(has_class_header, "At least one chunk should contain class header context");
    }

    #[test]
    fn test_java_comments_indexed() {
        let code = r#"
package com.example.math;

/**
 * Calculates the sum of two integers.
 * @param a the first operand
 * @param b the second operand
 * @return the sum of a and b
 */
public int add(int a, int b) {
    return a + b;
}
// This is a single-line comment explaining the logic
public int multiply(int a, int b) {
    return a * b;
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "MathUtils.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let has_javadoc = chunk_texts.iter().any(|t| t.contains("@param"));
        let has_line_comment = chunk_texts.iter().any(|t| t.contains("single-line comment"));
        assert!(has_javadoc, "Javadoc comment should be indexed");
        assert!(has_line_comment, "Line comment should be indexed");
    }

    #[test]
    fn test_java_static_initializer() {
        let code = r#"
package com.example.config;

import java.util.HashMap;
import java.util.Map;

public class Config {
    private static final Map<String, String> DEFAULTS;

    static {
        DEFAULTS = new HashMap<>();
        DEFAULTS.put("host", "localhost");
        DEFAULTS.put("port", "8080");
    }

    public String get(String key) {
        return DEFAULTS.getOrDefault(key, "");
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "Config.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert!(chunk_texts.iter().any(|t| t.contains("static {") || t.contains("static{")),
            "Static initializer should be indexed");
    }

    #[test]
    fn test_java_fields() {
        let code = r#"
package com.example.model;

import java.time.Instant;

public class User {
    private Long id;
    private String name;
    private String email;
    private Instant createdAt;
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "User.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let has_fields = chunk_texts.iter().any(|t| t.contains("private"));
        assert!(has_fields, "Fields should be indexed as chunks");
    }

    #[test]
    fn test_package_declaration_not_fragmented() {
        let code = r#"
package com.github.ryarnyah.rag;

import java.util.List;
import java.util.Map;

public class MyClass {
    public void doStuff() {}
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "MyClass.java");
        // The package declaration must be one chunk — "ryarnyah" must NOT
        // appear as a standalone chunk.
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        for text in &chunk_texts {
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.len() == 1 && words[0] == "ryarnyah" {
                panic!("'ryarnyah' must not be a standalone chunk: {:?}", chunk_texts);
            }
        }
        // The full package line should be present as one chunk
        assert!(
            chunk_texts.iter().any(|t| t.contains("package com.github.ryarnyah.rag")),
            "Full package declaration should be a single chunk"
        );
    }

    #[test]
    fn test_import_declarations_not_fragmented() {
        let code = r#"
package com.example;

import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;

public class Controller {
    @GetMapping("/hello")
    public String hello() { return "hi"; }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "Controller.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        // Each import should be its own chunk, not split into fragments
        for text in &chunk_texts {
            let words: Vec<&str> = text.split_whitespace().collect();
            // No single-word chunks like "GetMapping" or "annotation"
            if words.len() == 1 {
                panic!(
                    "Single-word chunk found '{}': {:?}",
                    words[0], chunk_texts
                );
            }
        }
        assert!(
            chunk_texts.iter().any(|t| t.contains("import java.util.List")),
            "Import java.util.List should be intact"
        );
        assert!(
            chunk_texts.iter().any(|t| t.contains("import org.springframework")),
            "Import org.springframework should be intact"
        );
    }

    #[test]
    fn test_overlap_provides_context() {
        // Two consecutive functions; verify that the second chunk's start_offset
        // is before its node start (i.e., overlap is applied)
        let code = r#"
fn first() {
    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
}

fn second() {
    let x = 10;
    let y = 20;
    let z = 30;
    let w = 40;
}
"#;
        // Each function is ~14 words; use max=20 so each fits as one chunk
        let chunker = SyntaxChunker::new(20, 4);
        let chunks = chunker.chunk_text(code, "test.rs");
        assert_eq!(chunks.len(), 2, "Should produce 2 chunks, got {}", chunks.len());
        // The second chunk should start before the 'fn second()' node
        // because overlap words should pull in some preceding text
        let second_fn_start = code.find("fn second()").unwrap();
        assert!(
            chunks[1].start_offset < second_fn_start,
            "Second chunk should have overlap prefix: start_offset={} < fn second at {}",
            chunks[1].start_offset,
            second_fn_start
        );
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

    // ── RAG-reliability tests ──────────────────────────────────────────

    #[test]
    fn test_chunks_not_split_mid_statement() {
        // Verify no chunk contains a partial statement (unbalanced braces)
        let code = r#"
package com.example.parser;

public class Parser {
    public void parse(String input) {
        if (input == null) {
            throw new IllegalArgumentException("input is null");
        }
        for (char c : input.toCharArray()) {
            process(c);
        }
        System.out.println("done");
    }

    private void process(char c) {
        // process
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "Parser.java");
        for chunk in &chunks {
            let open = chunk.text.matches('{').count();
            let close = chunk.text.matches('}').count();
            assert_eq!(
                open, close,
                "Unbalanced braces in chunk: '{}' (open={}, close={})",
                &chunk.text[..chunk.text.len().min(80)],
                open,
                close
            );
        }
    }

    #[test]
    fn test_chunks_are_self_contained() {
        // Each chunk should contain meaningful, complete constructs
        let code = r#"
package com.example.service;

import java.util.Optional;

public class UserService {
    /**
     * Find a user by their unique identifier.
     */
    public User findById(Long id) {
        return repository.findById(id);
    }

    /**
     * Create a new user account.
     */
    public User createUser(String name, String email) {
        User user = new User(name, email);
        return repository.save(user);
    }

    /**
     * Delete a user account permanently.
     */
    public void deleteUser(Long id) {
        repository.deleteById(id);
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "UserService.java");
        // Each chunk should have non-trivial content
        for chunk in &chunks {
            let word_count = chunk.text.split_whitespace().count();
            assert!(
                word_count >= 2,
                "Chunk too small ({} words): '{}'",
                word_count,
                &chunk.text[..chunk.text.len().min(60)]
            );
        }
    }

    #[test]
    fn test_method_chunks_include_signature() {
        // Method-level chunks should include the method name so a search for
        // "findById" matches the chunk
        let code = r#"
package com.example.repository;

import java.util.List;
import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;

public class UserRepository {
    @PersistenceContext
    private EntityManager em;

    public User findById(Long id) {
        return em.find(User.class, id);
    }

    public List<User> findByEmail(String email) {
        return em.createQuery("SELECT u FROM User u WHERE u.email = :email")
            .setParameter("email", email)
            .getResultList();
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "UserRepository.java");
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        // Search for method name should match a chunk
        assert!(
            chunk_texts.iter().any(|t| t.contains("findById")),
            "Chunk containing 'findById' should exist for search"
        );
        assert!(
            chunk_texts.iter().any(|t| t.contains("findByEmail")),
            "Chunk containing 'findByEmail' should exist for search"
        );
    }

    #[test]
    fn test_large_method_split_preserves_context() {
        // A very large method should be split but child chunks should have
        // the class/method header as context
        let body_lines: String = (0..200)
            .map(|i| format!("        process(\"item_{i}\");"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!(
            "package com.example.batch;\n\
             \n\
             import java.util.List;\n\
             \n\
             public class BatchProcessor {{\n\
             \n\
             /**\n\
              * Process all items in the batch.\n\
              */\n\
             public void processBatch(List<String> items) {{\n\
             {body_lines}\n\
             }}\n\
             }}"
        );
        // Method is ~600+ words; max=10 forces split into many small chunks
        let chunker = SyntaxChunker::new(10, 2);
        let chunks = chunker.chunk_text(&code, "BatchProcessor.java");
        assert!(chunks.len() > 1, "Should split large method into multiple chunks, got {}", chunks.len());
        // At least one chunk should contain the class header
        assert!(
            chunks.iter().any(|c| c.text.contains("class BatchProcessor")),
            "Split chunks should include class header context"
        );
    }

    #[test]
    fn test_offset_reconstruction_accuracy() {
        // Every chunk's text should exactly match the source slice
        let code = r#"
package com.example.calculator;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "Calculator.java");
        for chunk in &chunks {
            let reconstructed = &code[chunk.start_offset..chunk.end_offset];
            assert_eq!(
                reconstructed, chunk.text,
                "Chunk text doesn't match source offset for chunk_index={}",
                chunk.chunk_index
            );
        }
    }

    #[test]
    fn test_no_duplicate_overlap_content() {
        // When overlap is applied, the same source text should not appear twice
        // within the same chunk
        let code = r#"
fn alpha() {
    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
    let e = 5;
}

fn beta() {
    let x = 10;
    let y = 20;
    let z = 30;
    let w = 40;
    let v = 50;
}
"#;
        let chunker = SyntaxChunker::new(512, 10);
        let chunks = chunker.chunk_text(code, "test.rs");
        for chunk in &chunks {
            // Count occurrences of "let a" — should be at most 1
            let count = chunk.text.matches("let a = 1").count();
            assert!(
                count <= 1,
                "Overlap should not cause duplicate content within a chunk: found {} occurrences",
                count
            );
        }
    }

    #[test]
    fn test_java_searchable_by_annotation() {
        // Annotations should be findable via search
        let code = r#"
package com.example.web;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
public class UserController {
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public User createUser(@RequestBody CreateUserRequest request) {
        return userService.createUser(request.getName(), request.getEmail());
    }
}
"#;
        let chunker = SyntaxChunker::new(512, 64);
        let chunks = chunker.chunk_text(code, "UserController.java");
        let all_text = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_text.contains("@RestController"), "Annotations should be in chunks");
        assert!(all_text.contains("@GetMapping"), "Method annotations should be in chunks");
        assert!(all_text.contains("UserController"), "Class name should be in chunks");
    }

    #[test]
    fn test_overlap_word_count_matches_config() {
        // Verify overlap works: each function is a standalone chunk with preceding
        // text overlap. Use max_chunk_size large enough that each function fits.
        let code = r#"
fn first() {
    let a = 1; let b = 2; let c = 3; let d = 4;
    let e = 5; let f = 6; let g = 7; let h = 8;
}

fn second() {
    let x = 10; let y = 20; let z = 30; let w = 40;
    let v = 50; let u = 60; let t = 70; let s = 80;
}
"#;
        // Each function has ~36 words; use max=50 so each fits as one chunk
        let overlap_words = 4;
        let chunker = SyntaxChunker::new(50, overlap_words);
        let chunks = chunker.chunk_text(code, "test.rs");
        assert_eq!(chunks.len(), 2, "Should produce 2 chunks, got {}", chunks.len());
        // The overlap prefix should have approximately `overlap_words` words
        let second_fn_byte = code.find("fn second()").unwrap();
        let prefix = &code[chunks[1].start_offset..second_fn_byte];
        let prefix_word_count = prefix.split_whitespace().count();
        assert!(
            prefix_word_count >= overlap_words / 2,
            "Overlap prefix should have ~{} words, got {}",
            overlap_words,
            prefix_word_count
        );
    }

    #[test]
    fn test_rust_file_with_many_imports_no_crash() {
        // Reproduces a crash where split_by_words overshot past the child
        // node's end boundary when processing non-top-level oversized nodes.
        // The source_file is oversized, its children (use statements) are
        // also oversized with max_chunk_size=3, triggering the word-split path.
        let code = r#"
use crate::IndexResult;
use crate::docs;
use crate::rag::RagCore;
use crate::schemar_ext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, ErrorData, NumberOrString, ProgressNotificationParam,
};

pub struct RagServer;
"#;
        let chunker = SyntaxChunker::new(3, 1);
        let chunks = chunker.chunk_text(code, "mcp.rs");
        assert!(!chunks.is_empty(), "Should produce chunks");
        for chunk in &chunks {
            assert!(
                chunk.start_offset <= chunk.end_offset,
                "Invalid range: start={} end={}",
                chunk.start_offset,
                chunk.end_offset
            );
            assert!(
                chunk.end_offset <= code.len(),
                "end_offset {} exceeds text length {}",
                chunk.end_offset,
                code.len()
            );
            assert_eq!(
                &code[chunk.start_offset..chunk.end_offset],
                chunk.text,
                "Chunk text doesn't match source"
            );
        }
    }

    #[test]
    fn test_large_java_file_no_crash() {
        // A realistic Java file with many imports and a large class body
        // that forces the non-top-level oversized split path.
        let imports: String = (0..50)
            .map(|i| format!("import com.example.package{i}.Class{i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let methods: String = (0..30)
            .map(|i| {
                format!(
                    "    public void method_{i}(int a, int b) {{\n        int result = a + b + {i};\n        System.out.println(result);\n    }}"
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let code = format!("package com.example;\n\n{imports}\n\npublic class LargeClass {{\n{methods}\n}}");

        let chunker = SyntaxChunker::new(10, 2);
        let chunks = chunker.chunk_text(&code, "LargeClass.java");
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                chunk.start_offset <= chunk.end_offset,
                "Invalid range: start={} end={}",
                chunk.start_offset,
                chunk.end_offset
            );
            assert!(
                chunk.end_offset <= code.len(),
                "end_offset {} exceeds text length {}",
                chunk.end_offset,
                code.len()
            );
        }
        // Should have multiple chunks for 30 methods
        assert!(chunks.len() > 5, "Should produce many chunks, got {}", chunks.len());
    }
}
