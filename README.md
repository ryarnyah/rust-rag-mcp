<p align="center">
  <img src="assets/logo.svg" alt="rust-rag-mcp" width="100">
</p>

<h3 align="center">A blazing-fast RAG server with a vector database built from scratch in Rust.</h3>

<p align="center">
  No external vector DB. No API keys. No nonsense.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#cli-reference">CLI</a> •
  <a href="#mcp-integration">MCP Integration</a> •
  <a href="#formal-verification">Formal Verification</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## Why rust-rag-mcp?

Most RAG solutions bolt together a vector database, an embedding service, and a document parser. **rust-rag-mcp** is different:

- **The vector DB is yours.** Custom HNSW index with mmap storage, SIMD-optimized distance computation, and PostgreSQL-style WAL — all ~4,200 lines of Rust, zero external dependencies.
- **Code-aware chunking.** Tree-sitter parses your source into an AST and chunks by semantic boundaries (functions, structs, classes), not arbitrary word counts.
- **Runs 100% locally.** 40+ ONNX embedding models via fastembed. No API keys, no network calls, no vendor lock-in.
- **Crash-safe by design.** WAL with CRC32 integrity checks, idempotent recovery, and a three-step flush protocol — the same guarantees you'd expect from a production database.

## Quick Start

```bash
# Install
cargo install --path .

# Index your project
rust-rag-mcp index ./src/

# Search
rust-rag-mcp search "vector similarity"
```

That's it. No Docker. No external services. Just a single binary.

## Features

### Custom Vector Database

A from-scratch HNSW implementation with:
- **Memory-mapped storage** — zero-copy reads, OS-managed caching, datasets larger than RAM
- **SIMD-optimized cosine distance** — automatic AVX2/AVX-512 vectorization
- **PostgreSQL-style WAL** — LOG-BEFORE-APPLY with crash recovery and segment rotation
- **Soft deletes with compaction** — tombstone tracking with automatic background cleanup

### Hybrid Search (BM25 + HNSW)

Every search runs two retrievers — dense HNSW (paraphrase, semantics) and BM25 lexical
(exact identifiers, rare tokens, acronyms) — and fuses their ranked lists with Reciprocal
Rank Fusion (RRF, `k = 60`). Only *ranks* are fused, so cosine similarity and BM25 weights
never have to be comparable: a document that ranks well under either retriever surfaces.

The BM25 index is a from-scratch inverted index (`bm25` module, Okapi BM25, `k1 = 1.2`,
`b = 0.75`, language-neutral tokenizer). It is managed like the HNSW index: persisted as a
`db.bm25` sidecar with generation tokens, validated at open, maintained incrementally on
index/delete, and rebuilt from the metadata scan when anything mismatches — a rebuild
re-tokenizes stored chunks, it never re-embeds.

Three modes, shared by the MCP `search` tool and the CLI `--mode` flag:

| Mode | Retrievers | Score |
|------|-----------|-------|
| `hybrid` (default) | HNSW + BM25, RRF-fused | RRF weight in `(0, ~0.033]`, monotonic in fused rank |
| `semantic` | HNSW only | cosine similarity in `[0, 1]` |
| `lexical` | BM25 only (no query embedding) | unbounded BM25 weight |

Because an RRF weight is intentionally tiny (it encodes *rank agreement*, not similarity), every hybrid result also carries
`components` — the raw per-retriever score and the 1-based pool rank that were fused:

```json
{
  "score": 0.0325,
  "components": {
    "dense":  { "score": 0.8412, "rank": 1 },
    "lexical": { "score": 7.1325, "rank": 2 }
  }
}
```

`score` is exactly `Σ 1 / (60 + rank)` over the sides present (`1/61 + 1/62` above), so you can reproduce the fused number
and see *why* a document placed where it did — cosine and BM25 stay on their own interpretable scales. A side that is
missing (or `null` in JSON) means the document never entered that retriever's top-50 pool. The CLI prints the same
components inline: `(rrf, score: 0.0325; dense: 0.8412 @1, bm25: 7.1325 @2)`. Semantic and lexical results omit
`components` — their `score` already is the raw value.

### Syntax-Aware Code Chunking

Source code is parsed into an AST and chunked by semantic boundaries:

| Language | Extensions |
|----------|------------|
| Rust | `.rs` |
| Python | `.py` |
| JavaScript | `.js`, `.jsx` |
| TypeScript | `.ts`, `.tsx` |
| Go | `.go` |
| C | `.c`, `.h` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh` |
| Java | `.java` |
| C# | `.cs` |
| Ruby | `.rb` |
| PHP | `.php` |
| Scala | `.scala`, `.sc` |
| HTML | `.html`, `.htm` |
| CSS | `.css`, `.scss`, `.less` |
| JSON | `.json` |
| YAML | `.yaml`, `.yml` |
| Lua | `.lua` |
| Zig | `.zig` |
| Elixir | `.ex`, `.exs` |
| Erlang | `.erl`, `.hrl` |
| HCL | `.hcl`, `.tf`, `.tfvars` |
| Protobuf | `.proto` |
| Bash | `.sh`, `.bash`, `.zsh` |
| CMake | `.cmake` |
| Make | `.mk`, `Makefile` |

Non-code files fall back to word-count chunking.

### Document Indexing

PDF, DOCX, XLSX, PPTX — extract text and index it alongside your source code. Content-hash deduplication ensures re-indexing is fast.

### Local Embeddings

40+ models from BGE, MiniLM, Nomic, Snowflake, Jina, GTE, and more. Runs entirely offline via ONNX.

Asymmetric models are embedded with their official task instructions: queries get `query: ` (E5),
`search_query: ` (Nomic, ModernBERT), a `task: search result | query: ` prompt (EmbeddingGemma), or a
retrieval instruction sentence (BGE, mxbai, Snowflake Arctic); documents get the matching passage
instruction (`passage: `, `search_document: `, `title: none | text: `, ...) where the model's card
requires one — BGE/mxbai/Arctic documents are never prefixed. Symmetric models (MiniLM, MPNet, GTE,
Jina, ...) are embedded verbatim with no prefixes.

**Switching models or upgrading across prefix-policy changes requires a fresh re-index.**

## CLI Reference

### Start MCP Server

```bash
rust-rag-mcp serve [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--db-path <PATH>` | `.rag-db` | Database path |
| `--cache-path <PATH>` | `.rag-cache` | Cache path |
| `--model <MODEL>` | `Xenova/bge-small-en-v1.5` | Embedding model |
| `--chunk-size <N>` | `512` | Chunk size |
| `--overlap <N>` | `64` | Chunk overlap |

### Index Documents

```bash
rust-rag-mcp index <PATHS>... [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--db-path <PATH>` | `.rag-db` | Database path |
| `--cache-path <PATH>` | `.rag-cache` | Cache path |
| `--model <MODEL>` | `Xenova/bge-small-en-v1.5` | Embedding model |
| `--chunk-size <N>` | `512` | Chunk size |
| `--overlap <N>` | `64` | Chunk overlap |

```bash
# Index a Rust project
rust-rag-mcp index ./src/

# Index specific source files
rust-rag-mcp index main.py lib.rs utils.js

# Index mixed documents and code
rust-rag-mcp index ./docs/ ./src/ report.pdf
```

### Search

```bash
rust-rag-mcp search <QUERY>... [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--top-k <N>` | `5` | Number of results |
| `--mode <MODE>` | `hybrid` | `hybrid` (HNSW + BM25, RRF), `semantic`, or `lexical` (BM25 only, skips embedding) |
| `--db-path <PATH>` | `.rag-db` | Database path |
| `--cache-path <PATH>` | `.rag-cache` | Cache path |
| `--model <MODEL>` | `Xenova/bge-small-en-v1.5` | Embedding model |

### Delete Source

```bash
rust-rag-mcp delete <SOURCE_PATH> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--db-path <PATH>` | `.rag-db` | Database path |
| `--cache-path <PATH>` | `.rag-cache` | Cache path |
| `--model <MODEL>` | `Xenova/bge-small-en-v1.5` | Embedding model |

### Other Commands

```bash
rust-rag-mcp sources [OPTIONS]    # List indexed sources
rust-rag-mcp stats [OPTIONS]      # Show database statistics
rust-rag-mcp models               # List available embedding models
```

`sources` and `stats` accept `--db-path`, `--cache-path`, and `--model` options (same defaults as above).

## MCP Integration

### opencode

```json
{
  "mcp": {
    "rag": {
      "command": [
        "rust-rag-mcp",
        "serve",
        "--db-path", "/home/user/.rag-db",
        "--cache-path", "/home/user/.rag-db",
      ]
    }
  }
}
```

Add to `~/.config/opencode/opencode.json`.

### Claude Desktop

```json
{
  "mcpServers": {
    "rag": {
      "command": "rust-rag-mcp",
      "args": ["serve", "--db-path", ".rag-db"]
    }
  }
}
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`.

### VS Code (GitHub Copilot)

```json
{
  "servers": {
    "rag": {
      "command": "rust-rag-mcp",
      "args": ["serve", "--db-path", ".rag-db"]
    }
  }
}
```

Add to `.vscode/mcp.json`.

## Configuration

Set log level via `RUST_LOG`:

```bash
RUST_LOG=info rust-rag-mcp serve
```

LanceDB logs are set to `warn` by default to reduce noise.

## Architecture

```
rust-rag-mcp
├── r_vector    HNSW index + mmap storage (2,000+ lines)
├── wal         PostgreSQL-style write-ahead log
├── bm25        BM25 lexical index + RRF hybrid fusion
├── syntax_chunker  Tree-sitter AST-aware chunking (25 languages)
├── embeddings  fastembed ONNX local embeddings
├── docs        PDF/DOCX/XLSX/PPTX text extraction
├── rag         Core orchestrator + metadata index
└── mcp         MCP server (8 tools over stdio)
```

All ~4,200 lines of the vector database are written from scratch. No wrappers. No hidden dependencies.

## License

Apache-2.0
