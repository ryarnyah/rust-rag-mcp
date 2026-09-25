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

> **Switching models or upgrading across prefix-policy changes requires a fresh re-index.**
> Vectors are only compatible when model *and* prefix policy match what they were indexed with;
> content-hash dedup will not re-embed unchanged files on its own. Delete the old `.rag-db` (or run
> `delete-source` + re-`index`) when changing `--model`.

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
| `--source <SOURCE>` | — | Filter by source |
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
├── syntax_chunker  Tree-sitter AST-aware chunking (25 languages)
├── embeddings  fastembed ONNX local embeddings
├── docs        PDF/DOCX/XLSX/PPTX text extraction
├── rag         Core orchestrator + metadata index
└── mcp         MCP server (8 tools over stdio)
```

All ~4,200 lines of the vector database are written from scratch. No wrappers. No hidden dependencies.

## License

Apache-2.0
