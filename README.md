# rust-rag-mcp

RAG MCP server using rig, fastembed, LanceDB, unpdf, undoc.

## Features

- **MCP Server** - Run as an MCP server over stdio transport
- **Document Indexing** - Index PDF, DOCX, XLSX, PPTX files and directories
- **Syntax-Aware Code Chunking** - Uses tree-sitter to parse source code into AST, chunking by semantic boundaries (functions, structs, classes) instead of word count
- **Semantic Search** - Search indexed documents using fastembed embeddings
- **Multiple Models** - Supports various embedding models via fastembed

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
```

## Usage

### Start MCP Server

```bash
rust-rag-mcp serve [OPTIONS]
```

Options:
- `--db-path <PATH>` - Database path (default: `.rig-rag-db`)
- `--model <MODEL>` - Embedding model (default: `Xenova/bge-small-en-v1.5`)
- `--chunk-size <N>` - Chunk size for splitting (default: 512)
- `--overlap <N>` - Overlap between chunks (default: 64)

### Index Documents

```bash
rust-rag-mcp index <PATHS>... [OPTIONS]
```

Index files or directories. Supports PDF, DOCX, XLSX, PPTX.

#### Supported Code Languages

Source code files are automatically parsed with tree-sitter and chunked by
semantic boundaries (functions, structs, classes, modules) rather than word count.
This produces higher-quality embeddings for code search.

| Language    | Extensions              |
|-------------|-------------------------|
| Rust        | `.rs`                   |
| Python      | `.py`                   |
| JavaScript  | `.js`, `.jsx`           |
| TypeScript  | `.ts`, `.tsx`           |
| Go          | `.go`                   |
| Java        | `.java`                 |
| C           | `.c`, `.h`              |
| C++         | `.cpp`, `.cc`, `.cxx`, `.hpp` |
| HTML        | `.html`, `.htm`         |
| CSS         | `.css`, `.scss`         |
| JSON        | `.json`, `.jsonc`        |

Non-code files (PDF, DOCX, plain text, etc.) fall back to word-count chunking.

#### Examples

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

Options:
- `--top-k <N>` - Number of results (default: 5)
- `--source <SOURCE>` - Filter by source
- `--db-path <PATH>` - Database path
- `--model <MODEL>` - Embedding model

### List Sources

```bash
rust-rag-mcp sources [OPTIONS]
```

### Show Statistics

```bash
rust-rag-mcp stats [OPTIONS]
```

### List Available Models

```bash
rust-rag-mcp models
```

## Configuration

Set log level via `RUST_LOG` environment variable:

```bash
RUST_LOG=info rust-rag-mcp serve
```

LanceDB logs are set to warn by default to reduce noise.

### MCP Client Configuration

#### opencode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "rust-rag-mcp",
      "args": ["serve", "--db-path", ".rig-rag-db"]
    }
  }
}
```

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "rust-rag-mcp",
      "args": ["serve", "--db-path", ".rig-rag-db"]
    }
  }
}
```

#### VS Code (GitHub Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "rag": {
      "command": "rust-rag-mcp",
      "args": ["serve", "--db-path", ".rig-rag-db"]
    }
  }
}
```

## License

MIT