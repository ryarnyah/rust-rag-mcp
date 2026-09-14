# rust-rag-mcp

RAG MCP server using rig, fastembed, LanceDB, unpdf, undoc.

## Features

- **MCP Server** - Run as an MCP server over stdio transport
- **Document Indexing** - Index PDF, DOCX, XLSX, PPTX files and directories
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

## License

MIT