use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use rmcp::ServiceExt;
use rust_rag_mcp::{docs, mcp, rag};

#[derive(Parser)]
#[command(
    name = "rust-rag-mcp",
    about = "RAG MCP server with fastembed, LanceDB, PDF/docx/xlsx/pptx"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start MCP server (stdio transport)
    Serve {
        #[arg(long, default_value = ".rag-db")]
        db_path: String,

        #[arg(long, default_value = ".rag-cache")]
        cache_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,

        #[arg(long, default_value_t = 512)]
        chunk_size: usize,

        #[arg(long, default_value_t = 64)]
        overlap: usize,
    },

    /// Index a file or directory (skips unchanged files automatically)
    Index {
        #[arg(required = true)]
        paths: Vec<String>,

        #[arg(long, default_value = ".rag-db")]
        db_path: String,

        #[arg(long, default_value = ".rag-cache")]
        cache_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,

        #[arg(long, default_value_t = 512)]
        chunk_size: usize,

        #[arg(long, default_value_t = 64)]
        overlap: usize,
    },

    /// Search the knowledge base
    Search {
        #[arg(required = true)]
        query: Vec<String>,

        #[arg(long, default_value = ".rag-db")]
        db_path: String,

        #[arg(long, default_value = ".rag-cache")]
        cache_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,

        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },

    /// List indexed sources
    Sources {
        #[arg(long, default_value = ".rag-db")]
        db_path: String,

        #[arg(long, default_value = ".rag-cache")]
        cache_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,
    },

    /// Show index statistics
    Stats {
        #[arg(long, default_value = ".rag-db")]
        db_path: String,

        #[arg(long, default_value = ".rag-cache")]
        cache_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,
    },

    /// Remove a source document from the index
    Delete {
        /// Full source path to remove (use 'sources' command to list paths)
        source_path: String,

        #[arg(long, default_value = ".rag-db")]
        db_path: String,

        #[arg(long, default_value = ".rag-cache")]
        cache_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,
    },

    /// List available embedding models
    Models,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
                .add_directive("lance=error".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            db_path,
            cache_path,
            model,
            chunk_size,
            overlap,
        } => {
            tracing::info!("Starting RAG MCP server");
            let server =
                mcp::RagServer::new(&db_path, &cache_path, &model, chunk_size, overlap).await?;
            let service = server
                .serve(rmcp::transport::stdio())
                .await
                .inspect_err(|e| {
                    tracing::error!("MCP server error: {:?}", e);
                })?;
            service.waiting().await?;
        }

        Commands::Index {
            paths,
            db_path,
            cache_path,
            model,
            chunk_size,
            overlap,
        } => {
            let core =
                rag::RagCore::new(&db_path, &cache_path, &model, chunk_size, overlap).await?;
            for path_str in &paths {
                let path = std::path::Path::new(path_str);
                if !path.exists() {
                    eprintln!("Path not found: {}", path_str);
                    continue;
                }

                let mut stack = vec![path.to_path_buf()];
                while let Some(current) = stack.pop() {
                    if current.is_dir() {
                        if let Ok(mut entries) = tokio::fs::read_dir(&current).await {
                            while let Ok(Some(entry)) = entries.next_entry().await {
                                let file_path = entry.path();
                                stack.push(file_path);
                            }
                        }
                    } else if current.is_file() && docs::supported_extension(&current) {
                        match core.index_file(&current).await {
                            Ok(rust_rag_mcp::IndexResult::Indexed(count)) => {
                                println!("Indexed {}: {} chunks", current.display(), count)
                            }
                            Ok(rust_rag_mcp::IndexResult::Skipped) => {
                                println!("Skipped {} (unchanged)", current.display())
                            }
                            Err(e) => eprintln!("Failed {}: {}", current.display(), e),
                        }
                    }
                }
            }
            core.flush().await?;
            core.close().await?;
        }

        Commands::Search {
            query,
            db_path,
            cache_path,
            model,
            top_k,
        } => {
            let core = rag::RagCore::new(&db_path, &cache_path, &model, 512, 64).await?;
            let query_str = query.join(" ");
            let results: Vec<rust_rag_mcp::SearchResult> = core.search(&query_str, top_k).await?;

            if results.is_empty() {
                println!("No results found.");
            } else {
                for (i, result) in results.iter().enumerate() {
                    println!(
                        "[{}] (score: {:.4}) [{}:{}] {}",
                        i + 1,
                        result.score,
                        result.chunk.source,
                        result.chunk.chunk_index,
                        result.chunk.text,
                    );
                }
            }
            core.close().await?;
        }

        Commands::Sources {
            db_path,
            cache_path,
            model,
        } => {
            let core = rag::RagCore::new(&db_path, &cache_path, &model, 512, 64).await?;
            let sources = core.list_sources().await?;
            if sources.is_empty() {
                println!("No indexed sources.");
            } else {
                for source in &sources {
                    println!("{}", source);
                }
            }
        }

        Commands::Stats {
            db_path,
            cache_path,
            model,
        } => {
            let core = rag::RagCore::new(&db_path, &cache_path, &model, 512, 64).await?;
            let count = core.chunk_count().await?;
            let sources = core.list_sources().await?;
            println!("Indexed chunks: {}", count);
            println!("Indexed sources: {}", sources.len());
            for source in &sources {
                println!("  - {}", source);
            }
            core.close().await?;
        }

        Commands::Delete {
            source_path,
            db_path,
            cache_path,
            model,
        } => {
            let core = rag::RagCore::new(&db_path, &cache_path, &model, 512, 64).await?;
            match core.delete_source(&source_path).await {
                Ok(()) => println!("Deleted: {}", source_path),
                Err(e) => eprintln!("Failed to delete '{}': {}", source_path, e),
            }
            core.close().await?;
        }

        Commands::Models => {
            let models = rag::RagCore::list_embedding_models();
            println!("Available embedding models:");
            for model in &models {
                println!("  {}", model);
            }
        }
    }

    Ok(())
}
