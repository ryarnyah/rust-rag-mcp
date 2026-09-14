use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use rmcp::ServiceExt;
use rust_rag_mcp::{docs, mcp, rag};

#[derive(Parser)]
#[command(name = "rust-rag-mcp", about = "RAG MCP server with fastembed, LanceDB, PDF/docx/xlsx/pptx")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start MCP server (stdio transport)
    Serve {
        #[arg(long, default_value = ".rig-rag-db")]
        db_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,

        #[arg(long, default_value_t = 512)]
        chunk_size: usize,

        #[arg(long, default_value_t = 64)]
        overlap: usize,
    },

    /// Index a file or directory
    Index {
        #[arg(required = true)]
        paths: Vec<String>,

        #[arg(long, default_value = ".rig-rag-db")]
        db_path: String,

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

        #[arg(long, default_value = ".rig-rag-db")]
        db_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,

        #[arg(long, default_value_t = 5)]
        top_k: usize,

        #[arg(long)]
        source: Option<String>,
    },

    /// List indexed sources
    Sources {
        #[arg(long, default_value = ".rig-rag-db")]
        db_path: String,

        #[arg(long, default_value = "Xenova/bge-small-en-v1.5")]
        model: String,
    },

    /// Show index statistics
    Stats {
        #[arg(long, default_value = ".rig-rag-db")]
        db_path: String,

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
            EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            db_path,
            model,
            chunk_size,
            overlap,
        } => {
            tracing::info!("Starting RAG MCP server");
            let server = mcp::RagServer::new(&db_path, &model, chunk_size, overlap).await?;
            let service = server.serve(rmcp::transport::stdio()).await.inspect_err(|e| {
                tracing::error!("MCP server error: {:?}", e);
            })?;
            service.waiting().await?;
        }

        Commands::Index {
            paths,
            db_path,
            model,
            chunk_size,
            overlap,
        } => {
            let core = rag::RagCore::new(&db_path, &model, chunk_size, overlap).await?;
            for path_str in &paths {
                let path = std::path::Path::new(path_str);
                if path.is_dir() {
                    for entry in std::fs::read_dir(path)? {
                        let entry = entry?;
                        let file_path = entry.path();
                        if docs::supported_extension(&file_path) {
                            match core.index_file(&file_path).await {
                                Ok(count) => println!("Indexed {}: {} chunks", file_path.display(), count),
                                Err(e) => eprintln!("Failed {}: {}", file_path.display(), e),
                            }
                        }
                    }
                } else if path.is_file() {
                    match core.index_file(path).await {
                        Ok(count) => println!("Indexed {}: {} chunks", path.display(), count),
                        Err(e) => eprintln!("Failed {}: {}", path.display(), e),
                    }
                } else {
                    eprintln!("Path not found: {}", path_str);
                }
            }
        }

        Commands::Search {
            query,
            db_path,
            model,
            top_k,
            source,
        } => {
            let core = rag::RagCore::new(&db_path, &model, 512, 64).await?;
            let query_str = query.join(" ");
            let results: Vec<rust_rag_mcp::SearchResult> = core.search(&query_str, top_k, source.as_deref()).await?;

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
        }

        Commands::Sources { db_path, model } => {
            let core = rag::RagCore::new(&db_path, &model, 512, 64).await?;
            let sources = core.list_sources().await?;
            if sources.is_empty() {
                println!("No indexed sources.");
            } else {
                for source in &sources {
                    println!("{}", source);
                }
            }
        }

        Commands::Stats { db_path, model } => {
            let core = rag::RagCore::new(&db_path, &model, 512, 64).await?;
            let count = core.chunk_count().await?;
            let sources = core.list_sources().await?;
            println!("Indexed chunks: {}", count);
            println!("Indexed sources: {}", sources.len());
            for source in &sources {
                println!("  - {}", source);
            }
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
