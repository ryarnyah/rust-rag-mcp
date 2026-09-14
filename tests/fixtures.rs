pub fn sample_text() -> String {
    "Rust is a systems programming language focused on safety, speed, and concurrency. \
     It achieves memory safety without garbage collection through its ownership system. \
     The borrow checker enforces strict rules about how references can be used. \
     Rust is used in web browsers, operating systems, and game engines. \
     The crate ecosystem provides libraries for almost any task. \
     Async Rust enables efficient concurrent programming with the tokio runtime. \
     The Rust compiler provides detailed error messages to help developers fix issues. \
     Cargo is the build system and package manager for Rust projects."
        .to_string()
}

pub fn sample_text_with_sections() -> String {
    "--- Page 1 ---\n\
     Introduction to Machine Learning\n\n\
     Machine learning is a subset of artificial intelligence that focuses on building \
     systems that learn from data. These systems improve their performance over time \
     without being explicitly programmed. The field has grown rapidly in recent years.\n\n\
     --- Page 2 ---\n\
     Deep Learning Fundamentals\n\n\
     Deep learning uses neural networks with many layers to model complex patterns. \
     Transformers have revolutionized natural language processing. The attention mechanism \
     allows models to focus on relevant parts of the input.\n\n\
     --- Page 3 ---\n\
     Retrieval Augmented Generation\n\n\
     RAG combines information retrieval with text generation. Documents are split into \
     chunks and converted to vector embeddings. Semantic search finds relevant chunks \
     which are then provided as context to a language model."
        .to_string()
}
