use crate::DocumentChunk;
use uuid::Uuid;

pub struct Chunker {
    pub max_chunk_size: usize,
    pub overlap: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self {
            max_chunk_size: 512,
            overlap: 64,
        }
    }
}

impl Chunker {
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size,
            overlap,
        }
    }

    pub fn chunk_text(&self, text: &str, source: &str) -> Vec<DocumentChunk> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut start_word = 0;

        while start_word < words.len() {
            let end_word = std::cmp::min(start_word + self.max_chunk_size, words.len());

            let chunk_text: String = words[start_word..end_word].join(" ");
            let start_offset = text.find(words[start_word]).unwrap_or(0);
            let end_offset = text
                .find(words[end_word - 1])
                .map(|pos| pos + words[end_word - 1].len())
                .unwrap_or(text.len());

            chunks.push(DocumentChunk {
                id: Uuid::new_v4().to_string(),
                text: chunk_text,
                source: source.to_string(),
                chunk_index: chunks.len() as u32,
                start_offset,
                end_offset,
            });

            if end_word >= words.len() {
                break;
            }

            start_word += self.max_chunk_size - self.overlap;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_text() {
        let chunker = Chunker::new(10, 2);
        let chunks = chunker.chunk_text("hello world", "test.txt");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].source, "test.txt");
    }

    #[test]
    fn test_chunk_long_text() {
        let chunker = Chunker::new(3, 1);
        let text = "a b c d e f g h i j";
        let chunks = chunker.chunk_text(text, "test.txt");
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_chunk_empty_text() {
        let chunker = Chunker::default();
        let chunks = chunker.chunk_text("", "test.txt");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_whitespace_only() {
        let chunker = Chunker::default();
        let chunks = chunker.chunk_text("   \n\t  ", "test.txt");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_preserves_offsets() {
        let chunker = Chunker::new(5, 0);
        let text = "the quick brown fox jumps";
        let chunks = chunker.chunk_text(text, "test.txt");
        for chunk in &chunks {
            assert!(chunk.start_offset <= chunk.end_offset);
            assert!(chunk.end_offset <= text.len());
        }
    }
}
