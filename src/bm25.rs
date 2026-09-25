//! BM25 lexical index with a `db.bm25` sidecar, managed the same way the
//! HNSW index and the `.srcidx` sidecar are: validated at open by generation
//! tokens, self-healing via a rebuild from the metadata scan, maintained
//! incrementally on insert/delete, and persisted only at quiescent points.
//!
//! # Why BM25 alongside HNSW
//!
//! Dense retrieval misses exact identifiers, rare tokens, and acronyms
//! (a query for `parse_source_index` ranks poorly in embedding space), and
//! lexical retrieval misses paraphrase. [`rrf_fuse`] combines both ranked
//! lists with Reciprocal Rank Fusion so a document that ranks well in
//! *either* retriever surfaces — without having to compare incomparable
//! score scales (cosine vs BM25).
//!
//! # Analyzer
//!
//! Language-neutral tokenization: split on non-alphanumeric boundaries,
//! lowercase ([`tokenize`]). The *same* function runs at index and query
//! time, which is the only consistency requirement BM25 has. No stopwords
//! and no stemming — BM25's IDF already flattens common terms, and any
//! language-specific rules would be guesses for this mixed docs+code
//! corpus.
//!
//! # Scoring
//!
//! Okapi BM25 with the standard defaults `k1 = 1.2`, `b = 0.75` and the
//! Lucene-style IDF, which stays positive for every term:
//!
//! ```text
//! idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
//! score    = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))
//! ```
//!
//! Query tokens are *not* deduplicated: this is bag-of-words scoring, and a
//! repeated query term legitimately weighs more.
//!
//! # Persistence: the `.bm25` sidecar
//!
//! Same contract as `.srcidx`: the header carries `(vec_len,
//! meta_record_count)` from the freshly opened database (WAL replay
//! included), and the file is used only on an exact match. Both tokens
//! move on every insert and delete, so a sidecar missing any mutation
//! cannot validate — and compaction renumbers ids, mismatching too. It is
//! a cache of derived state: absent, stale, or corrupt all mean "rebuild
//! from the metadata scan", which is always correct.
//!
//! It is written at quiescent points only — after a startup rebuild, and
//! in `close()` under the `ops` mutex every mutator holds. A crash before
//! `close()` leaves the previous file, whose tokens then mismatch →
//! rebuild (a full corpus re-tokenization, not re-embedding — no model
//! involved). Writes are temp file + `sync_all` + rename, failures warned
//! and never fatal.
//!
//! # Identity
//!
//! BM25 doc ids *are* vector ids: [`crate::r_vector::VectorDb::insert`]
//! only ever appends (`id = len()`), so ids are never recycled and a
//! document keeps one id for life. Removal is done with the stored chunk
//! text ([`Bm25Index::remove_document`]'s contract) so the postings are
//! dropped with the exact tokenization that added them. As a defensive
//! backstop, callers skip deleted ids at query time ([`Bm25Index::search`]
//! itself has no view of deletions), so a leftover posting can waste a
//! lookup but can never surface a dead document.

use anyhow::{Context, Result};
use std::collections::HashMap;

/// BM25 term-weighting parameter: controls term-frequency saturation.
const BM25_K1: f64 = 1.2;

/// BM25 length-normalization parameter: 0 = ignore length, 1 = full
/// normalization against the corpus average.
const BM25_B: f64 = 0.75;

/// Sidecar magic: ASCII "BM25" + format version. Any other value —
/// another file, a truncated header, a future version — means "unusable,
/// rebuild".
const BM25_MAGIC: u64 = 0x424D_3235_0000_0001;

/// Fixed sidecar header: magic, `vec_len`, `meta_record_count`,
/// live-doc count, term count.
const BM25_HEADER: usize = 32;

/// RRF smoothing constant (Cormack et al., SIGIR 2009). `60` is the
/// value from the original paper and the de-facto standard: it keeps
/// rank-1 contributions (`1/61`) comfortably above rank-`k` noise while
/// still letting a document ranked high in *both* lists overtake a
/// document ranked high in only one.
pub const RRF_K: f64 = 60.0;

/// Splits `text` into analyzer tokens: maximal runs of alphanumeric
/// characters, lowercased.
///
/// Split-then-lowercase matters for Unicode: `İ` (U+0130) lowercases to
/// two code points, one of which is a combining mark — lowering per
/// character would split it on a boundary the raw text never had. By
/// tokenizing first and lowering second, both index and query derive
/// their boundaries from the same unmutated text, so lookups always
/// agree.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current).to_lowercase());
        }
    }
    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }
    tokens
}

/// An in-memory Okapi BM25 index over chunk texts.
///
/// Keys are vector ids (see the module docs for why they are stable);
/// values are `(doc_id, term_frequency)` postings kept sorted by
/// `doc_id`, plus per-document token counts for length normalization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bm25Index {
    /// term → postings sorted by doc id ascending. Appending is
    /// order-preserving because vector ids only ever grow; removal uses
    /// binary search. An empty postings list is never kept — the term
    /// key itself is dropped instead, which keeps `term_count` (and the
    /// sidecar) an exact mirror of the index.
    postings: HashMap<String, Vec<(u32, u32)>>,
    /// doc id → token count of the chunk (`dl` in BM25).
    doc_len: HashMap<u32, u32>,
    /// Σ doc lengths, maintained incrementally so `avgdl` is O(1).
    total_len: u64,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live documents in the index.
    pub fn doc_count(&self) -> usize {
        self.doc_len.len()
    }

    /// Mean document length (`avgdl`), or 1.0 for an all-empty corpus —
    /// a placeholder that keeps `dl/avgdl` at 0 instead of NaN.
    fn avgdl(&self) -> f64 {
        if self.doc_len.is_empty() {
            return 1.0;
        }
        let avg = self.total_len as f64 / self.doc_len.len() as f64;
        if avg > 0.0 { avg } else { 1.0 }
    }

    /// Indexes one document under `id`.
    ///
    /// Contract: `id` must not already be indexed. Vector ids are never
    /// recycled, and every insertion path in `RagCore` adds exactly once
    /// per freshly allocated id, so a duplicate means a caller bug — the
    /// add is refused (logged, unchanged) rather than corrupting tf
    /// statistics with double postings. Refusal rather than a panic
    /// keeps debug and release behavior identical.
    pub fn add_document(&mut self, id: u32, text: &str) {
        if self.doc_len.contains_key(&id) {
            tracing::warn!(vector_id = id, "bm25: refusing duplicate add");
            return;
        }

        let tokens = tokenize(text);

        // Term frequencies for this document.
        let mut tf: HashMap<&str, u32> = HashMap::with_capacity(tokens.len());
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }

        for (term, freq) in tf {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push((id, freq));
        }
        self.total_len += tokens.len() as u64;
        self.doc_len.insert(id, tokens.len() as u32);
    }

    /// Removes a document from the index.
    ///
    /// Contract: `text` must be the exact text the document was indexed
    /// with (callers pass the stored chunk text read back from the
    /// database), because postings are located by re-tokenizing it —
    /// there is deliberately no forward index, which would roughly
    /// double this index's memory. A mismatched text removes nothing
    /// from `postings` but still drops the length record; the module
    /// docs' query-time deletion guard is what keeps such a leftover
    /// posting from ever surfacing a dead document.
    pub fn remove_document(&mut self, id: u32, text: &str) {
        let Some(len) = self.doc_len.remove(&id) else {
            return; // not indexed (e.g. already removed): no-op
        };
        self.total_len = self.total_len.saturating_sub(len as u64);

        let mut terms = tokenize(text);
        terms.sort();
        terms.dedup();
        for term in terms {
            let empty = match self.postings.get_mut(&term) {
                Some(postings) => match postings.binary_search_by(|&(doc, _)| doc.cmp(&id)) {
                    Ok(pos) => {
                        postings.remove(pos);
                        postings.is_empty()
                    }
                    Err(_) => false,
                },
                None => false,
            };
            if empty {
                self.postings.remove(&term);
            }
        }
    }

    /// Top-`k` documents for `query`, ranked by descending BM25 score,
    /// ties broken by ascending doc id (deterministic output for equal
    /// scores — RRF only consumes ranks, so stability here is ranking
    /// stability there).
    ///
    /// This function has no view of vector-table tombstones: callers
    /// must skip ids the database reports as deleted (see module docs).
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f64)> {
        if k == 0 || self.doc_len.is_empty() {
            return Vec::new();
        }
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let n = self.doc_len.len() as f64;
        let avgdl = self.avgdl();

        let mut scores: HashMap<u32, f64> = HashMap::new();
        for token in &tokens {
            let Some(postings) = self.postings.get(token) else {
                continue; // term never seen in the corpus
            };
            let df = postings.len() as f64;
            // Lucene-style IDF: ln(1 + (N - df + 0.5)/(df + 0.5)) —
            // strictly positive, so a ubiquitous term can never *lose*
            // a document points.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, tf) in postings {
                let tf = tf as f64;
                let dl = self.doc_len.get(&doc).copied().unwrap_or(0) as f64;
                let norm = 1.0 - BM25_B + BM25_B * (dl / avgdl);
                *scores.entry(doc).or_default() +=
                    idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * norm);
            }
        }

        let mut ranked: Vec<(u32, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        ranked
    }
}

/// Fuses ranked lists with Reciprocal Rank Fusion: each document's
/// fused score is `Σ 1 / (RRF_K + rank)` over every list it appears in
/// (ranks are 1-based). Only ranks are consumed — score scales (cosine
/// vs BM25) never need to be comparable.
///
/// `lists` must be pre-ranked best-first (typically truncated to a
/// common candidate pool). Output is sorted by fused score descending,
/// ties broken by ascending doc id, truncated to `top_k`.
pub fn rrf_fuse(lists: &[Vec<u32>], top_k: usize) -> Vec<(u32, f64)> {
    if top_k == 0 {
        return Vec::new();
    }
    let mut scores: HashMap<u32, f64> = HashMap::new();
    for list in lists {
        for (rank, &doc) in list.iter().enumerate() {
            *scores.entry(doc).or_default() += 1.0 / (RRF_K + (rank + 1) as f64);
        }
    }

    let mut fused: Vec<(u32, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    fused.truncate(top_k);
    fused
}

/// Serialize the index for `db.bm25`.
///
/// Layout (all little-endian): the 32-byte header, then the postings —
/// terms in lexicographic order (collected and sorted from the hashmap,
/// so a rewrite of an unchanged index is byte-identical), each as
/// `u32` length + bytes, `u32` posting count, then `(u32 doc, u32 tf)`
/// pairs in ascending doc order — then the length table as ascending
/// `(u32 doc, u32 len)` pairs. `total_len` is recomputed from the
/// length table at parse time rather than stored.
pub fn serialize_bm25(index: &Bm25Index, vec_len: u64, meta_record_count: u64) -> Result<Vec<u8>> {
    let mut terms: Vec<&String> = index.postings.keys().collect();
    terms.sort_unstable();

    let mut docs: Vec<(u32, u32)> = index.doc_len.iter().map(|(&id, &len)| (id, len)).collect();
    docs.sort_unstable();

    let body: usize = terms
        .iter()
        .map(|t| 8 + t.len() + index.postings[*t].len() * 8)
        .sum::<usize>()
        + docs.len() * 8;

    let mut buf = Vec::with_capacity(BM25_HEADER + body);
    buf.extend_from_slice(&BM25_MAGIC.to_le_bytes());
    buf.extend_from_slice(&vec_len.to_le_bytes());
    buf.extend_from_slice(&meta_record_count.to_le_bytes());
    buf.extend_from_slice(
        &u32::try_from(docs.len())
            .context("too many bm25 documents")?
            .to_le_bytes(),
    );
    buf.extend_from_slice(
        &u32::try_from(terms.len())
            .context("too many bm25 terms")?
            .to_le_bytes(),
    );

    for term in terms {
        let postings = &index.postings[term];
        buf.extend_from_slice(
            &u32::try_from(term.len())
                .context("bm25 term too long")?
                .to_le_bytes(),
        );
        buf.extend_from_slice(term.as_bytes());
        buf.extend_from_slice(
            &u32::try_from(postings.len())
                .context("too many bm25 postings")?
                .to_le_bytes(),
        );
        for &(doc, tf) in postings {
            buf.extend_from_slice(&doc.to_le_bytes());
            buf.extend_from_slice(&tf.to_le_bytes());
        }
    }
    for (doc, len) in docs {
        buf.extend_from_slice(&doc.to_le_bytes());
        buf.extend_from_slice(&len.to_le_bytes());
    }
    debug_assert_eq!(buf.len(), BM25_HEADER + body);
    Ok(buf)
}

/// Read a `u32` at `*cur`, advancing past it; `None` on any overrun.
fn take_u32(bytes: &[u8], cur: &mut usize) -> Option<u32> {
    let end = cur.checked_add(4)?;
    let slice = bytes.get(*cur..end)?;
    *cur = end;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

/// Parse and validate `db.bm25`. Returns `None` — meaning "rebuild from
/// the metadata scan" — for *any* deviation: missing/foreign file, short
/// read, bad magic, token mismatch, an id outside `vec_len`, an empty
/// term, a zero tf, non-ascending or duplicated postings, duplicated
/// length entries, invalid UTF-8, a posting whose doc has no length
/// record, truncated entries, or trailing bytes. Like `.srcidx`, every
/// rejection just falls back to the always-correct scan.
pub fn parse_bm25(bytes: &[u8], vec_len: u64, meta_record_count: u64) -> Option<Bm25Index> {
    if bytes.len() < BM25_HEADER {
        return None;
    }
    let magic = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
    if magic != BM25_MAGIC {
        return None;
    }
    let got_vec_len = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let got_record_count = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
    if got_vec_len != vec_len || got_record_count != meta_record_count {
        return None;
    }
    let doc_count = u32::from_le_bytes(bytes[24..28].try_into().ok()?) as usize;
    let term_count = u32::from_le_bytes(bytes[28..32].try_into().ok()?) as usize;

    // Allocation bound before trusting counts from a possibly corrupt
    // header: every length entry costs 8 bytes, every term costs at
    // least 4 (len) + 1 (min token) + 4 (count) + 8 (min posting) = 17;
    // 16 is a safe lower bound for the check (never rejects a valid
    // file, still caps both loops by file size).
    let body = bytes.len() - BM25_HEADER;
    if term_count
        .saturating_mul(16)
        .saturating_add(doc_count.saturating_mul(8))
        > body
    {
        return None;
    }

    let mut cur = BM25_HEADER;
    let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::with_capacity(term_count);
    for _ in 0..term_count {
        let term_len = take_u32(bytes, &mut cur)? as usize;
        if term_len == 0 {
            return None; // the tokenizer never emits empty terms
        }
        let end = cur.checked_add(term_len)?;
        let slice = bytes.get(cur..end)?;
        let term = std::str::from_utf8(slice).ok()?.to_owned();
        cur = end;

        let n_postings = take_u32(bytes, &mut cur)? as usize;
        if n_postings == 0 {
            return None; // empty postings lists are dropped, not written
        }
        if n_postings.saturating_mul(8) > bytes.len() - cur {
            return None;
        }
        let mut list = Vec::with_capacity(n_postings);
        let mut prev: Option<u32> = None;
        for _ in 0..n_postings {
            let doc = take_u32(bytes, &mut cur)?;
            let tf = take_u32(bytes, &mut cur)?;
            if u64::from(doc) >= vec_len || tf == 0 {
                return None;
            }
            if prev.is_some_and(|p| doc <= p) {
                return None; // strictly ascending: no duplicates, no reordering
            }
            prev = Some(doc);
            list.push((doc, tf));
        }
        if postings.insert(term, list).is_some() {
            return None; // duplicate term key
        }
    }

    if doc_count.saturating_mul(8) > bytes.len() - cur {
        return None;
    }
    let mut doc_len = HashMap::with_capacity(doc_count);
    let mut total_len: u64 = 0;
    for _ in 0..doc_count {
        let doc = take_u32(bytes, &mut cur)?;
        let len = take_u32(bytes, &mut cur)?;
        if u64::from(doc) >= vec_len {
            return None;
        }
        if doc_len.insert(doc, len).is_some() {
            return None; // duplicate doc in the length table
        }
        total_len += len as u64;
    }

    // Trailing bytes mean a torn or tampered body: reject rather than
    // trust a prefix.
    if cur != bytes.len() {
        return None;
    }

    // Coherence between the two sections: every posting must have a
    // length record, or its document would score with a fabricated dl.
    for list in postings.values() {
        for (doc, _) in list {
            if !doc_len.contains_key(doc) {
                return None;
            }
        }
    }

    Some(Bm25Index {
        postings,
        doc_len,
        total_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_lowercases_and_splits_on_non_alphanumeric() {
        assert_eq!(tokenize("Hello, World!"), vec!["hello", "world"]);
        assert_eq!(
            tokenize("foo_bar-baz/qux"),
            vec!["foo", "bar", "baz", "qux"]
        );
        assert_eq!(
            tokenize("v1.5 release (2026)"),
            vec!["v1", "5", "release", "2026"]
        );
        assert_eq!(
            tokenize("parseSourceIndex_fn"),
            vec!["parsesourceindex", "fn"],
            "no identifier splitting: whole alphanumeric runs stay whole"
        );
        assert_eq!(tokenize("   \t\n "), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
        // Boundary detection runs on the raw text, before lowering.
        assert_eq!(tokenize("Ünïcodé Ärger"), vec!["ünïcodé", "ärger"]);
    }

    /// The formula pinned end-to-end against hand-computed values
    /// (computed independently — see the literals):
    ///
    /// corpus: d0 = "a b c" (dl 3), d1 = "a" (dl 1); query "a".
    /// N=2, df=2, avgdl=2, k1=1.2, b=0.75
    /// idf = ln(1.2) = 0.1823215567939546
    /// d0 = idf · 2.2 / (1 + 1.2·(0.25 + 0.75·3/2)) = idf · 2.2/2.65
    /// d1 = idf · 2.2 / (1 + 1.2·(0.25 + 0.75·1/2)) = idf · 2.2/1.75
    #[test]
    fn bm25_score_matches_hand_computed_value() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "a b c");
        idx.add_document(1, "a");

        let hits = idx.search("a", 10);
        assert_eq!(hits.len(), 2);
        // Shorter document with the same tf ranks first.
        assert_eq!(hits[0].0, 1);
        assert!(
            (hits[0].1 - 0.2292042428266858).abs() < 1e-12,
            "{}",
            hits[0].1
        );
        assert_eq!(hits[1].0, 0);
        assert!(
            (hits[1].1 - 0.15136129243271704).abs() < 1e-12,
            "{}",
            hits[1].1
        );
    }

    /// Ranking properties that must hold regardless of constants:
    /// higher tf wins, rarer terms dominate, and an unmatched document
    /// is never returned.
    #[test]
    fn bm25_ranks_by_rarity_and_term_frequency() {
        let mut idx = Bm25Index::new();
        idx.add_document(10, "quokka quokka marsupial"); // tf=2 of rare term
        idx.add_document(11, "quokka only once"); // tf=1 of same term
        idx.add_document(12, "common word in most documents");
        idx.add_document(13, "nothing relevant here");

        let hits = idx.search("quokka", 10);
        let ids: Vec<u32> = hits.iter().map(|&(id, _)| id).collect();
        assert_eq!(ids, vec![10, 11], "tf order, unmatched docs excluded");
        assert!(hits.iter().all(|&(_, s)| s > 0.0));

        // The rare term outranks the common one by a wide margin.
        let rare = idx.search("quokka", 1)[0].1;
        let mut common = Bm25Index::new();
        common.add_document(1, "common word");
        common.add_document(2, "common word again");
        common.add_document(3, "common word also");
        let common_top = common.search("common", 1)[0].1;
        assert!(rare > common_top, "{rare} !> {common_top}");

        // Query terms never seen: no hits, no panic.
        assert!(idx.search("unobtainium", 5).is_empty());
        assert!(idx.search("", 5).is_empty());
        assert!(idx.search("quokka", 0).is_empty());
    }

    /// Removal is by stored text: hits and statistics both drop.
    #[test]
    fn remove_document_drops_hits_and_statistics() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "alpha beta gamma");
        idx.add_document(1, "delta epsilon");
        assert_eq!(idx.doc_count(), 2);

        idx.remove_document(0, "alpha beta gamma");
        assert_eq!(idx.doc_count(), 1);
        assert!(idx.search("alpha", 10).is_empty(), "postings removed");
        assert_eq!(
            idx.search("delta", 10)
                .iter()
                .map(|&(id, _)| id)
                .collect::<Vec<_>>(),
            vec![1],
            "survivor still searchable"
        );
        // avgdl now reflects only the survivor (total 3 / count 1 —
        // not the pre-delete 5 / 2): the survivor's score must equal
        // the score it gets in a freshly built single-doc index.
        let survivor = idx.search("delta", 10)[0].1;
        let mut fresh = Bm25Index::new();
        fresh.add_document(1, "delta epsilon");
        assert_eq!(survivor, fresh.search("delta", 10)[0].1);

        // Removing an unknown id is a no-op, not a panic.
        idx.remove_document(99, "whatever");
        assert_eq!(idx.doc_count(), 1);
    }

    /// The `remove_document` contract (exact stored text) violated on
    /// purpose: the length record still drops, the postings do not, and
    /// the index stays consistent — search never NaNs, and the stale
    /// posting is exactly why `RagCore` liveness-checks lexical
    /// candidates against the vector table before returning them.
    #[test]
    fn remove_document_contract_violation_keeps_index_consistent() {
        let mut idx = Bm25Index::new();
        idx.add_document(1, "delta epsilon");

        idx.remove_document(1, "mismatched text"); // wrong text: postings survive
        assert_eq!(idx.doc_count(), 0);

        // With no live documents the empty-corpus guard short-circuits
        // before any (now meaningless) IDF math: nothing surfaces, no
        // NaN, no panic.
        assert!(idx.search("delta", 10).is_empty());

        // Once a live document exists again, the stale posting *does*
        // resurface the dead doc (dl falls back to 0) — the hazard the
        // query-time tombstone guard exists for.
        idx.add_document(2, "delta again");
        let ids: Vec<u32> = idx.search("delta", 10).iter().map(|&(id, _)| id).collect();
        assert!(
            ids.contains(&1),
            "stale posting must be visible to this test so the guard's necessity is pinned: {ids:?}"
        );
        assert!(ids.contains(&2));
    }

    #[test]
    fn duplicate_add_is_refused_not_corrupting() {
        let mut idx = Bm25Index::new();
        idx.add_document(7, "once");
        let before = idx.clone();
        idx.add_document(7, "twice");
        assert_eq!(idx, before, "second add must be refused");
    }

    /// RRF semantics: presence in both lists dominates; equal scores
    /// order deterministically by id; ranks are 1-based.
    #[test]
    fn rrf_fuse_ranks_overlap_above_singletons() {
        let fused = rrf_fuse(&[vec![1, 2, 3], vec![3, 4, 5]], 10);
        // doc 3 = 1/63 (list 1, rank 3) + 1/61 (list 2, rank 1);
        // doc 1 = 1/61; docs 2, 4 = 1/62 (tie → id order); doc 5 = 1/63.
        let ids: Vec<u32> = fused.iter().map(|&(id, _)| id).collect();
        assert_eq!(ids, vec![3, 1, 2, 4, 5]);
        assert!(
            (fused[0].1 - (1.0 / 63.0 + 1.0 / 61.0)).abs() < 1e-15,
            "{}",
            fused[0].1
        );
        assert!((fused[1].1 - 1.0 / 61.0).abs() < 1e-15, "{}", fused[1].1);
        assert!(
            fused[1].1 < 1.0 / RRF_K,
            "ranks are 1-based: a first-place doc scores 1/61, never 1/60"
        );

        // top_k truncates; empty inputs and top_k=0 are safe.
        assert_eq!(rrf_fuse(&[vec![1, 2], vec![2]], 2).len(), 2);
        assert!(rrf_fuse(&[], 5).is_empty());
        assert!(rrf_fuse(&[vec![1]], 0).is_empty());
    }

    #[test]
    fn sidecar_roundtrip() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "Rust is a systems programming language");
        idx.add_document(1, "ownership rules the borrow checker enforces");
        idx.add_document(2, "systèmes système");
        let bytes = serialize_bm25(&idx, 3, 5).unwrap();
        let parsed = parse_bm25(&bytes, 3, 5).unwrap();
        assert_eq!(parsed, idx);
        assert_eq!(parsed.doc_count(), 3);
    }

    #[test]
    fn sidecar_empty_roundtrip() {
        let idx = Bm25Index::new();
        let bytes = serialize_bm25(&idx, 0, 0).unwrap();
        assert_eq!(parse_bm25(&bytes, 0, 0).unwrap(), idx);

        // A corpus with only zero-token documents serializes to a
        // header with zero terms/docs and must still validate.
        let mut degenerate = Bm25Index::new();
        degenerate.add_document(0, "!!! ...");
        assert_eq!(degenerate.doc_count(), 1);
        let bytes = serialize_bm25(&degenerate, 1, 1).unwrap();
        assert_eq!(parse_bm25(&bytes, 1, 1).unwrap(), degenerate);
    }

    /// Deterministic bytes: a rewrite with no changes is identical
    /// (the writer can then avoid pointless renames if ever added),
    /// regardless of hashmap iteration order.
    #[test]
    fn sidecar_bytes_are_deterministic() {
        let mut idx = Bm25Index::new();
        for (i, text) in ["beta alpha", "gamma beta", "alpha gamma delta"]
            .iter()
            .enumerate()
        {
            idx.add_document(i as u32, text);
        }
        let a = serialize_bm25(&idx, 3, 3).unwrap();
        let b = serialize_bm25(&idx, 3, 3).unwrap();
        assert_eq!(a, b);
    }

    /// The whole safety argument: an exact generation match is required.
    /// Any insert or delete since the persist moves a token.
    #[test]
    fn sidecar_rejects_any_token_mismatch() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "one");
        let bytes = serialize_bm25(&idx, 10, 20).unwrap();
        assert!(parse_bm25(&bytes, 10, 20).is_some());
        assert!(
            parse_bm25(&bytes, 11, 20).is_none(),
            "row append moves vec_len"
        );
        assert!(
            parse_bm25(&bytes, 10, 21).is_none(),
            "tombstone moves record count"
        );
        assert!(
            parse_bm25(&bytes, 9, 19).is_none(),
            "compaction rewrites both"
        );
    }

    /// Foreign file or truncation at *any* offset must be rejected
    /// outright — never partially trusted.
    #[test]
    fn sidecar_rejects_foreign_and_truncated_bytes() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "alpha beta");
        idx.add_document(1, "beta gamma");
        let bytes = serialize_bm25(&idx, 2, 2).unwrap();

        let mut foreign = bytes.clone();
        foreign[0] = 0xFF;
        assert!(parse_bm25(&foreign, 2, 2).is_none());

        for cut in 0..bytes.len() {
            assert!(
                parse_bm25(&bytes[..cut], 2, 2).is_none(),
                "prefix of {cut} bytes"
            );
        }
    }

    /// Torn/tampered bodies: trailing bytes, out-of-range ids, zero
    /// tf, out-of-order postings, a posting without a length record,
    /// and header counts that outgrow the body all mean "rebuild".
    #[test]
    fn sidecar_rejects_tampered_body() {
        let mut idx = Bm25Index::new();
        idx.add_document(1, "alpha beta");
        let base = serialize_bm25(&idx, 4, 4).unwrap();

        // Trailing garbage.
        let mut trailing = base.clone();
        trailing.extend_from_slice(&[0u8; 4]);
        assert!(parse_bm25(&trailing, 4, 4).is_none());

        // Header counts that could never fit the body (allocation bound).
        let mut absurd = base.clone();
        absurd[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_bm25(&absurd, 4, 4).is_none());

        // Doc id equal to vec_len: one past the last addressable row.
        // Layout: 32-byte header, then term len (4), term "alpha" (5),
        // count (4), doc (4), tf (4), then len entries.
        let mut oob = base.clone();
        let doc_at = BM25_HEADER + 4 + "alpha".len() + 4;
        oob[doc_at..doc_at + 4].copy_from_slice(&4u32.to_le_bytes());
        assert!(parse_bm25(&oob, 4, 4).is_none());

        // Zero tf (never written: tf ≥ 1 by construction).
        let mut zero_tf = base.clone();
        zero_tf[doc_at + 4..doc_at + 8].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse_bm25(&zero_tf, 4, 4).is_none());

        // Postings must ascend: swap two docs' ids in a shared term's
        // list so it becomes descending.
        let mut two = Bm25Index::new();
        two.add_document(0, "shared");
        two.add_document(2, "shared");
        let mut bytes = serialize_bm25(&two, 4, 4).unwrap();
        let first_doc_at = BM25_HEADER + 4 + "shared".len() + 4;
        let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
        a.copy_from_slice(&bytes[first_doc_at..first_doc_at + 4]);
        let second_at = first_doc_at + 8;
        b.copy_from_slice(&bytes[second_at..second_at + 4]);
        bytes[first_doc_at..first_doc_at + 4].copy_from_slice(&b);
        bytes[second_at..second_at + 4].copy_from_slice(&a);
        assert!(
            parse_bm25(&bytes, 4, 4).is_none(),
            "descending postings must be rejected"
        );

        // A posting whose doc has no length record (sections incoherent).
        let mut orphan = serialize_bm25(&two, 4, 4).unwrap();
        // Length-table entries: doc (4) + len (4), at the very end.
        let len_doc_at = orphan.len() - 8;
        orphan[len_doc_at..len_doc_at + 4].copy_from_slice(&3u32.to_le_bytes());
        // Docs 0 and 2 are posted; retarget one length entry to 3 so
        // posted doc (0 or 2) loses its record.
        assert!(
            parse_bm25(&orphan, 4, 4).is_none(),
            "orphan posting rejected"
        );
    }
}
