//! Shared layout helpers for the optional on-disk sidecars (`db.srcidx`,
//! `db.bm25`). Both files open with the same 32-byte prefix — magic, the two
//! freshness tokens they are validated against, and two `u32` counts — and
//! both are read back with the same bounded cursor, so the parsing lives here
//! once. Every sidecar is an optimization: a rejected read falls back to the
//! always-correct scan of the metadata.

use anyhow::{Context, Result};

/// Bytes before the first payload entry: `magic`, `vec_len`,
/// `meta_record_count`, `count_a`, `count_b` — all little-endian.
pub(crate) const HEADER_LEN: usize = 32;

/// The validated prefix shared by every sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Header {
    pub(crate) vec_len: u64,
    pub(crate) meta_record_count: u64,
    /// First of the two format-specific counts in the header (chunk sources
    /// for `.srcidx`, documents for `.bm25`).
    pub(crate) count_a: u32,
    /// Second of the two format-specific counts (doc sources / terms).
    pub(crate) count_b: u32,
}

/// The size left for payload entries once the header is consumed.
pub(crate) fn body_len(bytes: &[u8]) -> usize {
    bytes.len().saturating_sub(HEADER_LEN)
}

/// Read the shared header. `None` when the buffer is shorter than
/// [`HEADER_LEN`] or the magic does not match this format.
pub(crate) fn read_header(bytes: &[u8], magic: u64) -> Option<Header> {
    if bytes.len() < HEADER_LEN {
        return None;
    }
    if u64::from_le_bytes(bytes[0..8].try_into().ok()?) != magic {
        return None;
    }
    Some(Header {
        vec_len: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
        meta_record_count: u64::from_le_bytes(bytes[16..24].try_into().ok()?),
        count_a: u32::from_le_bytes(bytes[24..28].try_into().ok()?),
        count_b: u32::from_le_bytes(bytes[28..32].try_into().ok()?),
    })
}

/// Append the shared header at the front of a fresh buffer. Counts arrive as
/// `u32` so a format that needs a different overflow message converts them
/// itself.
pub(crate) fn push_header(buf: &mut Vec<u8>, magic: u64, header: &Header) {
    debug_assert!(buf.is_empty(), "the header is written first");
    buf.extend_from_slice(&magic.to_le_bytes());
    buf.extend_from_slice(&header.vec_len.to_le_bytes());
    buf.extend_from_slice(&header.meta_record_count.to_le_bytes());
    buf.extend_from_slice(&header.count_a.to_le_bytes());
    buf.extend_from_slice(&header.count_b.to_le_bytes());
    debug_assert_eq!(buf.len(), HEADER_LEN);
}

/// Narrow a payload-derived count to `u32`, naming the payload it bounds.
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("too many {what}"))
}

/// Narrow a name length to `u32`.
pub(crate) fn name_len(name: &str, what: &str) -> Result<u32> {
    u32::try_from(name.len()).with_context(|| format!("{what} too long"))
}

/// Read a `u32` at `*cur`, advancing past it; `None` on any overrun.
pub(crate) fn take_u32(bytes: &[u8], cur: &mut usize) -> Option<u32> {
    let end = cur.checked_add(4)?;
    let slice = bytes.get(*cur..end)?;
    *cur = end;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

/// Read a length-prefixed UTF-8 name at `*cur`; `None` on any overrun
/// or invalid UTF-8.
pub(crate) fn take_name(bytes: &[u8], cur: &mut usize) -> Option<String> {
    let len = take_u32(bytes, cur)? as usize;
    let end = cur.checked_add(len)?;
    let slice = bytes.get(*cur..end)?;
    *cur = end;
    std::str::from_utf8(slice).ok().map(str::to_owned)
}

/// Write a length-prefixed name.
pub(crate) fn push_name(buf: &mut Vec<u8>, name: &str, what: &str) -> Result<()> {
    buf.extend_from_slice(&name_len(name, what)?.to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    Ok(())
}
