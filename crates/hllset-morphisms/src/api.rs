//! Default application-level interface to the two morphisms.
//!
//! The DSL remains the place for custom HLLSet building (Lua scripts,
//! low-level `hllset.add_token`). When nothing special is needed, an
//! application can call [`ingest`] and [`materialize`] directly instead of
//! scripting the pipeline.
//!
//! # Default ingest — end-to-end, padded, n-gram encoded
//!
//! An **ordered token collection** is processed in one pass:
//!
//! ```text
//! tokens = [t1, ..., tn]
//! padded = [PAD, t1, ..., tn, PAD, PAD]     (1 start pad, 2 end pads)
//! ```
//!
//! Every real token `t_i` has exactly three hashes pointing at it, one per
//! n-gram channel (each channel uses its own seed, `[0, 1, 2]`):
//!
//! - channel 0: 1-gram `t_i`
//! - channel 1: 2-gram `t_i · t_{i+1}`
//! - channel 2: 3-gram `t_i · t_{i+1} · t_{i+2}`
//!
//! with `t_{n+1} = t_{n+2} = PAD`. Each hash sets its atom in the channel
//! sketch and registers the **pointed-to token** (`t_i`) in the channel LUT
//! fiber; TF counts every observation of every real token. The two trailing
//! pads guarantee the last token still participates in a 3-gram.
//!
//! # The default return — the SHA1 of the new HLLSet
//!
//! Ingest's purpose is the **population of LUTs and hllsetLUT** as a side
//! effect, and its default return is the SHA1 of the new HLLSet:
//!
//! - [`Ingested::key`] — `h:<sha1>` of the projection
//!   `pr-HLLSet(L) = G1 ∪ G2 ∪ G3` (the collection's new HLLSet). G1/G2/G3
//!   are shared, scheme-agnostic channels;
//! - [`Ingested::keys`] — the `h:<sha1>` keys of the three channel
//!   originals;
//! - [`Ingested::luts`] — the n-gram token LUTs (named `ng:G1` … `ng:G3`;
//!   they preserve the processed tokens and their order);
//! - [`Ingested::hllset_lut`] — the hllsetLUT populated with every created
//!   channel HLLSet (preserves the new HLLSets by their SHA1 keys).
//!
//! # Default materialize — LUT-first, ordered
//!
//! The three `(sketch, LUT)` pairs are materialized LUT-first with TF
//! consulted only for collided bits. The default result is **ordered**: the
//! original sequence is reconstructed by following the 3-gram chain anchored
//! at the start pad (`(PAD, t1, t2)`, then `(t_{i-1}, t_i, t_{i+1})` until
//! the trailing pad). Pass [`MaterializeOptions::no_order`] for the plain
//! (bytewise-sorted) set instead.

use std::collections::BTreeSet;

use ::hllset_lut::LutIndex;
use hllset_contracts::BitAddress;
use hllset_core::HLLSet;

use crate::hllset_lut::HllsetLut;
use crate::materialize::materialize as materialize_lut_first;
use crate::scheme::{lut_name, NG};
use crate::tf::TfTable;

/// The default boundary token: one start pad and two end pads.
pub const PAD: &[u8] = b"<PAD>";

/// Number of n-gram channels (1-, 2-, and 3-grams).
pub const CHANNELS: usize = 3;

/// The channel-selected seeds for the n-gram regime (same shape as the
/// soldered n-seed seed set: one seed per encoding channel).
pub const CHANNEL_SEEDS: [u64; CHANNELS] = [0, 1, 2];

/// The names of the channel HLLSets — named HLLSets on the hllsetLUT.
/// G1/G2/G3 are immutable: each ingest creates a new Gx HLLSet (new SHA1);
/// old versions stay registered and remain addressable.
pub const CHANNEL_NAMES: [&str; CHANNELS] = ["G1", "G2", "G3"];

/// The result of default ingestion: three sketches, three LUTs, TF, and the
/// preservation side effects.
///
/// This is the direct-application counterpart of the DSL's `inscribe`: an
/// ordered token collection, padded and n-gram encoded end-to-end. The
/// **default case** is [`key`](Self::key) — the SHA1 (`h:<sha1>`) of the
/// new HLLSet (`pr-HLLSet = G1 ∪ G2 ∪ G3`). The side effects preserve the
/// work: the token LUTs keep the processed tokens, and
/// [`hllset_lut`](Self::hllset_lut) keeps the created channel HLLSets by
/// their SHA1 keys.
#[derive(Clone, Debug)]
pub struct Ingested {
    /// One sketch per n-gram channel (G1, G2, G3). The HLLSet itself is
    /// bootstrap-scheme agnostic.
    pub sketches: [HLLSet; CHANNELS],
    /// One reverse index per n-gram channel. The fiber of a bit holds the
    /// **pointed-to token** (the first component of the n-gram that hashed
    /// to that bit).
    pub luts: [LutIndex; CHANNELS],
    /// Monotonic term-frequency table over the original tokens.
    pub tf: TfTable,
    /// Number of real tokens ingested (occurrences, not distinct tokens).
    pub tokens: usize,
    /// The pad token used at the boundaries.
    pub pad: Vec<u8>,
    /// The projection HLLSet of the collection: `G1 ∪ G2 ∪ G3`
    /// (`pr-HLLSet(L) = ingest(L)` — the new HLLSet the default ingest
    /// returns the key of).
    pub projection: HLLSet,
    /// SHA1 of the projection (`h:<sha1>`) — **the default return of
    /// ingest**. G1/G2/G3 are shared, scheme-agnostic channels.
    pub key: String,
    /// SHA1 keys of the three channel HLLSets (`h:<sha1>` for G1, G2, G3).
    pub keys: [String; CHANNELS],
    /// The hllsetLUT populated by this ingest: every created channel HLLSet
    /// registered under its name and touch-counted (the preservation side
    /// effect).
    pub hllset_lut: HllsetLut,
}

impl Default for Ingested {
    fn default() -> Self {
        Self::new()
    }
}

impl Ingested {
    /// An empty ingestion result with the default pad.
    pub fn new() -> Self {
        let empty = HLLSet::new();
        let key = empty.content_key();
        Self {
            sketches: std::array::from_fn(|_| HLLSet::new()),
            luts: std::array::from_fn(|_| LutIndex::default()),
            tf: TfTable::new(),
            tokens: 0,
            pad: PAD.to_vec(),
            projection: empty,
            key,
            keys: std::array::from_fn(|_| String::new()),
            hllset_lut: HllsetLut::new(),
        }
    }

    /// The ordered restoration with default options.
    pub fn materialize(&self) -> Vec<Vec<u8>> {
        materialize(self)
    }

    /// The unordered restoration (LUT-first, TF for ambiguity only).
    pub fn materialize_no_order(&self) -> BTreeSet<Vec<u8>> {
        materialize_no_order(self)
    }

    /// The scheme-prefixed names of the n-gram LUTs: `ng:G1`, `ng:G2`,
    /// `ng:G3`. The prefix selects the LUT at materialization time; the
    /// Gn HLLSets themselves are shared and scheme-agnostic.
    pub fn lut_names(&self) -> [String; CHANNELS] {
        std::array::from_fn(|ch| lut_name(NG, ch))
    }
}

/// Ingest an ordered token collection with the default padding
/// (`PAD` at the start, `PAD` twice at the end).
///
/// ```rust
/// use hllset_morphisms::{ingest, materialize};
///
/// let ing = ingest(["the", "cat", "sat"]);
/// let ordered = materialize(&ing);
/// assert_eq!(ordered, vec![b"the".to_vec(), b"cat".to_vec(), b"sat".to_vec()]);
/// ```
pub fn ingest<I, S>(tokens: I) -> Ingested
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
{
    ingest_with_pad(tokens, PAD)
}

/// Ingest an ordered token collection with an explicit boundary token
/// (still one start pad and two end pads).
pub fn ingest_with_pad<I, S>(tokens: I, pad: &[u8]) -> Ingested
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
{
    let real: Vec<Vec<u8>> = tokens.into_iter().map(|t| t.as_ref().to_vec()).collect();
    let mut out = Ingested {
        pad: pad.to_vec(),
        ..Ingested::new()
    };

    if real.is_empty() {
        return out;
    }

    // padded = [PAD] + real + [PAD, PAD]
    let mut padded: Vec<&[u8]> = Vec::with_capacity(real.len() + 3);
    padded.push(pad);
    padded.extend(real.iter().map(|t| t.as_slice()));
    padded.push(pad);
    padded.push(pad);

    // Walk every n-gram window of every order. The first component of a
    // window is the token that n-gram "points at". Real windows register
    // their first component in the LUT fiber and set the sketch atom;
    // pad-leading windows (e.g. `(PAD, t1)`, `(PAD, t1, t2)`) only set their
    // atom, so the ordered materializer can anchor at the start pad.
    for n in 1..=CHANNELS {
        let channel = n - 1;
        let seed = CHANNEL_SEEDS[channel];
        for (start, window) in padded.windows(n).enumerate() {
            let ngram = join(window);
            let addr = BitAddress::of_token_seeded(&ngram, seed);
            out.sketches[channel].add_bit(addr.bit());
            // Real tokens occupy padded positions 1..=real.len().
            if start >= 1 && start <= real.len() {
                out.luts[channel]
                    .insert_token_at(window[0].to_vec(), addr.bit());
            }
        }
    }

    for token in &real {
        out.tf.increment(token);
    }
    out.tokens = real.len();

    // The default return: the SHA1 of the new HLLSet — the projection
    // `pr-HLLSet(L) = G1 ∪ G2 ∪ G3`, prefixed with the bootstrap scheme.
    // The `ng` prefix tells materialization to use the n-gram LUTs (order
    // can be restored).
    out.projection = out.sketches.iter().fold(HLLSet::new(), |acc, s| acc.union(s));
    out.key = out.projection.content_key();

    // Preservation side effects: every created channel HLLSet is registered
    // and touch-counted in the hllsetLUT under its **name** (G1/G2/G3), keyed
    // by its scheme-prefixed SHA1. Named HLLSets are immutable: the next
    // ingest creates new Gx HLLSets (new keys); the old ones stay registered.
    for (ch, sketch) in out.sketches.iter().enumerate() {
        out.keys[ch] = sketch.content_key();
        out.hllset_lut
            .register_named(CHANNEL_NAMES[ch], &out.keys[ch]);
        out.hllset_lut
            .touch_named(CHANNEL_NAMES[ch], &out.keys[ch]);
    }

    out
}

/// The default case in one call: ingest an ordered token collection and
/// return the SHA1 of the new HLLSet (`h:<sha1>`). G1/G2/G3 are shared,
/// scheme-agnostic channels; the bootstrap scheme lives on the LUT names.
///
/// ```rust
/// use hllset_morphisms::ingest_key;
///
/// let key = ingest_key(["the", "cat", "sat"]);
/// assert!(key.starts_with("h:"));
/// ```
pub fn ingest_key<I, S>(tokens: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
{
    ingest(tokens).key
}

/// How the default materializer returns restored tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Order {
    /// Reconstruct the original token sequence by following the 3-gram chain.
    #[default]
    Ordered,
    /// Return the plain restored set (bytewise-sorted, deterministic).
    NoOrder,
}

/// Options for [`materialize_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaterializeOptions {
    pub order: Order,
}

impl MaterializeOptions {
    /// Default options: ordered restoration.
    pub const fn ordered() -> Self {
        Self { order: Order::Ordered }
    }

    /// Plain set restoration (`no_order`).
    pub const fn no_order() -> Self {
        Self { order: Order::NoOrder }
    }
}

/// Materialize with default options: LUT-first over the three channels,
/// then ordered reconstruction via the 3-gram chain.
pub fn materialize(ingested: &Ingested) -> Vec<Vec<u8>> {
    materialize_with(ingested, &MaterializeOptions::ordered())
}

/// Materialize with explicit options.
///
/// - [`Order::Ordered`] (default): reconstructs the original sequence.
/// - [`Order::NoOrder`]: returns the restored set in bytewise order.
pub fn materialize_with(ingested: &Ingested, opts: &MaterializeOptions) -> Vec<Vec<u8>> {
    let tokens = unordered_tokens(ingested);
    match opts.order {
        Order::Ordered => order_tokens(&tokens, ingested),
        Order::NoOrder => tokens.into_iter().collect(),
    }
}

/// The `no_order` restoration: the plain set, bytewise-sorted.
pub fn materialize_no_order(ingested: &Ingested) -> BTreeSet<Vec<u8>> {
    unordered_tokens(ingested)
}

/// LUT-first materialization over all three `(sketch, LUT)` pairs; TF is
/// consulted only when a bit has more than one candidate.
pub fn unordered_tokens(ingested: &Ingested) -> BTreeSet<Vec<u8>> {
    let pairs: Vec<(&HLLSet, &LutIndex)> = ingested
        .sketches
        .iter()
        .zip(ingested.luts.iter())
        .collect();
    materialize_lut_first(&pairs, &ingested.tf)
}

/// The Gn gate: `Gx ∩ H` — extract the Gx channel bits of any HLLSet H.
///
/// Gn channels are shared gate masks, one per channel:
///
/// ```text
/// G1 = G1_ng ∪ G1_ns      G2 = G2_ng ∪ G2_ns      G3 = G3_ng ∪ G3_ns
/// ```
///
/// **Bits are anonymous.** A bit does not remember which token — or which
/// bootstrap scheme — set it. The same bit may have been set by a 1-gram,
/// a seed-0 hash, or PAD; a collision is fine, and resolving it is the
/// **materializer's** problem (bootstrapping + disambiguation), not the
/// gate's.
///
/// The gate only extracts bits. Interpreting them (1-gram vs seed-0 atoms)
/// happens in materialization, in the context of a specific HLLSet and a
/// chosen LUT. The gate is symmetric lattice intersection; the naming makes
/// the channel role explicit in the same way `project` does for time travel.
pub fn gate(gx: &HLLSet, h: &HLLSet) -> HLLSet {
    gx.intersection(h)
}

/// Reconstruct the original order by following the 3-gram chain anchored at
/// the start pad: `(PAD, t1, t2)`, then `(t_{i-1}, t_i, t_{i+1})`, until the
/// chain reaches the trailing pad.
fn order_tokens(tokens: &BTreeSet<Vec<u8>>, ingested: &Ingested) -> Vec<Vec<u8>> {
    if tokens.is_empty() || ingested.tokens == 0 {
        return Vec::new();
    }

    let pad = ingested.pad.as_slice();

    // Anchor: the real token with the start-pad bigram `(PAD, t)` set.
    let starts: Vec<&Vec<u8>> = tokens
        .iter()
        .filter(|t| sketch_contains(ingested, 1, &join2(pad, t)))
        .collect();

    for start in starts {
        let mut path = vec![start.clone()];
        if ingested.tokens == 1 {
            return path;
        }
        if walk(tokens, ingested, pad.to_vec(), start.clone(), &mut path) {
            return path;
        }
    }

    // Fallback: cannot anchor/follow the chain — return the unordered set.
    tokens.iter().cloned().collect()
}

/// Depth-first walk along the 3-gram chain. `prev` and `cur` are the last
/// two restored tokens; `path` holds the real tokens restored so far (the
/// pads are never pushed).
fn walk(
    tokens: &BTreeSet<Vec<u8>>,
    ingested: &Ingested,
    prev: Vec<u8>,
    cur: Vec<u8>,
    path: &mut Vec<Vec<u8>>,
) -> bool {
    let pad = ingested.pad.as_slice();

    if path.len() > ingested.tokens {
        return false;
    }
    if path.len() == ingested.tokens {
        // The chain must terminate at the trailing pad: the final real
        // 3-gram is `(prev, cur, PAD)`.
        return sketch_contains(ingested, 2, &join3(&prev, &cur, pad));
    }

    // Real-token successors first (deterministic bytewise order), then the
    // trailing pad.
    let mut nexts: Vec<Vec<u8>> = tokens
        .iter()
        .filter(|c| sketch_contains(ingested, 2, &join3(&prev, &cur, c)))
        .cloned()
        .collect();
    if sketch_contains(ingested, 2, &join3(&prev, &cur, pad)) {
        nexts.push(pad.to_vec());
    }

    for next in nexts {
        if next.as_slice() == pad {
            // The chain ends here. It is only the true end when the exact
            // number of real tokens has been restored.
            if path.len() == ingested.tokens {
                return true;
            }
            continue;
        }
        path.push(next.clone());
        if walk(tokens, ingested, cur.clone(), next, path) {
            return true;
        }
        path.pop();
    }
    false
}

/// `true` when the channel sketch contains the atom of `bytes` (hashed with
/// the channel's seed).
fn sketch_contains(ingested: &Ingested, channel: usize, bytes: &[u8]) -> bool {
    let addr = BitAddress::of_token_seeded(bytes, CHANNEL_SEEDS[channel]);
    ingested.sketches[channel].has_bit(addr.reg(), addr.tz())
}

/// Join tokens with the soldered NUL separator (single tokens pass through
/// unchanged).
fn join(parts: &[&[u8]]) -> Vec<u8> {
    if parts.len() == 1 {
        return parts[0].to_vec();
    }
    let total: usize = parts.iter().map(|p| p.len()).sum::<usize>() + parts.len() - 1;
    let mut out = Vec::with_capacity(total);
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push(0u8);
        }
        out.extend_from_slice(p);
    }
    out
}

fn join2(a: &[u8], b: &[u8]) -> Vec<u8> {
    join(&[a, b])
}

fn join3(a: &[u8], b: &[u8], c: &[u8]) -> Vec<u8> {
    join(&[a, b, c])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(tokens: &[&str]) -> Vec<Vec<u8>> {
        tokens.iter().map(|t| t.as_bytes().to_vec()).collect()
    }

    #[test]
    fn ingest_sets_three_pointers_per_token() {
        let tokens = bytes(&["a", "b", "c"]);
        let ing = ingest(&tokens);

        assert_eq!(ing.tokens, 3);
        assert_eq!(ing.pad, PAD.to_vec());

        // 1-gram channel: every real token + the pad is present.
        for t in ["a", "b", "c", "<PAD>"] {
            assert!(sketch_contains(&ing, 0, t.as_bytes()), "1-gram {t} missing");
        }

        // 2-gram channel: each real token points via its own bigram; the
        // start pad anchors (PAD, a).
        for (a, b) in [("a", "b"), ("b", "c"), ("c", "<PAD>"), ("<PAD>", "a")] {
            assert!(
                sketch_contains(&ing, 1, &join2(a.as_bytes(), b.as_bytes())),
                "2-gram {a}·{b} missing"
            );
        }

        // 3-gram channel: each real token points via its own trigram; the
        // two end pads make the last token a 3-gram.
        for (a, b, c) in [
            ("a", "b", "c"),
            ("b", "c", "<PAD>"),
            ("c", "<PAD>", "<PAD>"),
            ("<PAD>", "a", "b"),
        ] {
            assert!(
                sketch_contains(&ing, 2, &join3(a.as_bytes(), b.as_bytes(), c.as_bytes())),
                "3-gram {a}·{b}·{c} missing"
            );
        }

        // The LUT fibers hold the pointed-to token, not the n-gram bytes.
        let g2 = BitAddress::of_token_seeded(&join2(b"a", b"b"), CHANNEL_SEEDS[1]);
        assert!(ing.luts[1].fiber(g2.bit()).contains(&b"a".to_vec()));
        assert!(!ing.luts[1].fiber(g2.bit()).contains(&join2(b"a", b"b")));

        let g3 = BitAddress::of_token_seeded(&join3(b"c", PAD, PAD), CHANNEL_SEEDS[2]);
        assert!(ing.luts[2].fiber(g3.bit()).contains(&b"c".to_vec()));

        // TF counts observations, not n-grams.
        assert_eq!(ing.tf.count(b"a"), 1);
        assert_eq!(ing.tf.count(b"b"), 1);
        assert_eq!(ing.tf.count(b"c"), 1);
        assert_eq!(ing.tf.count(PAD), 0);
    }

    #[test]
    fn materialize_roundtrips_in_order() {
        let tokens = bytes(&["the", "cat", "sat", "on", "the", "mat"]);
        let ing = ingest(&tokens);
        let restored = materialize(&ing);
        assert_eq!(restored, tokens, "ordered round-trip");
    }

    #[test]
    fn materialize_no_order_returns_the_set() {
        let tokens = bytes(&["z", "a", "m"]);
        let ing = ingest(&tokens);
        let set = materialize_no_order(&ing);
        let mut expected: BTreeSet<Vec<u8>> = tokens.iter().cloned().collect();
        assert_eq!(set, expected);

        expected = tokens.clone().into_iter().collect();
        let via_opts = materialize_with(&ing, &MaterializeOptions::no_order());
        assert_eq!(via_opts, expected.into_iter().collect::<Vec<_>>());
    }

    #[test]
    fn duplicate_tokens_roundtrip() {
        let tokens = bytes(&["a", "b", "a", "b"]);
        let ing = ingest(&tokens);
        assert_eq!(materialize(&ing), tokens, "duplicates keep their positions");
    }

    #[test]
    fn single_token_roundtrip_and_two_pads() {
        let tokens = bytes(&["solo"]);
        let ing = ingest(&tokens);
        assert_eq!(materialize(&ing), tokens);

        // The two end pads make the last (only) token a 3-gram.
        assert!(sketch_contains(
            &ing,
            2,
            &join3(b"solo", PAD, PAD),
        ));
        // And the start pad anchors the bigram (PAD, solo).
        assert!(sketch_contains(&ing, 1, &join2(PAD, b"solo")));
    }

    #[test]
    fn two_tokens_roundtrip() {
        let tokens = bytes(&["x", "y"]);
        let ing = ingest(&tokens);
        assert_eq!(materialize(&ing), tokens);
    }

    #[test]
    fn empty_collection_materializes_empty() {
        let ing = ingest(Vec::<Vec<u8>>::new());
        assert_eq!(ing.tokens, 0);
        assert!(materialize(&ing).is_empty());
        assert!(materialize_no_order(&ing).is_empty());
    }

    #[test]
    fn custom_pad_roundtrip() {
        let tokens = bytes(&["a", "b", "c"]);
        let ing = ingest_with_pad(&tokens, b"<S>");
        assert_eq!(ing.pad, b"<S>");
        assert_eq!(materialize(&ing), tokens);
    }

    #[test]
    fn last_token_participates_as_a_3gram() {
        let tokens = bytes(&["a", "b"]);
        let ing = ingest(&tokens);
        // b's 3-gram is (b, PAD, PAD) — the two trailing pads.
        assert!(sketch_contains(&ing, 2, &join3(b"b", PAD, PAD)));
    }

    #[test]
    fn ingest_returns_sha1_of_the_projection() {
        let tokens = bytes(&["a", "b", "c"]);
        let ing = ingest(&tokens);

        // Default case: the SHA1 of the new HLLSet (scheme-agnostic Gx).
        assert!(ing.key.starts_with("h:"), "key = {}", ing.key);
        assert_eq!(ing.key, ing.projection.content_key());
        assert_eq!(
            ing.projection.popcount(),
            ing.sketches[0].union(&ing.sketches[1]).union(&ing.sketches[2]).popcount(),
            "projection = G1 ∪ G2 ∪ G3"
        );

        // The three channel originals each have their own SHA1.
        for (ch, key) in ing.keys.iter().enumerate() {
            assert!(key.starts_with("h:"));
            assert_eq!(*key, ing.sketches[ch].content_key());
        }
        // The n-gram LUTs are named by scheme + channel.
        assert_eq!(ing.lut_names(), ["ng:G1", "ng:G2", "ng:G3"]);

        // Preservation side effect: all three new HLLSets are in the hllsetLUT
        // under their names (G1/G2/G3) with TH = 1 (register + touch).
        assert_eq!(ing.hllset_lut.len(), 3);
        for (ch, key) in ing.keys.iter().enumerate() {
            assert_eq!(
                ing.hllset_lut.th_named(CHANNEL_NAMES[ch], key),
                1,
                "one touch per created named HLLSet"
            );
        }
        assert_eq!(
            ing.hllset_lut.ranked().len(),
            3,
            "ranked() reports (name, key, th) entries"
        );
    }

    #[test]
    fn ingest_key_is_the_default_case() {
        let key = ingest_key(["the", "cat", "sat"]);
        assert!(key.starts_with("h:"));
        assert_eq!(key, ingest(["the", "cat", "sat"]).key);
    }

    #[test]
    fn empty_ingest_has_no_preserved_hllsets() {
        let ing = ingest(Vec::<Vec<u8>>::new());
        assert!(ing.key.starts_with("h:"), "empty projection still has a key");
        assert!(ing.hllset_lut.is_empty(), "nothing created → nothing preserved");
        assert!(ing.keys.iter().all(|k| k.is_empty()));
    }

    #[test]
    fn gn_is_a_gate_over_both_bootstrap_schemes() {
        use crate::ingest::Ingest;

        let ng = ingest(["cat", "sat"]);
        let mut ns = Ingest::new();
        ns.ingest_tokens([&b"cat"[..], &b"sat"[..]]);

        // The 1-gram of a token IS the seed-0 hash of the same token: both
        // schemes set the same atom in the shared G1.
        let addr = BitAddress::of_token_seeded(b"cat", 0);
        assert!(ng.sketches[0].has_bit(addr.reg(), addr.tz()));
        assert!(ns.hllset(0).has_bit(addr.reg(), addr.tz()));

        // The gate extracts each scheme's component from the shared channel.
        let g1_shared = ng.sketches[0].union(ns.hllset(0));
        assert_eq!(
            gate(&g1_shared, &ng.sketches[0]).content_key(),
            ng.sketches[0].content_key(),
            "gate(G1, H_ng)"
        );
        assert_eq!(
            gate(&g1_shared, ns.hllset(0)).content_key(),
            ns.hllset(0).content_key(),
            "gate(G1, H_ns)"
        );

        // Same gate property on G2, where the schemes genuinely differ
        // (2-gram "cat|sat" vs seed-1 "cat"/"sat").
        let g2_shared = ng.sketches[1].union(ns.hllset(1));
        assert_eq!(
            gate(&g2_shared, &ng.sketches[1]).content_key(),
            ng.sketches[1].content_key(),
            "gate(G2, H_ng)"
        );
        assert_eq!(
            gate(&g2_shared, ns.hllset(1)).content_key(),
            ns.hllset(1).content_key(),
            "gate(G2, H_ns)"
        );
    }
}
