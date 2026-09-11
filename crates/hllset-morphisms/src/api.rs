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
//! The three `(sketch, LUT)` pairs are materialized LUT-first, **keeping
//! every reference**: a collided bit restores all of its candidate tokens
//! (probabilistic restoration — no TF filtering). The default result is
//! **ordered**: the original sequence is reconstructed by a De Bruijn walk
//! anchored at the start pad. The walk is **count-constrained** (every token
//! restores exactly as many times as TF observed it) and checks **joint
//! 2-/3-/4-gram edges** (the 4-gram side channel makes transitions
//! essentially collision-free on dense frames). TF **scores the
//! transitions** — greedy decoding with backtracking, or beam search via
//! [`MaterializeOptions::beam`]. TF ranks paths, never filters candidates.
//! Pass [`MaterializeOptions::no_order`] for the plain (bytewise-sorted) set
//! instead.

use std::collections::{BTreeMap, BTreeSet};

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
    /// The 4-gram order side channel (n-gram only, seed 3). It is not part
    /// of the shared G1/G2/G3 projection; the ordered materializer uses it
    /// to make De Bruijn transitions essentially collision-free.
    pub g4: HLLSet,
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
            g4: HLLSet::new(),
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

    /// The unordered restoration (LUT-first; collided bits keep every
    /// candidate — probabilistic restoration, no TF filtering).
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

    // The 4-gram order side channel: set atoms for every 4-gram window (no
    // LUT insert — it is used only for order restoration).
    for window in padded.windows(4) {
        let ngram = join(window);
        let addr = BitAddress::of_token_seeded(&ngram, 3);
        out.g4.add_bit(addr.bit());
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterializeOptions {
    pub order: Order,
    /// Beam width for ordered restoration: `1` = greedy De Bruijn walk,
    /// `> 1` = keep the top-k partial paths by cumulative TF score.
    pub beam: usize,
}

impl Default for MaterializeOptions {
    fn default() -> Self {
        Self::ordered()
    }
}

impl MaterializeOptions {
    /// Default options: ordered restoration, greedy De Bruijn walk.
    pub const fn ordered() -> Self {
        Self {
            order: Order::Ordered,
            beam: 1,
        }
    }

    /// Plain set restoration (`no_order`).
    pub const fn no_order() -> Self {
        Self {
            order: Order::NoOrder,
            beam: 1,
        }
    }

    /// Ordered restoration with beam search: keep the top-`width` partial
    /// paths by cumulative TF score. `width == 1` is the greedy walk.
    pub const fn beam(width: usize) -> Self {
        Self {
            order: Order::Ordered,
            beam: width,
        }
    }
}

/// Materialize with default options: LUT-first over the three channels,
/// then ordered reconstruction via the 3-gram chain (greedy De Bruijn walk).
pub fn materialize(ingested: &Ingested) -> Vec<Vec<u8>> {
    materialize_with(ingested, &MaterializeOptions::ordered())
}

/// Materialize with explicit options.
///
/// - [`Order::Ordered`] (default): reconstructs the original sequence via a
///   De Bruijn walk; [`MaterializeOptions::beam`] `> 1` keeps the top-k
///   partial paths by cumulative TF score (beam search).
/// - [`Order::NoOrder`]: returns the restored set in bytewise order.
pub fn materialize_with(ingested: &Ingested, opts: &MaterializeOptions) -> Vec<Vec<u8>> {
    let tokens = unordered_tokens(ingested);
    match opts.order {
        Order::Ordered => order_tokens(&tokens, ingested, opts.beam.max(1)),
        Order::NoOrder => tokens.into_iter().collect(),
    }
}

/// Ordered restoration with beam search of the given width.
///
/// Keeps the top-`width` partial De Bruijn paths by cumulative TF score and
/// returns the best completed sequence. Falls back to the greedy walk, then
/// to the unordered set, if no beam hypothesis completes.
pub fn materialize_beam(ingested: &Ingested, width: usize) -> Vec<Vec<u8>> {
    materialize_with(ingested, &MaterializeOptions::beam(width))
}

/// The `no_order` restoration: the plain set, bytewise-sorted.
pub fn materialize_no_order(ingested: &Ingested) -> BTreeSet<Vec<u8>> {
    unordered_tokens(ingested)
}

/// LUT-first materialization over all three `(sketch, LUT)` pairs. Every
/// candidate referenced by an active bit is kept — a collided bit restores
/// all of its tokens (probabilistic restoration; TF is never used to
/// filter).
pub fn unordered_tokens(ingested: &Ingested) -> BTreeSet<Vec<u8>> {
    let pairs: Vec<(&HLLSet, &LutIndex)> = ingested
        .sketches
        .iter()
        .zip(ingested.luts.iter())
        .collect();
    materialize_lut_first(&pairs)
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
/// **materializer's** problem (it keeps every candidate — probabilistic
/// restoration), not the gate's.
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
///
/// Order restoration is **count-constrained TF-scored decoding** over the De
/// Bruijn graph: the 1-gram channel is the vocabulary, the 2-/3-gram
/// channels are the edges, TF scores the transitions, and the **multiset from
/// TF** constrains every token to its exact observation count — a token can
/// never be reused more times than it appeared. `beam == 1` is greedy
/// decoding with backtracking; `beam > 1` keeps the top-k partial paths by
/// cumulative TF score. This is the same shape as LLM token generation.
fn order_tokens(
    tokens: &BTreeSet<Vec<u8>>,
    ingested: &Ingested,
    beam: usize,
) -> Vec<Vec<u8>> {
    if tokens.is_empty() || ingested.tokens == 0 {
        return Vec::new();
    }

    // The count constraint: every token restores exactly as many times as it
    // was observed (TF counts observations, not n-grams).
    let remaining: BTreeMap<Vec<u8>, usize> = tokens
        .iter()
        .map(|t| (t.clone(), ingested.tf.count(t) as usize))
        .filter(|(_, n)| *n > 0)
        .collect();

    if beam > 1 {
        if let Some(path) = beam_search(tokens, ingested, beam, &remaining) {
            return path;
        }
    }

    let pad = ingested.pad.as_slice();

    // Anchor: the real token with the start-pad bigram `(PAD, t)` set, most
    // frequent first, and with remaining count available.
    let mut starts: Vec<&Vec<u8>> = tokens
        .iter()
        .filter(|t| remaining.get(*t).copied().unwrap_or(0) > 0)
        .filter(|t| sketch_contains(ingested, 1, &join2(pad, t)))
        .collect();
    starts.sort_by(|a, b| {
        ingested
            .tf
            .count(b)
            .cmp(&ingested.tf.count(a))
            .then(a.cmp(b))
    });

    // Node budget: the DFS is complete in principle but collision edges can
    // blow up on dense frames; give up and fall back instead of hanging.
    let mut budget = 20_000usize;
    for start in starts {
        if ingested.tokens == 1 {
            return vec![start.clone()];
        }
        let mut path = vec![start.clone()];
        let mut rem = remaining.clone();
        if let Some(n) = rem.get_mut(start) {
            *n -= 1;
        }
        if walk(
            tokens,
            ingested,
            pad.to_vec(),
            start.clone(),
            &mut path,
            &mut rem,
            &mut budget,
        ) {
            return path;
        }
        if budget == 0 {
            break;
        }
    }

    // Fallback: cannot anchor/follow the chain — return the unordered set.
    tokens.iter().cloned().collect()
}

/// One beam-search hypothesis: its path, cumulative TF score, and the
/// remaining token counts.
struct BeamState {
    path: Vec<Vec<u8>>,
    score: u64,
    remaining: BTreeMap<Vec<u8>, usize>,
}

/// Beam search over the De Bruijn graph: keep the top-`width` partial paths
/// by cumulative TF score, expand them one token at a time under the
/// count constraint, and return the highest-scoring completed sequence.
fn beam_search(
    tokens: &BTreeSet<Vec<u8>>,
    ingested: &Ingested,
    width: usize,
    initial_remaining: &BTreeMap<Vec<u8>, usize>,
) -> Option<Vec<Vec<u8>>> {
    let pad = ingested.pad.as_slice();
    let n = ingested.tokens as usize;
    let width = width.max(1);

    // Initial hypotheses: the start anchors, scored by their own TF.
    let mut beam: Vec<BeamState> = tokens
        .iter()
        .filter(|t| initial_remaining.get(*t).copied().unwrap_or(0) > 0)
        .filter(|t| sketch_contains(ingested, 1, &join2(pad, t)))
        .map(|t| {
            let mut rem = initial_remaining.clone();
            if let Some(n) = rem.get_mut(t) {
                *n -= 1;
            }
            BeamState {
                path: vec![t.clone()],
                score: ingested.tf.count(t),
                remaining: rem,
            }
        })
        .collect();
    if beam.is_empty() {
        return None;
    }
    sort_beam(&mut beam);
    beam.truncate(width);

    let mut seen: BTreeSet<Vec<Vec<u8>>> = beam.iter().map(|s| s.path.clone()).collect();
    let mut completed: Vec<(Vec<Vec<u8>>, u64)> = Vec::new();

    for _ in 0..=n {
        if beam.is_empty() {
            break;
        }

        let mut next: Vec<BeamState> = Vec::new();
        for st in &beam {
            if st.path.len() == n {
                // Complete hypothesis: all trailing edges must be set.
                let prev = &st.path[st.path.len() - 2];
                let cur = &st.path[st.path.len() - 1];
                let base = sketch_contains(ingested, 1, &join2(cur, pad))
                    && sketch_contains(ingested, 2, &join3(prev, cur, pad));
                let ok = if st.path.len() < 2 {
                    base
                } else {
                    let prevprev: Vec<u8> = if st.path.len() == 2 {
                        ingested.pad.to_vec()
                    } else {
                        st.path[st.path.len() - 3].clone()
                    };
                    base && sketch4_contains(ingested, &join4(&prevprev, prev, cur, pad))
                };
                if ok {
                    completed.push((st.path.clone(), st.score));
                }
                continue;
            }

            let prev: Vec<u8> = if st.path.len() == 1 {
                pad.to_vec()
            } else {
                st.path[st.path.len() - 2].clone()
            };
            let cur = st.path[st.path.len() - 1].clone();

            for c in tokens {
                if st.remaining.get(c).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let base = sketch_contains(ingested, 1, &join2(&cur, c))
                    && sketch_contains(ingested, 2, &join3(&prev, &cur, c));
                let ok = if st.path.len() < 2 {
                    base
                } else {
                    let prevprev: Vec<u8> = if st.path.len() == 2 {
                        ingested.pad.to_vec()
                    } else {
                        st.path[st.path.len() - 3].clone()
                    };
                    base && sketch4_contains(ingested, &join4(&prevprev, &prev, &cur, c))
                };
                if !ok {
                    continue;
                }
                let mut new_path = st.path.clone();
                new_path.push(c.clone());
                if !seen.insert(new_path.clone()) {
                    continue;
                }
                let mut new_rem = st.remaining.clone();
                if let Some(n) = new_rem.get_mut(c) {
                    *n -= 1;
                }
                let new_score = st.score + ingested.tf.count(c);
                next.push(BeamState {
                    path: new_path,
                    score: new_score,
                    remaining: new_rem,
                });
            }
        }

        if next.is_empty() {
            break;
        }
        sort_beam(&mut next);
        next.truncate(width);
        beam = next;
    }

    // Best completed hypothesis, cumulative TF first, then lexicographic.
    completed.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    completed.into_iter().next().map(|(path, _)| path)
}

/// Sort a beam by score (descending), then lexicographically for ties.
fn sort_beam(beam: &mut [BeamState]) {
    beam.sort_by(|a, b| b.score.cmp(&a.score).then(a.path.cmp(&b.path)));
}

/// Depth-first walk along the 3-gram chain under the count constraint.
/// `prev` and `cur` are the last two restored tokens; `path` holds the real
/// tokens restored so far (the pads are never pushed); `remaining` holds the
/// still-unrestored observation counts.
fn walk(
    tokens: &BTreeSet<Vec<u8>>,
    ingested: &Ingested,
    prev: Vec<u8>,
    cur: Vec<u8>,
    path: &mut Vec<Vec<u8>>,
    remaining: &mut BTreeMap<Vec<u8>, usize>,
    budget: &mut usize,
) -> bool {
    let pad = ingested.pad.as_slice();

    if *budget == 0 {
        return false;
    }
    *budget -= 1;

    if path.len() == ingested.tokens {
        // Complete: the chain must terminate at the trailing pad — the
        // 2-gram (cur, PAD), the 3-gram (prev, cur, PAD), and (when the
        // path is long enough) the 4-gram (prevprev, prev, cur, PAD).
        let base = sketch_contains(ingested, 1, &join2(&cur, pad))
            && sketch_contains(ingested, 2, &join3(&prev, &cur, pad));
        return if path.len() < 2 {
            base
        } else {
            let prevprev: Vec<u8> = if path.len() == 2 {
                ingested.pad.to_vec()
            } else {
                path[path.len() - 3].clone()
            };
            base && sketch4_contains(ingested, &join4(&prevprev, &prev, &cur, pad))
        };
    }

    // TF-ranked greedy decoding: successors with remaining count and BOTH
    // edges set — the 2-gram (cur, c) and the 3-gram (prev, cur, c). The
    // joint check multiplies the collision filters, so false successors
    // essentially vanish on dense frames.
    let mut nexts: Vec<Vec<u8>> = tokens
        .iter()
        .filter(|c| remaining.get(*c).copied().unwrap_or(0) > 0)
        .filter(|c| {
            let base = sketch_contains(ingested, 1, &join2(&cur, c))
                && sketch_contains(ingested, 2, &join3(&prev, &cur, c));
            if path.len() < 2 {
                base
            } else {
                let prevprev: Vec<u8> = if path.len() == 2 {
                    ingested.pad.to_vec()
                } else {
                    path[path.len() - 3].clone()
                };
                base && sketch4_contains(ingested, &join4(&prevprev, &prev, &cur, c))
            }
        })
        .cloned()
        .collect();
    nexts.sort_by(|a, b| {
        ingested
            .tf
            .count(b)
            .cmp(&ingested.tf.count(a))
            .then(a.cmp(b))
    });

    for next in nexts {
        if let Some(n) = remaining.get_mut(&next) {
            *n -= 1;
        }
        path.push(next.clone());
        if walk(
            tokens,
            ingested,
            cur.clone(),
            next.clone(),
            path,
            remaining,
            budget,
        ) {
            return true;
        }
        path.pop();
        if let Some(n) = remaining.get_mut(&next) {
            *n += 1;
        }
    }
    false
}

/// `true` when the channel sketch contains the atom of `bytes` (hashed with
/// the channel's seed).
fn sketch_contains(ingested: &Ingested, channel: usize, bytes: &[u8]) -> bool {
    let addr = BitAddress::of_token_seeded(bytes, CHANNEL_SEEDS[channel]);
    ingested.sketches[channel].has_bit(addr.reg(), addr.tz())
}

/// `true` when the 4-gram order side channel contains the atom of `bytes`
/// (hashed with seed 3).
fn sketch4_contains(ingested: &Ingested, bytes: &[u8]) -> bool {
    let addr = BitAddress::of_token_seeded(bytes, 3);
    ingested.g4.has_bit(addr.reg(), addr.tz())
}

/// Join tokens with the soldered NUL separator (single tokens pass through
/// unchanged).
pub(crate) fn join(parts: &[&[u8]]) -> Vec<u8> {
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

fn join4(a: &[u8], b: &[u8], c: &[u8], d: &[u8]) -> Vec<u8> {
    join(&[a, b, c, d])
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

    #[test]
    fn tf_ranks_de_bruijn_successors_in_order_restoration() {
        // Two valid chains of the same length share the start (x, y) and
        // branch on the next 3-gram: (x,y,a) vs (x,y,b). Bytewise order
        // would pick `a` first; TF ranking picks the observed `b`.
        let x = b"x".to_vec();
        let y = b"y".to_vec();
        let a = b"a".to_vec();
        let b = b"b".to_vec();
        let vocab = [&x, &y, &a, &b];

        let mut ing = Ingested::new();
        ing.tokens = 4;

        // 1-gram channel: the vocabulary, with LUT fibers pointing at it.
        for t in &vocab {
            let addr = BitAddress::of_token_seeded(t, CHANNEL_SEEDS[0]);
            ing.sketches[0].add_bit(addr.bit());
            ing.luts[0].insert_token_at((*t).clone(), addr.bit());
        }

        // 2-gram channel: the start anchor (PAD, x).
        let start_bigram =
            BitAddress::of_token_seeded(&join2(ing.pad.as_slice(), &x), CHANNEL_SEEDS[1]);
        ing.sketches[1].add_bit(start_bigram.bit());

        // 3-gram channel: both full chains.
        // Chain A (bytewise-first): x y a b
        // Chain B (observed):       x y b a
        let pad = ing.pad.clone();
        let grams: Vec<Vec<u8>> = vec![
            join3(&pad, &x, &y),
            join3(&x, &y, &a),
            join3(&y, &a, &b),
            join3(&a, &b, &pad),
            join3(&x, &y, &b),
            join3(&y, &b, &a),
            join3(&b, &a, &pad),
        ];
        for gram in &grams {
            let addr = BitAddress::of_token_seeded(gram, CHANNEL_SEEDS[2]);
            ing.sketches[2].add_bit(addr.bit());
        }

        // 4-gram order channel: the joint-check collisions filter.
        let fours: Vec<Vec<u8>> = vec![
            join4(&pad, &x, &y, &a),
            join4(&x, &y, &a, &b),
            join4(&y, &a, &b, &pad),
            join4(&pad, &x, &y, &b),
            join4(&x, &y, &b, &a),
            join4(&y, &b, &a, &pad),
        ];
        for gram in &fours {
            let addr = BitAddress::of_token_seeded(gram, 3);
            ing.g4.add_bit(addr.bit());
        }

        // 2-gram channel: both chains' transitions and terminations.
        let bigrams: Vec<Vec<u8>> = vec![
            join2(&x, &y),
            join2(&y, &a),
            join2(&a, &b),
            join2(&b, &pad),
            join2(&y, &b),
            join2(&b, &a),
            join2(&a, &pad),
        ];
        for gram in &bigrams {
            let addr = BitAddress::of_token_seeded(gram, CHANNEL_SEEDS[1]);
            ing.sketches[1].add_bit(addr.bit());
        }

        // TF scores the transitions: b >> a at the branch point.
        for t in &vocab {
            ing.tf.increment(t);
        }
        ing.tf.increment(&b);
        ing.tf.increment(&b);

        // Greedy TF-ranked De Bruijn decode restores the observed chain.
        assert_eq!(materialize(&ing), vec![x, y, b, a], "TF picks the observed branch");
    }

    #[test]
    fn beam_search_keeps_hypotheses_and_returns_the_best_completed_path() {
        // Two valid chains branch at (x, y). The greedy walk follows the
        // highest-TF successor `a` and returns [x, y, a, b] (total TF 8);
        // beam width 2 keeps the other hypothesis [x, y, c, d] (total TF 13)
        // and returns it as the best completed path.
        let x = b"x".to_vec();
        let y = b"y".to_vec();
        let a = b"a".to_vec();
        let b = b"b".to_vec();
        let c = b"c".to_vec();
        let d = b"d".to_vec();
        let vocab = [&x, &y, &a, &b, &c, &d];

        let mut ing = Ingested::new();
        ing.tokens = 4;

        // 1-gram channel: the vocabulary, with LUT fibers.
        for t in &vocab {
            let addr = BitAddress::of_token_seeded(t, CHANNEL_SEEDS[0]);
            ing.sketches[0].add_bit(addr.bit());
            ing.luts[0].insert_token_at((*t).clone(), addr.bit());
        }

        // 2-gram channel: the start anchor (PAD, x).
        let start_bigram =
            BitAddress::of_token_seeded(&join2(ing.pad.as_slice(), &x), CHANNEL_SEEDS[1]);
        ing.sketches[1].add_bit(start_bigram.bit());

        // 3-gram channel: both full chains.
        let pad = ing.pad.clone();
        let grams: Vec<Vec<u8>> = vec![
            join3(&pad, &x, &y),
            join3(&x, &y, &a),
            join3(&y, &a, &b),
            join3(&a, &b, &pad),
            join3(&x, &y, &c),
            join3(&y, &c, &d),
            join3(&c, &d, &pad),
        ];
        for gram in &grams {
            let addr = BitAddress::of_token_seeded(gram, CHANNEL_SEEDS[2]);
            ing.sketches[2].add_bit(addr.bit());
        }

        // 4-gram order channel: the joint-check collisions filter.
        let fours: Vec<Vec<u8>> = vec![
            join4(&pad, &x, &y, &a),
            join4(&x, &y, &a, &b),
            join4(&y, &a, &b, &pad),
            join4(&pad, &x, &y, &c),
            join4(&x, &y, &c, &d),
            join4(&y, &c, &d, &pad),
        ];
        for gram in &fours {
            let addr = BitAddress::of_token_seeded(gram, 3);
            ing.g4.add_bit(addr.bit());
        }

        // 2-gram channel: both chains' transitions and terminations.
        let bigrams: Vec<Vec<u8>> = vec![
            join2(&x, &y),
            join2(&y, &a),
            join2(&a, &b),
            join2(&b, &pad),
            join2(&y, &c),
            join2(&c, &d),
            join2(&d, &pad),
        ];
        for gram in &bigrams {
            let addr = BitAddress::of_token_seeded(gram, CHANNEL_SEEDS[1]);
            ing.sketches[1].add_bit(addr.bit());
        }

        // TF: the greedy branch has the higher first step (a=5) but the
        // lower total; the beam branch has the higher total (c=2, d=9).
        ing.tf.increment(&x);
        ing.tf.increment(&y);
        for _ in 0..5 {
            ing.tf.increment(&a);
        }
        ing.tf.increment(&b);
        for _ in 0..2 {
            ing.tf.increment(&c);
        }
        for _ in 0..9 {
            ing.tf.increment(&d);
        }

        // Greedy (beam width 1) follows the highest-TF successor first.
        assert_eq!(materialize(&ing), vec![x.clone(), y.clone(), a, b], "greedy path");

        // Beam width 2 keeps both hypotheses and returns the best complete.
        assert_eq!(
            materialize_beam(&ing, 2),
            vec![x, y, c, d],
            "beam picks the highest-scoring completed path"
        );
    }

    #[test]
    fn count_constraint_prevents_token_overuse() {
        // Vocabulary {a,b,c}, N=3, each observed once. The 3-gram channel
        // additionally carries trap edges (a,b,a) and (b,a,PAD): without the
        // count constraint the bytewise-first walk would return [a,b,a] —
        // reusing `a` twice and never restoring `c`. The constraint forces
        // the true [a,b,c].
        let a = b"a".to_vec();
        let b = b"b".to_vec();
        let c = b"c".to_vec();
        let vocab = [&a, &b, &c];

        let mut ing = Ingested::new();
        ing.tokens = 3;

        for t in &vocab {
            let addr = BitAddress::of_token_seeded(t, CHANNEL_SEEDS[0]);
            ing.sketches[0].add_bit(addr.bit());
            ing.luts[0].insert_token_at((*t).clone(), addr.bit());
            ing.tf.increment(t);
        }
        let start_bigram =
            BitAddress::of_token_seeded(&join2(ing.pad.as_slice(), &a), CHANNEL_SEEDS[1]);
        ing.sketches[1].add_bit(start_bigram.bit());

        let pad = ing.pad.clone();
        // 2-gram channel: true transitions, trap transition, both terminations.
        let bigrams: Vec<Vec<u8>> = vec![
            join2(&a, &b),
            join2(&b, &c),
            join2(&c, &pad),
            join2(&b, &a),
            join2(&a, &pad),
        ];
        for gram in &bigrams {
            let addr = BitAddress::of_token_seeded(gram, CHANNEL_SEEDS[1]);
            ing.sketches[1].add_bit(addr.bit());
        }

        let grams: Vec<Vec<u8>> = vec![
            join3(&pad, &a, &b), // true start
            join3(&a, &b, &c),   // true chain
            join3(&b, &c, &pad), // true end
            join3(&a, &b, &a),   // trap: loops back to a
            join3(&b, &a, &pad), // trap termination for [a,b,a]
        ];
        for gram in &grams {
            let addr = BitAddress::of_token_seeded(gram, CHANNEL_SEEDS[2]);
            ing.sketches[2].add_bit(addr.bit());
        }

        // 4-gram order channel: true transitions, trap transition, and both
        // terminations.
        let fours: Vec<Vec<u8>> = vec![
            join4(&pad, &a, &b, &c),
            join4(&a, &b, &c, &pad),
            join4(&pad, &a, &b, &a),
            join4(&a, &b, &a, &pad),
        ];
        for gram in &fours {
            let addr = BitAddress::of_token_seeded(gram, 3);
            ing.g4.add_bit(addr.bit());
        }

        assert_eq!(
            materialize(&ing),
            vec![a.clone(), b.clone(), c.clone()],
            "count constraint defeats the overuse trap"
        );
        assert_eq!(
            materialize_beam(&ing, 2),
            vec![a, b, c],
            "beam agrees under the constraint"
        );
    }

    #[test]
    fn exact_order_restoration_of_a_long_duplicated_collection() {
        // Deterministic pseudo-frame: 200 tokens over 60 distinct tids, with
        // repeats. The count-constrained De Bruijn walk must restore the
        // exact sequence — this is the image-application objective.
        let collection: Vec<String> = (0..200)
            .map(|i| format!("tid{}", (i * 37 + i / 7) % 60))
            .collect();
        let ing = ingest(collection.iter().map(|t| t.as_str()));

        let restored: Vec<String> = materialize(&ing)
            .into_iter()
            .map(|t| String::from_utf8_lossy(&t).into_owned())
            .collect();
        assert_eq!(restored, collection, "greedy restores the exact order");

        let restored_beam: Vec<String> = materialize_beam(&ing, 3)
            .into_iter()
            .map(|t| String::from_utf8_lossy(&t).into_owned())
            .collect();
        assert_eq!(restored_beam, collection, "beam-3 agrees");
    }
}
