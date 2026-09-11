//! ewm-git CLI demo — commits the conversation sentences as three-channel
//! global states, shows the commit DAG, the `H(t)` views, time-travel
//! projection, a merge, and GC.
//!
//! ```text
//! cargo run -p ewm-git
//! ```

use ewm_git::{view, views, BitTf, LatticeState, Gx, LooseStore, MemoryStore, Repository};
use hllset_core::core::hashing::murmur3_hash_seeded;
use hllset_core::HLLSet;

fn hllset_seeded(tokens: &[Vec<u8>], seed: u64) -> HLLSet {
    HLLSet::from_hashes(tokens.iter().map(|t| murmur3_hash_seeded(t, seed)))
}

/// G1 = seed-0 (1-gram), G2 = seed-1 (2-gram), G3 = seed-2 (3-gram).
/// n-grams and seeds are interchangeable — they are just bits.
fn lattice_state(tokens: &[Vec<u8>]) -> LatticeState {
    let g1 = hllset_seeded(tokens, 0);
    let g2 = hllset_seeded(tokens, 1);
    let g3 = hllset_seeded(tokens, 2);
    let mut tf = BitTf::new();
    tf.touch(&g1);
    LatticeState { g1, g2, g3, tf }
}

fn sentence_tokens(text: &str) -> Vec<Vec<Vec<u8>>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|w| w.as_bytes().to_vec()).collect())
        .collect()
}

fn main() {
    let path = "/home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-cortex/corpus/conversation.txt";
    let text = std::fs::read_to_string(path).expect("corpus file");
    let sentences = sentence_tokens(&text);
    println!("ewm-git — content-addressed evolution store (2005-style)");
    println!("  sentences: {}", sentences.len());

    let mut repo = Repository::new(MemoryStore::default());

    let mut head = repo.commit(&lattice_state(&sentences[0]), &[], "sentence 0").unwrap();
    for (i, tokens) in sentences.iter().enumerate().skip(1) {
        head = repo.commit(&lattice_state(tokens), &[head], &format!("sentence {i}")).unwrap();
    }

    let log = repo.log().unwrap();
    println!("  commits in HEAD history: {}", log.len());

    let v = view(&repo, &head).unwrap();
    println!();
    println!("  H(t) view of HEAD {} on G1:", head.short());
    println!("    S(t)   : {} bits", v.state.popcount());
    println!("    H(t-1) : {} bits", v.parent_state.popcount());
    println!("    D      : {} bits (departed)", v.departed.popcount());
    println!("    R      : {} bits (retained)", v.retained.popcount());
    println!("    N      : {} bits (new)", v.new.popcount());

    // All three channels are commit-linked.
    let vs = views(&repo, &head).unwrap();
    println!();
    println!("  channel snapshots at HEAD:");
    for (gx, cv) in Gx::ALL.iter().zip(vs.iter()) {
        println!("    {gx}: {} bits", cv.state.popcount());
    }

    // The lattice state also carries the bit-TF vector and the HLLSet-LUT.
    let tf = repo.state_tf(&head).unwrap();
    println!();
    println!("  bit-TF at HEAD: {} active bits, total TF {:.0}", tf.active_bits(), tf.total());
    println!("  HLLSet-LUT: {} original HLLSets", repo.hllset_lut().len());
    for (id, th) in repo.hllset_lut().ranked().into_iter().take(3) {
        println!("    {} → TH {}", id.short(), th);
    }

    // Time travel: project the HEAD state onto the FIRST commit.
    let first = &log[0];
    let query = repo.state(&head).unwrap();
    let projected = repo.project(&query, first, Gx::G1).unwrap();
    println!();
    println!("  time travel: HEAD state ({} bits) ∩ G1(commit {}) = {} bits",
        query.popcount(), first.short(), projected.popcount());
    for gx in Gx::ALL {
        let p = repo.project(&query, first, gx).unwrap();
        println!("    {gx} at first commit: {} of {} bits survive", p.popcount(), query.popcount());
    }

    // Merge the first and last commits — a derivation, not a commit.
    let joined = repo.merge(&log[0], &head).unwrap();
    let m = repo.merge_commit(&log[0], &head, "merge first and last").unwrap();
    let mv = view(&repo, &m).unwrap();
    println!();
    println!("  merge {} (parents {} and {}): G1 state {} bits",
        m.short(), log[0].short(), head.short(), mv.state.popcount());
    println!("    (join was derivable; explicit commit only for topology: {} == {})",
        joined.g1.popcount(), mv.state.popcount());

    // Context-size guidance: compound ops don't commit; warn + compress.
    if let Some(w) = repo.context_warning(100) {
        println!("  WARNING: {}", w);
    } else {
        println!("  context ok: {} bits (warn_above 100)", repo.context_size());
    }

    // GC keeps HEAD-reachable history only.
    let pruned = repo.gc(&[]).unwrap().pruned;
    println!();
    println!("  gc pruned {} unreachable objects", pruned);

    // Persistence check: same DAG over the loose-file store.
    let dir = std::env::temp_dir().join("ewm-git-cli-demo");
    let _ = std::fs::remove_dir_all(&dir);
    let mut loose = Repository::new(LooseStore::new(&dir));
    let c = loose.commit(&lattice_state(&sentences[0]), &[], "persisted").unwrap();
    assert_eq!(
        loose.state(&c).unwrap().popcount(),
        lattice_state(&sentences[0]).g1.popcount()
    );
    println!("  loose store roundtrip ✓ ({})", dir.display());
    let _ = std::fs::remove_dir_all(&dir);
}
