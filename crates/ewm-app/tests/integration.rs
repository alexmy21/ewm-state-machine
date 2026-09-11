//! ewm-app integration tests — the [UM] harness loop, in order.
//!
//! Encoding discipline (NEXT_SESSION §3.7): every test states which token
//! encoding it uses. The harness path is `tid{n}`; the dual-encoding test
//! exercises both `tid{n}` and 4-byte LE explicitly.

use ewm_app::{StateCache, StateMachine, StubLlm, TokenEncoding, TurnSource};
use ewm_git::{LatticeState, LooseStore, MemoryStore, ObjectId, ObjectStore, Repository};
use hllset_contracts::token::{token_in_bytes, token_in_bytes_le};
use hllset_core::HLLSet;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ewm-app-test-{}-{}-{}",
        std::process::id(),
        name,
        std::thread::current().name().unwrap_or("main")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

// ── 1. Smoke loop ───────────────────────────────────────────────────────────

#[test]
fn smoke_loop_ingest_commits_and_builds_state_tree() {
    // Encoding: tid{n} — the harness path.
    let mut app = StateMachine::new(MemoryStore::default());
    let mut cache = StateCache::empty();
    let turn1 = [10u32, 20, 30];
    let turn2 = [20u32, 30, 40];

    let out1 = app.run_turn(&mut cache, &turn1).expect("turn 1");
    let c1 = out1.commit.clone().expect("first turn commits");
    assert_eq!(out1.head.as_ref(), Some(&c1));
    assert!(app.repo().store().contains(&c1), "commit exists in the store");
    assert_eq!(cache.tree().leaves().len(), 1, "S(t) has one turn leaf");
    assert_eq!(
        out1.full_image,
        turn1.iter().map(|&n| token_in_bytes(n)).collect::<Vec<_>>(),
        "default morphisms restore the turn in order"
    );

    // Root commit: H(t-1) = ∅, so everything is N.
    let v1 = out1.commit_view.as_ref().expect("root view");
    assert_eq!(v1.parent_state.popcount(), 0);
    assert_eq!(v1.departed.popcount(), 0);
    assert_eq!(v1.retained.popcount(), 0);
    assert_eq!(v1.new.popcount(), v1.state.popcount());

    let out2 = app.run_turn(&mut cache, &turn2).expect("turn 2");
    let c2 = out2.commit.clone().expect("second turn commits");
    assert_eq!(out2.head.as_ref(), Some(&c2));

    // Tree-level D/R/N: turn2's leaf entered, turn1's leaf stayed.
    assert_eq!(cache.tree().leaves().len(), 2);
    assert_eq!(out2.diff.added, vec![cache.tree().leaves()[1].h.clone()]);
    assert_eq!(out2.diff.retained, vec![cache.tree().leaves()[0].h.clone()]);
    assert!(out2.diff.removed.is_empty());

    // Bit-level agreement: the tree math (cumulative working set) and
    // ewm-git's view of the head describe the same snapshot.
    let h_prev = cache.turns()[0].g1.clone();
    let s_now = out2.commit_view.as_ref().expect("head view").state.clone();
    let departed = h_prev.difference(&s_now);
    let retained = h_prev.intersection(&s_now);
    let novel = s_now.difference(&h_prev);

    let v2 = out2.commit_view.as_ref().expect("head view");
    assert_eq!(v2.departed.popcount(), departed.popcount());
    assert_eq!(v2.retained.popcount(), retained.popcount());
    assert_eq!(v2.new.popcount(), novel.popcount());

    // Noether invariants (ALGEBRAIC_FOUNDATION §3).
    assert_eq!(
        v2.departed.union(&v2.retained).popcount(),
        v2.parent_state.popcount(),
        "D ∪ R = H(t-1)"
    );
    assert_eq!(
        v2.retained.union(&v2.new).popcount(),
        v2.state.popcount(),
        "R ∪ N = S(t)"
    );
    assert_eq!(v2.departed.intersection(&v2.new).popcount(), 0, "D ∩ N = ∅");
}

// ── 2. Recovery: the cache is rebuilt, the [UM] is replaced ─────────────────

#[test]
fn recovery_reads_tip_and_resumes_without_replay() {
    // Encoding: tid{n}.
    let dir = temp_dir("recovery");
    let turn1 = [10u32, 20, 30];
    let turn2 = [40u32, 50];

    let head1 = {
        let mut app = StateMachine::new(LooseStore::new(&dir));
        let mut cache = StateCache::empty();
        app.run_turn(&mut cache, &turn1).expect("turn 1").commit.expect("commit")
    };

    // Crash (drop the [UM] AND the cache). Restart: a fresh [UM] over the
    // store, a fresh cache restored from the tip. Pop-not-rebuild.
    let mut app = StateMachine::open(LooseStore::new(&dir));
    let mut cache = StateCache::restore(app.repo());
    assert_eq!(app.head(), Some(&head1), "the head is the tip");
    assert_eq!(cache.tip.as_ref(), Some(&head1), "H(t-1) cache holds the tip");
    assert_eq!(app.repo().log().unwrap().len(), 1);
    assert_eq!(cache.turns().len(), 1, "presentation rebuilt from commit messages");
    assert_eq!(cache.turns()[0].ids, turn1);
    assert_eq!(cache.tree().leaves().len(), 1);

    // Replay the same content: same key ⇒ idempotent — no duplicated commit.
    let replay = app.run_turn(&mut cache, &turn1).expect("replay turn");
    assert_eq!(replay.commit, None, "no new bits → no commit");
    assert_eq!(app.head(), Some(&head1), "tip unchanged");
    assert_eq!(app.repo().log().unwrap().len(), 1, "no replay");

    // Resume with new content: exactly one new commit, parented to the
    // recovered tip.
    let resumed = app.run_turn(&mut cache, &turn2).expect("resume turn");
    let c2 = resumed.commit.expect("new content commits");
    assert_eq!(app.repo().log().unwrap().len(), 2);
    let commit = app.repo().read_commit(&c2).unwrap();
    assert_eq!(commit.parents, vec![head1], "resumed from the recovered tip");

    // The presentation now covers both collections (the replay turn is a
    // duplicate leaf, deduplicated by the tree).
    assert_eq!(cache.tree().leaves().len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── 2b. Gn are monotonic — history is implicit ──────────────────────────────

#[test]
fn gn_channels_are_monotonic_across_commits() {
    // Encoding: tid{n}.
    let mut app = StateMachine::new(MemoryStore::default());
    let mut cache = StateCache::empty();

    let out1 = app.run_turn(&mut cache, &[10u32, 20, 30]).expect("turn 1");
    let c1 = out1.commit.expect("commit 1");
    let out2 = app.run_turn(&mut cache, &[20u32, 30, 40]).expect("turn 2");
    let c2 = out2.commit.expect("commit 2");

    // Gn(t) = Gn(S(t)) ∪ Gn(t-1): the channel only grows.
    let g1_t_minus_1 = app.repo().state(&c1).expect("G1(t-1)");
    let g1_t = app.repo().state(&c2).expect("G1(t)");
    assert_eq!(
        g1_t_minus_1.difference(&g1_t).popcount(),
        0,
        "G1(t-1) ⊆ G1(t) — the channel is monotonic"
    );
    assert!(g1_t.popcount() > g1_t_minus_1.popcount(), "new bits joined");

    // G1(t-k): the 1-gram (seed-0) candidates available k commits back.
    assert_eq!(g1_t_minus_1.popcount(), 3, "three candidates one commit back");
    assert_eq!(g1_t.popcount(), 4, "four candidates now");
}

// ── 3. The [UM] never blocks on the token source ────────────────────────────

#[test]
fn um_never_blocks_on_the_token_source() {
    // Encoding: tid{n}.
    use std::sync::{mpsc, Arc, Barrier};

    let script: Vec<Vec<u32>> = vec![
        vec![1, 2],
        vec![2, 3],
        vec![3, 4],
        vec![4, 5],
        vec![5, 6],
    ];

    let script_len = script.len();
    let barrier = Arc::new(Barrier::new(2));
    let producer_barrier = Arc::clone(&barrier);
    let (tx, rx) = mpsc::channel::<Vec<u32>>();

    // The token source produces turns fire-and-forget. It holds NO handle to
    // the harness, so it cannot block on the state machine.
    let producer = std::thread::spawn(move || {
        let mut llm = StubLlm::new(script.clone());
        producer_barrier.wait(); // harness barrier only
        let mut sent = 0;
        while let Some(turn) = llm.next_turn() {
            tx.send(turn).expect("unbounded channel never blocks");
            sent += 1;
        }
        sent
    });

    barrier.wait(); // harness barrier only
    let mut app = StateMachine::new(MemoryStore::default());
    let mut cache = StateCache::empty();
    let mut received = 0;
    while let Ok(turn) = rx.recv() {
        let outcome = app.run_turn(&mut cache, &turn).expect("turn");
        assert!(outcome.commit.is_some(), "every scripted turn brings new bits");
        received += 1;
    }

    let sent = producer.join().expect("producer joins");
    assert_eq!(sent, script_len, "the source finished all turns on its own");
    assert_eq!(received, script_len, "the harness committed every turn");
    assert_eq!(app.repo().log().unwrap().len(), script_len);
}

// ── 4. Zero-copy: two threads share one store, CIDs only ────────────────────

#[test]
fn zero_copy_two_threads_share_one_store_by_cid() {
    // Encoding: tid{n} (the committed state is built from tid tokens).
    let dir = temp_dir("zero-copy");
    let shared = HLLSet::from_tokens([token_in_bytes(7), token_in_bytes(9)]);
    let expected_popcount = shared.popcount();

    let writer_dir = dir.clone();
    // The channel carries ONLY the CID — by type, payload bytes cannot cross.
    let (tx, rx) = std::sync::mpsc::channel::<ObjectId>();

    let writer = std::thread::spawn(move || {
        let mut repo: Repository<LooseStore> = Repository::new(LooseStore::new(&writer_dir));
        let cid = repo
            .commit(&LatticeState::single(&shared), &[], "shared object")
            .expect("commit");
        tx.send(cid.clone()).expect("send CID");
        cid
    });

    let cid = rx.recv().expect("receive CID");
    let reader_cid = cid.clone();
    let reader_dir = dir.clone();
    let reader = std::thread::spawn(move || {
        let repo: Repository<LooseStore> = Repository::open(LooseStore::new(&reader_dir));
        // The object is already resident in the shared store; the reader
        // dereferences it in place from the CID.
        repo.state(&reader_cid).expect("dereference").popcount()
    });

    let writer_cid = writer.join().expect("writer joins");
    assert_eq!(writer_cid, cid, "both threads agree on the content address");
    assert_eq!(
        reader.join().expect("reader joins"),
        expected_popcount,
        "the reader resolved the same object from the shared store"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── 5. Dual token encodings are both first-class ────────────────────────────

#[test]
fn dual_token_encodings_are_both_first_class() {
    // Explicit: this test exercises BOTH inscriptions, one assertion each.
    let tid = TokenEncoding::Tid;
    let le = TokenEncoding::Le;

    assert_eq!(tid.encode(44), b"tid44");
    assert_eq!(tid.parse(b"tid44"), Some(44));

    assert_eq!(le.encode(44), [44, 0, 0, 0]);
    assert_eq!(le.parse(&[44, 0, 0, 0]), Some(44));

    // Cross-parse fails by construction: neither inscription is the other.
    assert_eq!(tid.parse(&[44, 0, 0, 0]), None);
    assert_eq!(le.parse(b"tid44"), None);

    // Both inscriptions are equally valid tokens for the HLLSet lattice.
    let from_tid = HLLSet::from_tokens([tid.encode(44)]);
    let from_le = HLLSet::from_tokens([le.encode(44)]);
    assert_eq!(from_tid.popcount(), 1);
    assert_eq!(from_le.popcount(), 1);

    // The contracts expose the raw inscriptions too (soldered leaf).
    assert_eq!(token_in_bytes(44), b"tid44");
    assert_eq!(token_in_bytes_le(44), [44, 0, 0, 0]);
}

// ── 6. Direct access to ingest / materialize (no DSL needed) ────────────────

#[test]
fn direct_ingest_materialize_roundtrips_in_order() {
    // Encoding: application-level bytes (the default n-gram ingest path).
    let tokens: Vec<Vec<u8>> = ["the", "cat", "sat", "on", "the", "mat"]
        .iter()
        .map(|t| t.as_bytes().to_vec())
        .collect();

    let ing = ewm_app::ingest(&tokens);
    let restored = ewm_app::materialize(&ing);

    assert_eq!(restored, tokens, "default materialize restores the order");
    assert_eq!(ing.tokens, tokens.len());
    assert_eq!(ing.pad, ewm_app::PAD.to_vec());

    // The default case: ingest exposes the SHA1 of the new HLLSet. G1/G2/G3
    // are shared, scheme-agnostic channels; the scheme lives on LUT names.
    assert!(ing.key.starts_with("h:"), "projection key = {}", ing.key);
    assert_eq!(ing.key, ing.projection.content_key());
    assert_eq!(ing.lut_names(), ["ng:G1", "ng:G2", "ng:G3"]);

    // Preservation side effect: all three channel HLLSets are registered in
    // the hllsetLUT under their names (G1/G2/G3) with one touch each.
    assert_eq!(ing.hllset_lut.len(), 3);
    for (ch, key) in ing.keys.iter().enumerate() {
        assert_eq!(
            ing.hllset_lut.th_named(ewm_app::CHANNEL_NAMES[ch], key),
            1,
            "one touch per created named HLLSet"
        );
    }
}

#[test]
fn direct_materialize_no_order_returns_the_plain_set() {
    // Encoding: application-level bytes.
    let tokens = ["z", "a", "m"].map(|t| t.as_bytes().to_vec());
    let ing = ewm_app::ingest(&tokens);

    let set = ewm_app::materialize_no_order(&ing);
    let expected: std::collections::BTreeSet<Vec<u8>> = tokens.iter().cloned().collect();
    assert_eq!(set, expected);

    let via_opts = ewm_app::materialize_with(&ing, &ewm_app::MaterializeOptions::no_order());
    assert_eq!(via_opts, expected.into_iter().collect::<Vec<_>>());
}

// ── 7. The stub replays scripted turns deterministically ────────────────────

#[test]
fn stub_llm_replays_scripted_turns_deterministically() {
    // Encoding: tid{n}.
    let mut llm = StubLlm::with_script(&[&[1, 2, 3], &[4, 5]]);
    assert_eq!(llm.next_turn(), Some(vec![1, 2, 3]));
    assert_eq!(llm.next_turn(), Some(vec![4, 5]));
    assert_eq!(llm.next_turn(), None, "script exhausted");
}
