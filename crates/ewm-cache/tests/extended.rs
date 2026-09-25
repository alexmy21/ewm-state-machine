//! Acceptance tests for the Arrow extended cache (ARROW_CACHE.md §9):
//! IPC round-trips are byte-identical, a restored cache equals a
//! rebuilt one, `h:`/`t:` keys never change, and corruption is detected.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ewm_cache::schema::{
    from_ipc, hllset_lut_batch, lut_batch, lut_rows, sparse_slice_batch, sparse_slice_bits,
    tf_vec_batch, to_ipc, tree_leaves_batch, tree_levels_batch, turns_batch,
};
use ewm_cache::{content_key, ExtendedCache, Manifest, ARROW_CACHE_SCHEMA_VERSION};
use hllset_core::HLLSet;

fn tmp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "ewm-cache-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn sample_set(tokens: &[&str]) -> HLLSet {
    HLLSet::from_tokens(tokens.iter().map(|t| t.as_bytes()))
}

#[test]
fn ipc_round_trips_are_byte_identical() {
    // Every batch in §4: build → IPC → read → rebuild → IPC must be equal.
    let lut_rows_src = vec![
        (2u32, b"banana".to_vec()),
        (2u32, b"apple".to_vec()),
        (1u32, b"cherry".to_vec()),
    ];
    let batches = vec![
        hllset_lut_batch(&[
            ("G2".into(), "h:b".into(), 2),
            ("G1".into(), "h:a".into(), 1),
        ])
        .unwrap(),
        lut_batch(&lut_rows_src).unwrap(),
        tf_vec_batch(&vec![0.0; 32768]).unwrap(),
        tree_leaves_batch(&[
            ("h:b".into(), "l:1".into(), "v".into()),
            ("h:a".into(), "l:0".into(), "v".into()),
        ])
        .unwrap(),
        tree_levels_batch(&[(1, 1, "hh".into()), (0, 0, "h".into())]).unwrap(),
        turns_batch(&[(1, 2), (0, 1)]).unwrap(),
        sparse_slice_batch(&[7, 1, 7, 3]).unwrap(),
    ];

    for batch in batches {
        let bytes = to_ipc(&batch).unwrap();
        let read = from_ipc(&bytes).unwrap();
        assert_eq!(
            to_ipc(&read).unwrap(),
            bytes,
            "IPC round-trip must be byte-identical"
        );
    }

    // And the readers decode canonically (rows sorted, deduped where the
    // schema says so).
    let sparse = sparse_slice_batch(&[7, 1, 7, 3]).unwrap();
    assert_eq!(sparse_slice_bits(&sparse), vec![1, 3, 7]);

    let lut = lut_batch(&lut_rows_src).unwrap();
    let mut rows = lut_rows(&lut);
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, b"cherry".to_vec()),
            (2, b"apple".to_vec()),
            (2, b"banana".to_vec())
        ]
    );
}

#[test]
fn extended_cache_snapshots_validate_and_restore() {
    let dir = tmp_dir("extended");
    let cache = ExtendedCache::open(&dir).unwrap();

    // Content-addressed objects.
    let g1 = sample_set(&["apple", "banana"]);
    let g2 = sample_set(&["cherry"]);
    let g3 = sample_set(&["apple"]);
    let g1_key_before = g1.content_key();
    let sha1_g1 = cache.put_hllset(&g1).unwrap();
    let sha1_g2 = cache.put_hllset(&g2).unwrap();
    let sha1_g3 = cache.put_hllset(&g3).unwrap();
    let tf_vec = vec![0.5; 32768];
    let sha1_tf = cache.put_tfvec(&tf_vec).unwrap();

    // Tables.
    let lut = lut_batch(&[(1, b"apple".to_vec()), (2, b"banana".to_vec())]).unwrap();
    let turns = turns_batch(&[(0, 1), (0, 2), (1, 3)]).unwrap();
    let lut_sha1 = cache.write_table("ng:G1", &lut).unwrap();
    let turns_sha1 = cache.write_table("turns", &turns).unwrap();

    // Seal.
    let mut manifest = Manifest::new("tip-abc", "2026-09-16T00:00:00Z");
    manifest.schema_version = ARROW_CACHE_SCHEMA_VERSION;
    manifest.objects = BTreeMap::from([
        ("G1".to_string(), sha1_g1.clone()),
        ("G2".to_string(), sha1_g2.clone()),
        ("G3".to_string(), sha1_g3.clone()),
        ("tf_vec".to_string(), sha1_tf.clone()),
    ]);
    manifest.tables = BTreeMap::from([
        ("ng:G1".to_string(), lut_sha1),
        ("turns".to_string(), turns_sha1),
    ]);
    let (sealed, cid) = cache.seal(&manifest).unwrap();
    assert_eq!(sealed, manifest);
    assert!(cid.starts_with("c:"), "cache content key is c:<sha1>");
    assert_eq!(
        cid,
        content_key(&manifest.to_ipc().unwrap()),
        "content key is the SHA1 of the manifest bytes"
    );

    // Reopen and restore.
    let cache2 = ExtendedCache::open(&dir).unwrap();
    let opened = cache2.open_manifest().unwrap().unwrap();
    assert_eq!(opened, manifest, "manifest round-trips exactly");
    let snap = cache2.snapshot(&opened).unwrap();

    assert_eq!(snap.hllsets.len(), 3);
    assert_eq!(
        snap.hllsets["G1"].content_key(),
        g1_key_before,
        "cache round-trip changes no h: key"
    );
    assert_eq!(snap.hllsets["G1"].popcount(), g1.popcount());
    assert_eq!(snap.tf_vec.as_deref(), Some(tf_vec.as_slice()));
    assert_eq!(snap.tables.len(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn corruption_is_detected() {
    let dir = tmp_dir("corrupt");
    let cache = ExtendedCache::open(&dir).unwrap();
    let set = sample_set(&["apple"]);
    let sha1 = cache.put_hllset(&set).unwrap();

    let mut manifest = Manifest::new("tip", "t");
    manifest
        .objects
        .insert("G1".to_string(), sha1.clone());
    cache.seal(&manifest).unwrap();

    // Tamper with the object bytes: validation must fail with ShaMismatch.
    std::fs::write(
        dir.join("objects").join(format!("{sha1}.hllset")),
        b"tampered",
    )
    .unwrap();
    let err = cache.validate(&manifest).unwrap_err();
    assert!(
        matches!(err, ewm_cache::CacheError::ShaMismatch { .. }),
        "corruption detected: {err}"
    );

    // Wrong schema version is rejected outright.
    let mut bad = manifest.clone();
    bad.schema_version = 99;
    assert!(matches!(
        cache.seal(&bad),
        Err(ewm_cache::CacheError::SchemaVersion { found: 99 })
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hllset_keys_are_stable_across_the_cache() {
    let dir = tmp_dir("keys");
    let cache = ExtendedCache::open(&dir).unwrap();
    let set = sample_set(&["x", "y", "z"]);
    let key_before = set.content_key();
    let sha1 = cache.put_hllset(&set).unwrap();
    let restored = cache.get_hllset(&sha1).unwrap().unwrap();
    assert_eq!(restored.content_key(), key_before);
    assert_eq!(restored.content_key(), format!("h:{sha1}"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn shared_lut_merges_and_materializes_without_a_live_ingest() {
    use hllset_lut::LutIndex;

    let dir = tmp_dir("shared-lut");
    let cache = ExtendedCache::open(&dir).unwrap();

    // Two ingests contribute fibers to the shared ns:G1 table.
    let mut a = LutIndex::default();
    a.insert_token_at(b"fin_a".to_vec(), 5);
    a.insert_token_at(b"fin_b".to_vec(), 5);
    let mut b = LutIndex::default();
    b.insert_token_at(b"fin_b".to_vec(), 5); // duplicate (bit, token) — deduped
    b.insert_token_at(b"med_x".to_vec(), 9);

    cache.merge_lut("ns:G1", &a).unwrap();
    cache.merge_lut("ns:G1", &b).unwrap();

    let batch = cache.read_table("ns:G1").unwrap().unwrap();
    let rows = lut_rows(&batch);
    assert_eq!(
        rows,
        vec![
            (5, b"fin_a".to_vec()),
            (5, b"fin_b".to_vec()),
            (9, b"med_x".to_vec()),
        ]
    );

    // Materialize a state HLLSet against the shared table — no live ingest.
    let mut state = HLLSet::new();
    state.add_bit(5);
    let tokens = cache.materialize_lut(&state, "ns:G1").unwrap();
    assert_eq!(tokens.len(), 2);
    assert!(tokens.contains(&b"fin_a".to_vec()));
    assert!(tokens.contains(&b"fin_b".to_vec()));
    assert!(!tokens.contains(&b"med_x".to_vec()));

    // A missing table is a clean error.
    assert!(matches!(
        cache.materialize_lut(&state, "ng:G1"),
        Err(ewm_cache::CacheError::Missing { .. })
    ));

    let _ = std::fs::remove_dir_all(&dir);
}
