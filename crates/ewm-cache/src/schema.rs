//! Arrow schemas and canonical batch builders/readers for every cache
//! object in ARROW_CACHE.md §4.
//!
//! Every builder sorts its rows in canonical order before constructing the
//! batch, so two equal caches produce byte-identical IPC payloads
//! (IICA-friendly determinism, D1/§4).

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, Float64Array, StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;

use crate::storage::CacheError;

/// `hllset_lut` — `(name: Utf8, key: Utf8, th: UInt64)`, sorted by `(name, key)`.
pub fn hllset_lut_schema() -> Schema {
    Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("key", DataType::Utf8, false),
        Field::new("th", DataType::UInt64, false),
    ])
}

/// Build a canonical `hllset_lut` batch.
pub fn hllset_lut_batch(rows: &[(String, String, u64)]) -> Result<RecordBatch, ArrowError> {
    let mut rows: Vec<_> = rows.to_vec();
    rows.sort();
    let name = StringArray::from(rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>());
    let key = StringArray::from(rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>());
    let th = UInt64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>());
    RecordBatch::try_new(
        Arc::new(hllset_lut_schema()),
        vec![Arc::new(name), Arc::new(key), Arc::new(th)],
    )
}

/// Read an `hllset_lut` batch back into rows.
pub fn hllset_lut_rows(batch: &RecordBatch) -> Vec<(String, String, u64)> {
    let name = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name column");
    let key = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("key column");
    let th = batch
        .column(2)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("th column");
    (0..batch.num_rows())
        .map(|i| (name.value(i).to_string(), key.value(i).to_string(), th.value(i)))
        .collect()
}

/// A token LUT (`ng:G1` … `ns:G3`, `G2_2d` …) — `(bit: UInt32, token: Binary)`,
/// sorted by `(bit, token)`.
pub fn lut_schema() -> Schema {
    Schema::new(vec![
        Field::new("bit", DataType::UInt32, false),
        Field::new("token", DataType::Binary, false),
    ])
}

/// Build a canonical token-LUT batch.
pub fn lut_batch(rows: &[(u32, Vec<u8>)]) -> Result<RecordBatch, ArrowError> {
    let mut rows: Vec<_> = rows.to_vec();
    rows.sort();
    let bit = UInt32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>());
    let token = BinaryArray::from_iter_values(rows.iter().map(|r| r.1.as_slice()));
    RecordBatch::try_new(Arc::new(lut_schema()), vec![Arc::new(bit), Arc::new(token)])
}

/// Read a token-LUT batch back into rows.
pub fn lut_rows(batch: &RecordBatch) -> Vec<(u32, Vec<u8>)> {
    let bit = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("bit column");
    let token = batch
        .column(1)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("token column");
    (0..batch.num_rows())
        .map(|i| (bit.value(i), token.value(i).to_vec()))
        .collect()
}

/// `tf_vec` — `(index: UInt32, value: Float64)`, 32,768 rows, index ascending.
pub fn tf_vec_schema() -> Schema {
    Schema::new(vec![
        Field::new("index", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// Build the canonical `tf_vec` batch (requires exactly 32,768 values).
pub fn tf_vec_batch(values: &[f64]) -> Result<RecordBatch, ArrowError> {
    assert_eq!(
        values.len(),
        32768,
        "tf_vec must have 32768 entries (ARROW_CACHE.md §4.4)"
    );
    let index = UInt32Array::from((0..32768u32).collect::<Vec<_>>());
    let value = Float64Array::from(values.to_vec());
    RecordBatch::try_new(Arc::new(tf_vec_schema()), vec![Arc::new(index), Arc::new(value)])
}

/// Read a `tf_vec` batch back into values.
pub fn tf_vec_values(batch: &RecordBatch) -> Vec<f64> {
    let value = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("value column");
    (0..batch.num_rows()).map(|i| value.value(i)).collect()
}

/// `tree_leaves` — `(h: Utf8, lut: Utf8, view: Utf8)`, sorted by `(h, lut, view)`.
pub fn tree_leaves_schema() -> Schema {
    Schema::new(vec![
        Field::new("h", DataType::Utf8, false),
        Field::new("lut", DataType::Utf8, false),
        Field::new("view", DataType::Utf8, false),
    ])
}

/// Build a canonical `tree_leaves` batch.
pub fn tree_leaves_batch(rows: &[(String, String, String)]) -> Result<RecordBatch, ArrowError> {
    let mut rows: Vec<_> = rows.to_vec();
    rows.sort();
    let h = StringArray::from(rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>());
    let lut = StringArray::from(rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>());
    let view = StringArray::from(rows.iter().map(|r| r.2.as_str()).collect::<Vec<_>>());
    RecordBatch::try_new(
        Arc::new(tree_leaves_schema()),
        vec![Arc::new(h), Arc::new(lut), Arc::new(view)],
    )
}

/// Read a `tree_leaves` batch back into rows.
pub fn tree_leaves_rows(batch: &RecordBatch) -> Vec<(String, String, String)> {
    let h = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("h column");
    let lut = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("lut column");
    let view = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("view column");
    (0..batch.num_rows())
        .map(|i| {
            (
                h.value(i).to_string(),
                lut.value(i).to_string(),
                view.value(i).to_string(),
            )
        })
        .collect()
}

/// `tree_levels` — `(level: UInt32, index: UInt32, hash: Utf8)`, sorted by
/// `(level, index)`.
pub fn tree_levels_schema() -> Schema {
    Schema::new(vec![
        Field::new("level", DataType::UInt32, false),
        Field::new("index", DataType::UInt32, false),
        Field::new("hash", DataType::Utf8, false),
    ])
}

/// Build a canonical `tree_levels` batch.
pub fn tree_levels_batch(rows: &[(u32, u32, String)]) -> Result<RecordBatch, ArrowError> {
    let mut rows: Vec<_> = rows.to_vec();
    rows.sort();
    let level = UInt32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>());
    let index = UInt32Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>());
    let hash = StringArray::from(rows.iter().map(|r| r.2.as_str()).collect::<Vec<_>>());
    RecordBatch::try_new(
        Arc::new(tree_levels_schema()),
        vec![Arc::new(level), Arc::new(index), Arc::new(hash)],
    )
}

/// Read a `tree_levels` batch back into rows.
pub fn tree_levels_rows(batch: &RecordBatch) -> Vec<(u32, u32, String)> {
    let level = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("level column");
    let index = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("index column");
    let hash = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("hash column");
    (0..batch.num_rows())
        .map(|i| (level.value(i), index.value(i), hash.value(i).to_string()))
        .collect()
}

/// `turns` — `(turn: UInt64, id: UInt32)`, sorted by `(turn, id)`.
pub fn turns_schema() -> Schema {
    Schema::new(vec![
        Field::new("turn", DataType::UInt64, false),
        Field::new("id", DataType::UInt32, false),
    ])
}

/// Build a canonical `turns` batch.
pub fn turns_batch(rows: &[(u64, u32)]) -> Result<RecordBatch, ArrowError> {
    let mut rows: Vec<_> = rows.to_vec();
    rows.sort();
    let turn = UInt64Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>());
    let id = UInt32Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>());
    RecordBatch::try_new(Arc::new(turns_schema()), vec![Arc::new(turn), Arc::new(id)])
}

/// Read a `turns` batch back into rows.
pub fn turns_rows(batch: &RecordBatch) -> Vec<(u64, u32)> {
    let turn = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("turn column");
    let id = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("id column");
    (0..batch.num_rows())
        .map(|i| (turn.value(i), id.value(i)))
        .collect()
}

/// A sparse Gx slice — `(bit: UInt32)`, sorted by bit.
pub fn sparse_slice_schema() -> Schema {
    Schema::new(vec![Field::new("bit", DataType::UInt32, false)])
}

/// Build a canonical sparse-slice batch from the bits of an HLLSet.
pub fn sparse_slice_batch(bits: &[u32]) -> Result<RecordBatch, ArrowError> {
    let mut bits: Vec<_> = bits.to_vec();
    bits.sort();
    bits.dedup();
    let bit = UInt32Array::from(bits);
    RecordBatch::try_new(Arc::new(sparse_slice_schema()), vec![Arc::new(bit)])
}

/// Read a sparse-slice batch back into bits.
pub fn sparse_slice_bits(batch: &RecordBatch) -> Vec<u32> {
    let bit = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("bit column");
    (0..batch.num_rows()).map(|i| bit.value(i)).collect()
}

/// Serialize a batch to Arrow IPC file bytes.
pub fn to_ipc(batch: &RecordBatch) -> Result<Vec<u8>, CacheError> {
    let mut buf = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buf, &batch.schema())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Deserialize Arrow IPC file bytes into a single batch.
pub fn from_ipc(bytes: &[u8]) -> Result<RecordBatch, CacheError> {
    let reader = FileReader::try_new(std::io::Cursor::new(bytes), None)?;
    let mut batches = reader.into_iter();
    let batch = batches
        .next()
        .transpose()?
        .ok_or(CacheError::EmptyIpc)?;
    if batches.next().transpose()?.is_some() {
        return Err(CacheError::MultiBatchIpc);
    }
    Ok(batch)
}

// Keep the `ArrayRef` import referenced (re-export convenience for future
// backend code that builds batches dynamically).
#[allow(dead_code)]
fn _array_ref(_: ArrayRef) {}
