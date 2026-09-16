//! The ewm-git bridge: commit the fire log into a repository.
//!
//! Each [`FireRecord`] becomes one commit: the lattice state is the union of
//! the record's output values (`LatticeState::single`), the message carries
//! the op + input/output CIDs, and the parent is the repository HEAD. This
//! makes the dispatcher's trajectory durable in the state stack — the
//! operational graph commits exactly the states it produced.

use ewm_git::{LatticeState, ObjectId, ObjectStore, Repository};

use crate::dispatch::FireLog;
use crate::graph::OpGraph;
use hllset_core::HLLSet;

/// Commit every record of a fire log into `repo`, in fire order.
/// Returns the commit ids.
pub fn commit_fire_log<S: ObjectStore>(
    repo: &mut Repository<S>,
    graph: &OpGraph,
    log: &FireLog,
) -> Result<Vec<ObjectId>, ewm_git::StoreError> {
    let mut ids = Vec::with_capacity(log.records.len());
    for record in &log.records {
        let mut state_set = HLLSet::new();
        for cid in &record.outputs {
            if let Some(set) = graph.values.get(cid) {
                state_set = state_set.union(set);
            }
        }
        let state = LatticeState::single(&state_set);
        let parents: Vec<ObjectId> = repo.head().cloned().into_iter().collect();
        let message = format!(
            "op={} inputs={} outputs={}",
            record.op,
            record.inputs.join(","),
            record.outputs.join(",")
        );
        let id = repo.commit(&state, &parents, &message)?;
        ids.push(id);
    }
    Ok(ids)
}
