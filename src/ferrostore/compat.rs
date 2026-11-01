//! Read-only questions callers ask a store beyond the thirteen traits.
//! These sit beside the engine so `AnyStore` can ask a FerroStore what it
//! already asks a RedbStore.

use super::FerroStore;
use super::doc::{self, IMPORTS, LEDGER_ROOTS};
use fl_core::ids::{Kind, ProjectId};
use fl_core::iri::Iri;
use fl_core::store::StoreError;

impl FerroStore {
    /// The store's name in messages: what it was opened from: its file path, or `mem://` in memory.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Whether this store holds `id`. An I/O failure is an error, never `false`.
    pub fn owns(&self, id: &Iri) -> Result<bool, StoreError> {
        self.with_read(|txn| doc::owns(txn, &self.label, id))
    }

    /// The hash of the manifest `project` was imported from, or `None` if this
    /// store authors it. A project this store never held is `NotOwned`.
    pub fn imported_hash(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            doc::get_json(txn, IMPORTS, project.iri().as_str())
        })
    }

    /// Whether any project in this store has a routing map.
    pub fn holds_routing(&self) -> Result<bool, StoreError> {
        self.with_read(|txn| Ok(doc::holds_routing(txn)))
    }

    /// Whether this store records any ledger root.
    pub fn holds_a_ledger_root(&self) -> Result<bool, StoreError> {
        // Mirrors the redb presence check: any stored row counts, whatever JSON it holds.
        self.with_read(
            |txn| Ok(!doc::scan_json::<serde_json::Value>(txn, LEDGER_ROOTS)?.is_empty()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_label_names_the_store() {
        assert_eq!(FerroStore::in_memory().label(), "mem://");
    }

    #[test]
    fn any_stored_ledger_root_row_counts_whatever_json_it_holds() {
        let store = FerroStore::in_memory();
        assert!(!store.holds_a_ledger_root().unwrap());
        store
            .with_write(|txn| {
                doc::put_json(
                    txn,
                    LEDGER_ROOTS,
                    "R_1",
                    &serde_json::json!({"not": "a string"}),
                )
            })
            .unwrap();
        assert!(store.holds_a_ledger_root().unwrap());
    }
}
