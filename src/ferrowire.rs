//! `RedbStore` persistence for [`MessageStore`] (`FORMAT_WITH_WIRE = 7`, spec §2).

use fl_core::agent::{AgentRecord, AgentStatus};
use fl_core::at::At;
use fl_core::ferrowire::{
    DeliveryReceipt, DeliveryStatus, MessageKind, SendWireArgs, ThreadId, WireFilter,
    WireImportRejectReason, WireImportReport, WireInboxItem, WireMessage, resolve_recipients,
    validate_sender_name, validate_subject, validate_wire_body,
};
use fl_core::ids::{Kind, MessageId};
use fl_core::iri::Iri;
use fl_core::state::{StateError, validate_slug};
use fl_core::store::{MessageStore, StoreError};
use redb::{ReadableDatabase, ReadableTable, TableDefinition, TableHandle};

use crate::state::{AGENT_BY_NAME, AGENTS};
use crate::{
    ALIASES, FORMAT_WITH_WIRE, IDS, META, RedbStore, backend, decode, locate_in, mint, raise_format,
};

pub(crate) const WIRE_MESSAGES: TableDefinition<&str, &str> = TableDefinition::new("wire_messages");
pub(crate) const WIRE_MSG_LOG: TableDefinition<u64, &str> = TableDefinition::new("wire_msg_log");
pub(crate) const WIRE_MSG_SEQ: TableDefinition<&str, u64> = TableDefinition::new("wire_msg_seq");
pub(crate) const WIRE_INBOX: TableDefinition<(&str, &str), &str> =
    TableDefinition::new("wire_inbox");
pub(crate) const WIRE_RCPT_LOG: TableDefinition<u64, &str> = TableDefinition::new("wire_rcpt_log");
pub(crate) const WIRE_THREADS: TableDefinition<(&str, &str), ()> =
    TableDefinition::new("wire_threads");
pub(crate) const WIRE_REPLIES: TableDefinition<(&str, &str), ()> =
    TableDefinition::new("wire_replies");

const NEXT_WIRE_MSG_SEQ: &str = "next_wire_msg_seq";
const NEXT_WIRE_RCPT_SEQ: &str = "next_wire_rcpt_seq";

impl RedbStore {
    fn wtx_has_table(tx: &redb::WriteTransaction, name: &str) -> Result<bool, StoreError> {
        Ok(tx.list_tables().map_err(backend)?.any(|h| h.name() == name))
    }

    fn index_message_id_in(&self, tx: &redb::WriteTransaction, id: &Iri) -> Result<(), StoreError> {
        let already = {
            let ids = tx.open_table(IDS).map_err(backend)?;
            ids.get(id.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned())
        };
        if let Some(wire) = already {
            if wire == "alias" {
                return Err(StoreError::WrongKind {
                    id: id.clone(),
                    expected: Kind::Message,
                    found: Kind::Agent,
                });
            }
            let found = Kind::from_wire(&wire)
                .ok_or_else(|| decode(format!("unknown kind `{wire}` for {id}")))?;
            if found != Kind::Message {
                return Err(StoreError::WrongKind {
                    id: id.clone(),
                    expected: Kind::Message,
                    found,
                });
            }
            return Ok(());
        }
        tx.open_table(IDS)
            .map_err(backend)?
            .insert(id.as_str(), Kind::Message.as_wire())
            .map_err(backend)?;
        Ok(())
    }

    fn append_wire_message_in(
        &self,
        tx: &redb::WriteTransaction,
        msg: &WireMessage,
    ) -> Result<(), StoreError> {
        let already_logged = tx
            .open_table(WIRE_MSG_SEQ)
            .map_err(backend)?
            .get(msg.id.as_str())
            .map_err(backend)?
            .is_some();
        if !already_logged {
            let seq = {
                let mut meta = tx.open_table(META).map_err(backend)?;
                let next = meta
                    .get(NEXT_WIRE_MSG_SEQ)
                    .map_err(backend)?
                    .map(|g| g.value())
                    .unwrap_or(0)
                    + 1;
                meta.insert(NEXT_WIRE_MSG_SEQ, next).map_err(backend)?;
                next
            };
            tx.open_table(WIRE_MSG_LOG)
                .map_err(backend)?
                .insert(seq, msg.id.as_str())
                .map_err(backend)?;
            tx.open_table(WIRE_MSG_SEQ)
                .map_err(backend)?
                .insert(msg.id.as_str(), seq)
                .map_err(backend)?;
        }
        tx.open_table(WIRE_THREADS)
            .map_err(backend)?
            .insert((msg.thread_id.as_str(), msg.id.as_str()), ())
            .map_err(backend)?;
        if let Some(ref corr) = msg.correlation_id {
            tx.open_table(WIRE_REPLIES)
                .map_err(backend)?
                .insert((corr.as_str(), msg.id.as_str()), ())
                .map_err(backend)?;
        }
        let json = serde_json::to_string(msg).map_err(backend)?;
        tx.open_table(WIRE_MESSAGES)
            .map_err(backend)?
            .insert(msg.id.as_str(), json.as_str())
            .map_err(backend)?;
        Ok(())
    }

    fn append_wire_receipt_in(
        &self,
        tx: &redb::WriteTransaction,
        rcpt: &DeliveryReceipt,
    ) -> Result<(), StoreError> {
        let seq = {
            let mut meta = tx.open_table(META).map_err(backend)?;
            let next = meta
                .get(NEXT_WIRE_RCPT_SEQ)
                .map_err(backend)?
                .map(|g| g.value())
                .unwrap_or(0)
                + 1;
            meta.insert(NEXT_WIRE_RCPT_SEQ, next).map_err(backend)?;
            next
        };
        let json = serde_json::to_string(rcpt).map_err(backend)?;
        tx.open_table(WIRE_RCPT_LOG)
            .map_err(backend)?
            .insert(seq, json.as_str())
            .map_err(backend)?;
        Ok(())
    }

    fn all_agents_in_wtx(
        &self,
        tx: &redb::WriteTransaction,
    ) -> Result<Vec<AgentRecord>, StoreError> {
        if !Self::wtx_has_table(tx, AGENTS.name())? {
            return Ok(Vec::new());
        }
        let table = tx.open_table(AGENTS).map_err(backend)?;
        let mut out = Vec::new();
        for row in table.iter().map_err(backend)? {
            let (_, v) = row.map_err(backend)?;
            let rec: AgentRecord = serde_json::from_str(v.value()).map_err(decode)?;
            out.push(rec);
        }
        Ok(out)
    }

    fn resolve_iri_in_rtx(&self, tx: &redb::ReadTransaction, id: &Iri) -> Result<Iri, StoreError> {
        let ids = tx.open_table(IDS).map_err(backend)?;
        match locate_in(&self.label, id, &ids, || {
            tx.open_table(ALIASES).map_err(backend)
        }) {
            Ok((primary, _)) => Ok(primary),
            Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
            Err(e) => Err(e),
        }
    }

    fn resolve_iri_in_wtx(&self, tx: &redb::WriteTransaction, id: &Iri) -> Result<Iri, StoreError> {
        let ids = tx.open_table(IDS).map_err(backend)?;
        match locate_in(&self.label, id, &ids, || {
            tx.open_table(ALIASES).map_err(backend)
        }) {
            Ok((primary, _)) => Ok(primary),
            Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
            Err(e) => Err(e),
        }
    }

    fn recipient_candidates_in_rtx(
        &self,
        tx: &redb::ReadTransaction,
        query: &str,
    ) -> Result<(Option<Iri>, Option<String>), StoreError> {
        let trimmed = query.trim();
        let agents_table = match tx.open_table(AGENTS) {
            Ok(t) => Some(t),
            Err(redb::TableError::TableDoesNotExist(_)) => None,
            Err(e) => return Err(backend(e)),
        };
        let by_name_table = match tx.open_table(AGENT_BY_NAME) {
            Ok(t) => Some(t),
            Err(redb::TableError::TableDoesNotExist(_)) => None,
            Err(e) => return Err(backend(e)),
        };

        let lookup_by_name = |k: &str| -> Result<Option<Iri>, StoreError> {
            let Some(ref t) = by_name_table else {
                return Ok(None);
            };
            let Some(g) = t.get(k).map_err(backend)? else {
                return Ok(None);
            };
            let iri = Iri::parse(g.value()).map_err(decode)?;
            self.resolve_iri_in_rtx(tx, &iri).map(Some)
        };

        if let Ok(iri) = Iri::parse(trimmed) {
            let resolved_iri = self.resolve_iri_in_rtx(tx, &iri)?;
            if let Some(ref at) = agents_table
                && let Some(g) = at.get(resolved_iri.as_str()).map_err(backend)?
            {
                let rec: AgentRecord = serde_json::from_str(g.value()).map_err(decode)?;
                return Ok((Some(resolved_iri), Some(rec.name_key)));
            }
            if let Some(rest) = trimmed.strip_prefix("urn:fl:agent:")
                && let Ok(k) = validate_slug(rest)
            {
                return Ok((lookup_by_name(&k)?, Some(k)));
            }
            return Ok((Some(resolved_iri), None));
        }
        if let Ok(k) = validate_slug(trimmed) {
            return Ok((lookup_by_name(&k)?, Some(k)));
        }
        Ok((None, None))
    }

    fn recipient_candidates_in_wtx(
        &self,
        tx: &redb::WriteTransaction,
        query: &str,
    ) -> Result<(Option<Iri>, Option<String>), StoreError> {
        let trimmed = query.trim();
        let has_agents = Self::wtx_has_table(tx, AGENTS.name())?;
        let has_by_name = Self::wtx_has_table(tx, AGENT_BY_NAME.name())?;

        let lookup_by_name = |k: &str| -> Result<Option<Iri>, StoreError> {
            if !has_by_name {
                return Ok(None);
            }
            let raw = {
                let t = tx.open_table(AGENT_BY_NAME).map_err(backend)?;
                let Some(g) = t.get(k).map_err(backend)? else {
                    return Ok(None);
                };
                g.value().to_owned()
            };
            let iri = Iri::parse(&raw).map_err(decode)?;
            self.resolve_iri_in_wtx(tx, &iri).map(Some)
        };

        if let Ok(iri) = Iri::parse(trimmed) {
            let resolved_iri = self.resolve_iri_in_wtx(tx, &iri)?;
            if has_agents {
                let raw_rec = {
                    let at = tx.open_table(AGENTS).map_err(backend)?;
                    at.get(resolved_iri.as_str())
                        .map_err(backend)?
                        .map(|g| g.value().to_owned())
                };
                if let Some(json) = raw_rec {
                    let rec: AgentRecord = serde_json::from_str(&json).map_err(decode)?;
                    return Ok((Some(resolved_iri), Some(rec.name_key)));
                }
            }
            if let Some(rest) = trimmed.strip_prefix("urn:fl:agent:")
                && let Ok(k) = validate_slug(rest)
            {
                return Ok((lookup_by_name(&k)?, Some(k)));
            }
            return Ok((Some(resolved_iri), None));
        }
        if let Ok(k) = validate_slug(trimmed) {
            return Ok((lookup_by_name(&k)?, Some(k)));
        }
        Ok((None, None))
    }

    fn receipt_matches_candidates(
        rcpt: &DeliveryReceipt,
        rcpt_resolved_id: &Iri,
        agent_id: Option<&Iri>,
        name_key: Option<&str>,
        raw_query: &str,
    ) -> bool {
        let trimmed = raw_query.trim();
        if let Some(aid) = agent_id
            && rcpt_resolved_id == aid
        {
            return true;
        }
        if let Some(nk) = name_key
            && (rcpt.recipient_name == nk
                || rcpt.recipient_id.as_str() == format!("urn:fl:agent:{nk}"))
        {
            return true;
        }
        rcpt.recipient_id.as_str() == trimmed || rcpt.recipient_name.eq_ignore_ascii_case(trimmed)
    }

    /// Verify FerroWire on-disk table invariants (spec §2.3).
    pub fn verify_wire_invariants(&self) -> Result<(), StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let ids = tx.open_table(IDS).map_err(backend)?;
        let messages_table = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => Some(t),
            Err(redb::TableError::TableDoesNotExist(_)) => None,
            Err(e) => return Err(backend(e)),
        };

        // 1. Every `Kind::Message` entry in `IDS` must exist in `WIRE_MESSAGES`.
        for row in ids.iter().map_err(backend)? {
            let (k, v) = row.map_err(backend)?;
            if v.value() == Kind::Message.as_wire() {
                let exists = match messages_table {
                    Some(ref t) => t.get(k.value()).map_err(backend)?.is_some(),
                    None => false,
                };
                if !exists {
                    return Err(decode(format!(
                        "IDS holds message `{}` with no row in wire_messages",
                        k.value()
                    )));
                }
            }
        }

        let Some(ref messages) = messages_table else {
            return Ok(());
        };

        // 2. Every row in `WIRE_MESSAGES` must deserialize cleanly as a `WireMessage`
        //    whose `id` matches its table key and has a `Kind::Message` entry in `IDS`.
        for row in messages.iter().map_err(backend)? {
            let (k, v) = row.map_err(backend)?;
            let msg: WireMessage = serde_json::from_str(v.value()).map_err(decode)?;
            if msg.id.as_str() != k.value() {
                return Err(decode(format!(
                    "wire_messages key `{}` does not match message.id `{}`",
                    k.value(),
                    msg.id
                )));
            }
            let kind_wire = ids
                .get(k.value())
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            if kind_wire.as_deref() != Some(Kind::Message.as_wire()) {
                return Err(decode(format!(
                    "wire_messages row `{}` is missing Kind::Message in IDS (found {:?})",
                    k.value(),
                    kind_wire
                )));
            }
        }

        // 3. Every row in `WIRE_INBOX` must deserialize cleanly as a `DeliveryReceipt`
        //    whose `(recipient_id, message_id)` matches its key and whose `message_id` exists in `WIRE_MESSAGES`.
        if let Ok(inbox) = tx.open_table(WIRE_INBOX) {
            for row in inbox.iter().map_err(backend)? {
                let (k, v) = row.map_err(backend)?;
                let (rec_key, msg_key) = k.value();
                let rcpt: DeliveryReceipt = serde_json::from_str(v.value()).map_err(decode)?;
                if rcpt.recipient_id.as_str() != rec_key || rcpt.message_id.as_str() != msg_key {
                    return Err(decode(format!(
                        "wire_inbox key `({rec_key}, {msg_key})` does not match receipt `({}, {})`",
                        rcpt.recipient_id, rcpt.message_id
                    )));
                }
                if messages.get(msg_key).map_err(backend)?.is_none() {
                    return Err(decode(format!(
                        "wire_inbox receipt for `{msg_key}` references missing message"
                    )));
                }
            }
        }

        // 4. Every `(thread_id, message_id)` in `WIRE_THREADS` and `(correlation_id, message_id)`
        //    in `WIRE_REPLIES` must reference an existing `message_id` in `WIRE_MESSAGES`.
        if let Ok(threads) = tx.open_table(WIRE_THREADS) {
            for row in threads.iter().map_err(backend)? {
                let (k, _) = row.map_err(backend)?;
                let (_tid, mid) = k.value();
                if messages.get(mid).map_err(backend)?.is_none() {
                    return Err(decode(format!(
                        "wire_threads entry references missing message `{mid}`"
                    )));
                }
            }
        }
        if let Ok(replies) = tx.open_table(WIRE_REPLIES) {
            for row in replies.iter().map_err(backend)? {
                let (k, _) = row.map_err(backend)?;
                let (_cid, mid) = k.value();
                if messages.get(mid).map_err(backend)?.is_none() {
                    return Err(decode(format!(
                        "wire_replies entry references missing message `{mid}`"
                    )));
                }
            }
        }

        Ok(())
    }
}

impl MessageStore for RedbStore {
    fn send_wire(
        &self,
        args: SendWireArgs,
    ) -> Result<(WireMessage, Vec<DeliveryReceipt>), StoreError> {
        let subject =
            validate_subject(&args.subject).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let body =
            validate_wire_body(&args.body).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let sender_name = validate_sender_name(&args.sender_name)
            .map_err(|e| StoreError::InvalidState(e.to_string()))?;

        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WIRE)?;

        let (parent_thread, correlation_id) = match &args.reply_to {
            Some(parent_id) => {
                let parent_opt: Option<WireMessage> = {
                    let t = tx.open_table(WIRE_MESSAGES).map_err(backend)?;
                    match t.get(parent_id.as_str()).map_err(backend)? {
                        Some(g) => Some(serde_json::from_str(g.value()).map_err(decode)?),
                        None => None,
                    }
                };
                let Some(parent) = parent_opt else {
                    return Err(StoreError::MessageNotFound(parent_id.clone()));
                };
                let corr = if parent.kind == MessageKind::Ask {
                    Some(parent.id.clone())
                } else if parent.correlation_id.is_some() {
                    parent.correlation_id.clone()
                } else if args.kind == MessageKind::Reply {
                    Some(parent.id.clone())
                } else {
                    None
                };
                (Some(parent.thread_id), corr)
            }
            None => {
                if args.kind == MessageKind::Reply {
                    return Err(StoreError::InvalidState(
                        StateError::ReplyRequiresParent.to_string(),
                    ));
                }
                (None, None)
            }
        };

        let msg_id = match args.id {
            Some(id) => {
                self.index_message_id_in(&tx, id.iri())?;
                id
            }
            None => {
                let iri = mint()?;
                self.index_message_id_in(&tx, &iri)?;
                MessageId(iri)
            }
        };

        let thread_id = args
            .thread_id
            .or(parent_thread)
            .unwrap_or_else(|| msg_id.clone());
        let durable = args.durable
            || matches!(
                args.kind,
                MessageKind::Ask
                    | MessageKind::Handoff
                    | MessageKind::TaskOffer
                    | MessageKind::ApprovalRequest
            );

        let all_agents = self.all_agents_in_wtx(&tx)?;
        let sender_id = args.sender_id.or_else(|| {
            all_agents
                .iter()
                .find(|a| a.status == AgentStatus::Active && a.name_key == sender_name)
                .map(|a| a.id.clone())
        });

        let msg = WireMessage {
            id: msg_id.clone(),
            thread_id,
            reply_to: args.reply_to,
            correlation_id,
            sender_id,
            sender_name,
            sender_host: args.sender_host,
            to: args.to,
            kind: args.kind,
            urgency: args.urgency,
            durable,
            subject,
            body,
            created_at: args.at.clone(),
            deadline_at: args.deadline_at,
        };
        msg.validate()
            .map_err(|e| StoreError::InvalidState(e.to_string()))?;

        let recipients = resolve_recipients(
            &msg.to,
            msg.sender_id.as_ref(),
            &msg.sender_name,
            &all_agents,
        );

        self.append_wire_message_in(&tx, &msg)?;

        let mut receipts = Vec::with_capacity(recipients.len());
        {
            let mut inbox = tx.open_table(WIRE_INBOX).map_err(backend)?;
            for (recipient_id, recipient_name) in recipients {
                let rcpt = DeliveryReceipt {
                    message_id: msg.id.clone(),
                    recipient_id: recipient_id.clone(),
                    recipient_name,
                    status: DeliveryStatus::Pending,
                    attempts: 0,
                    updated_at: args.at.clone(),
                    acked_at: None,
                };
                let json = serde_json::to_string(&rcpt).map_err(backend)?;
                inbox
                    .insert((recipient_id.as_str(), msg.id.as_str()), json.as_str())
                    .map_err(backend)?;
                self.append_wire_receipt_in(&tx, &rcpt)?;
                receipts.push(rcpt);
            }
        }

        tx.commit().map_err(backend)?;
        Ok((msg, receipts))
    }

    fn get_wire_message(&self, id: &MessageId) -> Result<Option<WireMessage>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        match table.get(id.as_str()).map_err(backend)? {
            Some(g) => Ok(Some(serde_json::from_str(g.value()).map_err(decode)?)),
            None => Ok(None),
        }
    }

    fn list_wire_inbox(&self, filter: &WireFilter) -> Result<Vec<WireInboxItem>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let inbox = match tx.open_table(WIRE_INBOX) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let messages = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };

        let candidates = match &filter.recipient {
            Some(q) => Some(self.recipient_candidates_in_rtx(&tx, q)?),
            None => None,
        };

        let mut items = Vec::new();
        for row in inbox.iter().map_err(backend)? {
            let (_, v) = row.map_err(backend)?;
            let rcpt: DeliveryReceipt = serde_json::from_str(v.value()).map_err(decode)?;
            if let (Some(q), Some((aid, nk))) = (&filter.recipient, &candidates) {
                let rcpt_resolved = self.resolve_iri_in_rtx(&tx, &rcpt.recipient_id)?;
                if !Self::receipt_matches_candidates(
                    &rcpt,
                    &rcpt_resolved,
                    aid.as_ref(),
                    nk.as_deref(),
                    q,
                ) {
                    continue;
                }
            }
            if filter.unread_only
                && matches!(rcpt.status, DeliveryStatus::Acked | DeliveryStatus::Expired)
            {
                continue;
            }
            let Some(msg_guard) = messages.get(rcpt.message_id.as_str()).map_err(backend)? else {
                continue;
            };
            let msg: WireMessage = serde_json::from_str(msg_guard.value()).map_err(decode)?;
            if let Some(ref tid) = filter.thread_id
                && &msg.thread_id != tid
            {
                continue;
            }
            if let Some(ref cid) = filter.correlation_id
                && msg.correlation_id.as_ref() != Some(cid)
            {
                continue;
            }
            items.push(WireInboxItem {
                message: msg,
                receipt: rcpt,
            });
        }

        items.sort_by(|a, b| {
            (&a.message.id, &a.receipt.recipient_id).cmp(&(&b.message.id, &b.receipt.recipient_id))
        });
        if let Some(limit) = filter.limit
            && items.len() > limit
        {
            items.truncate(limit);
        }
        Ok(items)
    }

    fn list_wire_thread(&self, thread_id: &ThreadId) -> Result<Vec<WireMessage>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let threads = match tx.open_table(WIRE_THREADS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let messages = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };

        let mut out = Vec::new();
        for row in threads.range((thread_id.as_str(), "")..).map_err(backend)? {
            let (k, _) = row.map_err(backend)?;
            let (tid, mid) = k.value();
            if tid != thread_id.as_str() {
                break;
            }
            if let Some(g) = messages.get(mid).map_err(backend)? {
                let msg: WireMessage = serde_json::from_str(g.value()).map_err(decode)?;
                out.push(msg);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    fn list_wire_replies(
        &self,
        correlation_id: &MessageId,
    ) -> Result<Vec<WireMessage>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let replies = match tx.open_table(WIRE_REPLIES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let messages = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };

        let mut out = Vec::new();
        for row in replies
            .range((correlation_id.as_str(), "")..)
            .map_err(backend)?
        {
            let (k, _) = row.map_err(backend)?;
            let (cid, mid) = k.value();
            if cid != correlation_id.as_str() {
                break;
            }
            if let Some(g) = messages.get(mid).map_err(backend)? {
                let msg: WireMessage = serde_json::from_str(g.value()).map_err(decode)?;
                out.push(msg);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    fn list_wire_receipts(
        &self,
        message_id: &MessageId,
    ) -> Result<Vec<DeliveryReceipt>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let messages = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => {
                return Err(StoreError::MessageNotFound(message_id.clone()));
            }
            Err(e) => return Err(backend(e)),
        };
        if messages
            .get(message_id.as_str())
            .map_err(backend)?
            .is_none()
        {
            return Err(StoreError::MessageNotFound(message_id.clone()));
        }
        let inbox = match tx.open_table(WIRE_INBOX) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in inbox.iter().map_err(backend)? {
            let (k, v) = row.map_err(backend)?;
            let (_, mid) = k.value();
            if mid == message_id.as_str() {
                let rcpt: DeliveryReceipt = serde_json::from_str(v.value()).map_err(decode)?;
                out.push(rcpt);
            }
        }
        out.sort_by(|a, b| a.recipient_id.cmp(&b.recipient_id));
        Ok(out)
    }

    fn update_wire_receipt(
        &self,
        message_id: &MessageId,
        recipient: &str,
        status: DeliveryStatus,
        increment_attempts: bool,
        at: At,
    ) -> Result<Option<DeliveryReceipt>, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WIRE)?;

        let msg_exists = tx
            .open_table(WIRE_MESSAGES)
            .map_err(backend)?
            .get(message_id.as_str())
            .map_err(backend)?
            .is_some();
        if !msg_exists {
            return Err(StoreError::MessageNotFound(message_id.clone()));
        }

        let (aid, nk) = self.recipient_candidates_in_wtx(&tx, recipient)?;
        let mut matched: Option<(String, DeliveryReceipt)> = None;
        {
            let inbox = tx.open_table(WIRE_INBOX).map_err(backend)?;
            let mut candidates_raw = Vec::new();
            for row in inbox.iter().map_err(backend)? {
                let (k, v) = row.map_err(backend)?;
                let (rec_id_str, mid_str) = k.value();
                if mid_str == message_id.as_str() {
                    let rcpt: DeliveryReceipt = serde_json::from_str(v.value()).map_err(decode)?;
                    candidates_raw.push((rec_id_str.to_owned(), rcpt));
                }
            }
            drop(inbox);
            for (rec_key, rcpt) in candidates_raw {
                let rcpt_resolved = self.resolve_iri_in_wtx(&tx, &rcpt.recipient_id)?;
                if Self::receipt_matches_candidates(
                    &rcpt,
                    &rcpt_resolved,
                    aid.as_ref(),
                    nk.as_deref(),
                    recipient,
                ) {
                    matched = Some((rec_key, rcpt));
                    break;
                }
            }
        }

        let Some((rec_key, existing)) = matched else {
            return Ok(None);
        };

        let attempts = if increment_attempts {
            existing.attempts.saturating_add(1)
        } else {
            existing.attempts
        };
        let incoming = DeliveryReceipt {
            message_id: existing.message_id.clone(),
            recipient_id: existing.recipient_id.clone(),
            recipient_name: existing.recipient_name.clone(),
            status,
            attempts,
            updated_at: at.clone(),
            acked_at: if status == DeliveryStatus::Acked {
                Some(at)
            } else {
                None
            },
        };
        let merged = existing.merge(&incoming);
        let json = serde_json::to_string(&merged).map_err(backend)?;
        tx.open_table(WIRE_INBOX)
            .map_err(backend)?
            .insert((rec_key.as_str(), message_id.as_str()), json.as_str())
            .map_err(backend)?;
        self.append_wire_receipt_in(&tx, &merged)?;
        tx.commit().map_err(backend)?;
        Ok(Some(merged))
    }

    fn wire_messages_after(&self, after_seq: u64) -> Result<Vec<(u64, WireMessage)>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let log = match tx.open_table(WIRE_MSG_LOG) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let messages = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in log
            .range((
                std::ops::Bound::Excluded(after_seq),
                std::ops::Bound::Unbounded,
            ))
            .map_err(backend)?
        {
            let (k, v) = row.map_err(backend)?;
            if let Some(msg_guard) = messages.get(v.value()).map_err(backend)? {
                let msg: WireMessage = serde_json::from_str(msg_guard.value()).map_err(decode)?;
                out.push((k.value(), msg));
            }
        }
        Ok(out)
    }

    fn wire_receipts_after(
        &self,
        after_seq: u64,
    ) -> Result<Vec<(u64, DeliveryReceipt)>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let log = match tx.open_table(WIRE_RCPT_LOG) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let messages = match tx.open_table(WIRE_MESSAGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in log
            .range((
                std::ops::Bound::Excluded(after_seq),
                std::ops::Bound::Unbounded,
            ))
            .map_err(backend)?
        {
            let (k, v) = row.map_err(backend)?;
            let rcpt: DeliveryReceipt = serde_json::from_str(v.value()).map_err(decode)?;
            if messages
                .get(rcpt.message_id.as_str())
                .map_err(backend)?
                .is_some()
            {
                out.push((k.value(), rcpt));
            }
        }
        Ok(out)
    }

    fn import_wire_batch(
        &self,
        messages: &[WireMessage],
        receipts: &[DeliveryReceipt],
    ) -> Result<WireImportReport, StoreError> {
        if messages.is_empty() && receipts.is_empty() {
            return Ok(WireImportReport::default());
        }
        let mut report = WireImportReport::default();
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WIRE)?;
        let all_agents = self.all_agents_in_wtx(&tx)?;

        for msg in messages {
            // Each message stands or falls alone: a skip writes nothing for it.
            if let Err(e) = msg.validate() {
                report.reject(
                    &msg.id,
                    false,
                    WireImportRejectReason::InvalidMessage,
                    e.to_string(),
                );
                continue;
            }
            let existing: Option<WireMessage> = {
                let t = tx.open_table(WIRE_MESSAGES).map_err(backend)?;
                match t.get(msg.id.as_str()).map_err(backend)? {
                    Some(g) => Some(serde_json::from_str(g.value()).map_err(decode)?),
                    None => None,
                }
            };
            if let Some(ex) = existing {
                if &ex != msg {
                    report.reject(
                        &msg.id,
                        false,
                        WireImportRejectReason::TamperedMessage,
                        "imported wire message differs from stored copy",
                    );
                }
                continue;
            }
            match self.index_message_id_in(&tx, msg.id.iri()) {
                Ok(()) => {}
                Err(e @ StoreError::WrongKind { .. }) => {
                    report.reject(
                        &msg.id,
                        false,
                        WireImportRejectReason::IdCollision,
                        e.to_string(),
                    );
                    continue;
                }
                Err(e) => return Err(e),
            }
            self.append_wire_message_in(&tx, msg)?;
            report.imported_messages += 1;

            let resolved = resolve_recipients(
                &msg.to,
                msg.sender_id.as_ref(),
                &msg.sender_name,
                &all_agents,
            );
            for (rec_id, rec_name) in resolved {
                let in_incoming = receipts.iter().any(|r| {
                    r.message_id == msg.id
                        && (r.recipient_id == rec_id || r.recipient_name == rec_name)
                });
                let already_in_inbox = tx
                    .open_table(WIRE_INBOX)
                    .map_err(backend)?
                    .get((rec_id.as_str(), msg.id.as_str()))
                    .map_err(backend)?
                    .is_some();
                if !in_incoming && !already_in_inbox {
                    let rcpt = DeliveryReceipt {
                        message_id: msg.id.clone(),
                        recipient_id: rec_id.clone(),
                        recipient_name: rec_name,
                        status: DeliveryStatus::Pending,
                        attempts: 0,
                        updated_at: msg.created_at.clone(),
                        acked_at: None,
                    };
                    let json = serde_json::to_string(&rcpt).map_err(backend)?;
                    tx.open_table(WIRE_INBOX)
                        .map_err(backend)?
                        .insert((rec_id.as_str(), msg.id.as_str()), json.as_str())
                        .map_err(backend)?;
                    self.append_wire_receipt_in(&tx, &rcpt)?;
                }
            }
        }

        for rcpt in receipts {
            let msg_exists = tx
                .open_table(WIRE_MESSAGES)
                .map_err(backend)?
                .get(rcpt.message_id.as_str())
                .map_err(backend)?
                .is_some();
            if !msg_exists {
                report.reject(
                    &rcpt.message_id,
                    true,
                    WireImportRejectReason::UnknownMessage,
                    "no such wire message",
                );
                continue;
            }
            let existing: Option<DeliveryReceipt> = {
                let inbox = tx.open_table(WIRE_INBOX).map_err(backend)?;
                match inbox
                    .get((rcpt.recipient_id.as_str(), rcpt.message_id.as_str()))
                    .map_err(backend)?
                {
                    Some(g) => Some(serde_json::from_str(g.value()).map_err(decode)?),
                    None => None,
                }
            };
            match existing {
                Some(ex) => {
                    let merged = ex.merge(rcpt);
                    if merged != ex {
                        let json = serde_json::to_string(&merged).map_err(backend)?;
                        tx.open_table(WIRE_INBOX)
                            .map_err(backend)?
                            .insert(
                                (rcpt.recipient_id.as_str(), rcpt.message_id.as_str()),
                                json.as_str(),
                            )
                            .map_err(backend)?;
                        self.append_wire_receipt_in(&tx, &merged)?;
                        report.imported_receipts += 1;
                    }
                }
                None => {
                    let json = serde_json::to_string(rcpt).map_err(backend)?;
                    tx.open_table(WIRE_INBOX)
                        .map_err(backend)?
                        .insert(
                            (rcpt.recipient_id.as_str(), rcpt.message_id.as_str()),
                            json.as_str(),
                        )
                        .map_err(backend)?;
                    self.append_wire_receipt_in(&tx, rcpt)?;
                    report.imported_receipts += 1;
                }
            }
        }

        tx.commit().map_err(backend)?;
        Ok(report)
    }

    fn prune_wire(&self, before: At) -> Result<usize, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        if !Self::wtx_has_table(&tx, WIRE_MESSAGES.name())? {
            return Ok(0);
        }
        raise_format(&tx, FORMAT_WITH_WIRE)?;

        let all_messages: Vec<WireMessage> = {
            let t = tx.open_table(WIRE_MESSAGES).map_err(backend)?;
            let mut out = Vec::new();
            for row in t.iter().map_err(backend)? {
                let (_, v) = row.map_err(backend)?;
                out.push(serde_json::from_str(v.value()).map_err(decode)?);
            }
            out
        };

        let all_receipts: Vec<DeliveryReceipt> = if Self::wtx_has_table(&tx, WIRE_INBOX.name())? {
            let t = tx.open_table(WIRE_INBOX).map_err(backend)?;
            let mut out = Vec::new();
            for row in t.iter().map_err(backend)? {
                let (_, v) = row.map_err(backend)?;
                out.push(serde_json::from_str(v.value()).map_err(decode)?);
            }
            out
        } else {
            Vec::new()
        };

        let to_prune: Vec<WireMessage> = all_messages
            .into_iter()
            .filter(|msg| !msg.durable && msg.created_at < before)
            .filter(|msg| {
                all_receipts
                    .iter()
                    .filter(|r| r.message_id == msg.id)
                    .all(|r| matches!(r.status, DeliveryStatus::Acked | DeliveryStatus::Expired))
            })
            .collect();

        let count = to_prune.len();
        if count == 0 {
            return Ok(0);
        }

        let pruned_ids: std::collections::BTreeSet<MessageId> =
            to_prune.iter().map(|m| m.id.clone()).collect();

        for msg in to_prune {
            tx.open_table(WIRE_MESSAGES)
                .map_err(backend)?
                .remove(msg.id.as_str())
                .map_err(backend)?;
            tx.open_table(IDS)
                .map_err(backend)?
                .remove(msg.id.as_str())
                .map_err(backend)?;
            tx.open_table(WIRE_THREADS)
                .map_err(backend)?
                .remove((msg.thread_id.as_str(), msg.id.as_str()))
                .map_err(backend)?;
            if let Some(ref corr) = msg.correlation_id {
                tx.open_table(WIRE_REPLIES)
                    .map_err(backend)?
                    .remove((corr.as_str(), msg.id.as_str()))
                    .map_err(backend)?;
            }
            let seq = tx
                .open_table(WIRE_MSG_SEQ)
                .map_err(backend)?
                .remove(msg.id.as_str())
                .map_err(backend)?
                .map(|g| g.value());
            if let Some(s) = seq {
                tx.open_table(WIRE_MSG_LOG)
                    .map_err(backend)?
                    .remove(s)
                    .map_err(backend)?;
            }
            let rcpt_keys: Vec<String> = all_receipts
                .iter()
                .filter(|r| r.message_id == msg.id)
                .map(|r| r.recipient_id.as_str().to_owned())
                .collect();
            let mut inbox = tx.open_table(WIRE_INBOX).map_err(backend)?;
            for rk in rcpt_keys {
                inbox
                    .remove((rk.as_str(), msg.id.as_str()))
                    .map_err(backend)?;
            }
        }

        if Self::wtx_has_table(&tx, WIRE_RCPT_LOG.name())? {
            let mut rcpt_seqs_to_remove = Vec::new();
            {
                let rcpt_log = tx.open_table(WIRE_RCPT_LOG).map_err(backend)?;
                for row in rcpt_log.iter().map_err(backend)? {
                    let (k, v) = row.map_err(backend)?;
                    let rcpt: DeliveryReceipt = serde_json::from_str(v.value()).map_err(decode)?;
                    if pruned_ids.contains(&rcpt.message_id) {
                        rcpt_seqs_to_remove.push(k.value());
                    }
                }
            }
            let mut rcpt_log = tx.open_table(WIRE_RCPT_LOG).map_err(backend)?;
            for seq in rcpt_seqs_to_remove {
                rcpt_log.remove(seq).map_err(backend)?;
            }
        }

        tx.commit().map_err(backend)?;
        Ok(count)
    }
}
