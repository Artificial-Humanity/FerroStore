//! FerroWire message store.

use super::FerroStore;
use super::doc::{
    self, AGENT_BY_NAME, AGENTS, IDS, NEXT_WIRE_MSG_SEQ, NEXT_WIRE_RCPT_SEQ, WIRE_INBOX,
    WIRE_MESSAGES, WIRE_MSG_LOG, WIRE_MSG_SEQ, WIRE_RCPT_LOG, WIRE_REPLIES, WIRE_THREADS,
};
use crate::{FORMAT_WITH_WIRE, decode, mint};
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

fn index_message_id(txn: &mut super::Txn, id: &Iri) -> Result<(), StoreError> {
    if let Some(wire) = doc::get_json::<String>(txn, IDS, id.as_str())? {
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
    doc::put_json(txn, IDS, id.as_str(), &Kind::Message.as_wire())?;
    Ok(())
}

fn append_wire_message(txn: &mut super::Txn, msg: &WireMessage) -> Result<(), StoreError> {
    if doc::get_json::<u64>(txn, WIRE_MSG_SEQ, msg.id.as_str())?.is_none() {
        let seq = doc::bump_meta(txn, NEXT_WIRE_MSG_SEQ)?;
        doc::put_json(txn, WIRE_MSG_LOG, &doc::key_u64(seq), &msg.id.as_str())?;
        doc::put_json(txn, WIRE_MSG_SEQ, msg.id.as_str(), &seq)?;
    }
    doc::put_unit(
        txn,
        WIRE_THREADS,
        &doc::pair(msg.thread_id.as_str(), msg.id.as_str()),
    );
    if let Some(ref corr) = msg.correlation_id {
        doc::put_unit(
            txn,
            WIRE_REPLIES,
            &doc::pair(corr.as_str(), msg.id.as_str()),
        );
    }
    doc::put_json(txn, WIRE_MESSAGES, msg.id.as_str(), msg)?;
    Ok(())
}

fn append_wire_receipt(txn: &mut super::Txn, rcpt: &DeliveryReceipt) -> Result<(), StoreError> {
    let seq = doc::bump_meta(txn, NEXT_WIRE_RCPT_SEQ)?;
    doc::put_json(txn, WIRE_RCPT_LOG, &doc::key_u64(seq), rcpt)?;
    Ok(())
}

fn recipient_candidates(
    txn: &super::Txn,
    label: &str,
    query: &str,
) -> Result<(Option<Iri>, Option<String>), StoreError> {
    let trimmed = query.trim();
    let lookup_by_name = |txn: &super::Txn, key: &str| -> Result<Option<Iri>, StoreError> {
        let Some(raw) = doc::get_json::<String>(txn, AGENT_BY_NAME, key)? else {
            return Ok(None);
        };
        let iri = Iri::parse(&raw).map_err(decode)?;
        doc::resolve_iri(txn, label, &iri).map(Some)
    };
    if let Ok(iri) = Iri::parse(trimmed) {
        let resolved = doc::resolve_iri(txn, label, &iri)?;
        if let Some(rec) = doc::get_json::<AgentRecord>(txn, AGENTS, resolved.as_str())? {
            return Ok((Some(resolved), Some(rec.name_key)));
        }
        if let Some(rest) = trimmed.strip_prefix("urn:fl:agent:")
            && let Ok(key) = validate_slug(rest)
        {
            return Ok((lookup_by_name(txn, &key)?, Some(key)));
        }
        return Ok((Some(resolved), None));
    }
    if let Ok(key) = validate_slug(trimmed) {
        return Ok((lookup_by_name(txn, &key)?, Some(key)));
    }
    Ok((None, None))
}

fn receipt_matches(
    rcpt: &DeliveryReceipt,
    rcpt_resolved_id: &Iri,
    agent_id: Option<&Iri>,
    name_key: Option<&str>,
    raw_query: &str,
) -> bool {
    let trimmed = raw_query.trim();
    if let Some(agent_id) = agent_id
        && rcpt_resolved_id == agent_id
    {
        return true;
    }
    if let Some(name_key) = name_key
        && (rcpt.recipient_name == name_key
            || rcpt.recipient_id.as_str() == format!("urn:fl:agent:{name_key}"))
    {
        return true;
    }
    rcpt.recipient_id.as_str() == trimmed || rcpt.recipient_name.eq_ignore_ascii_case(trimmed)
}

fn inbox_rows(txn: &super::Txn) -> Result<Vec<(String, String, DeliveryReceipt)>, StoreError> {
    let mut out = Vec::new();
    for (id, doc) in txn.scan(WIRE_INBOX) {
        let Some((recipient, message)) = doc::split_pair(&id) else {
            return Err(decode(format!("wire_inbox key `{id}` is not a pair")));
        };
        out.push((
            recipient.to_string(),
            message.to_string(),
            serde_json::from_value(doc).map_err(decode)?,
        ));
    }
    Ok(out)
}

impl MessageStore for FerroStore {
    fn send_wire(
        &self,
        args: SendWireArgs,
    ) -> Result<(WireMessage, Vec<DeliveryReceipt>), StoreError> {
        let subject = validate_subject(&args.subject)
            .map_err(|err| StoreError::InvalidState(err.to_string()))?;
        let body = validate_wire_body(&args.body)
            .map_err(|err| StoreError::InvalidState(err.to_string()))?;
        let sender_name = validate_sender_name(&args.sender_name)
            .map_err(|err| StoreError::InvalidState(err.to_string()))?;
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_WIRE)?;
            let (parent_thread, correlation_id) = match &args.reply_to {
                Some(parent_id) => {
                    let Some(parent) =
                        doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, parent_id.as_str())?
                    else {
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
            let msg_id = match args.id.clone() {
                Some(id) => {
                    index_message_id(txn, id.iri())?;
                    id
                }
                None => {
                    let iri = mint()?;
                    index_message_id(txn, &iri)?;
                    MessageId(iri)
                }
            };
            let thread_id = args
                .thread_id
                .clone()
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
            let all_agents = doc::scan_json::<AgentRecord>(txn, AGENTS)?;
            let sender_id = args.sender_id.clone().or_else(|| {
                all_agents
                    .iter()
                    .find(|agent| {
                        agent.status == AgentStatus::Active && agent.name_key == sender_name
                    })
                    .map(|agent| agent.id.clone())
            });
            let msg = WireMessage {
                id: msg_id.clone(),
                thread_id,
                reply_to: args.reply_to.clone(),
                correlation_id,
                sender_id,
                sender_name: sender_name.clone(),
                sender_host: args.sender_host.clone(),
                to: args.to.clone(),
                kind: args.kind,
                urgency: args.urgency,
                durable,
                subject: subject.clone(),
                body: body.clone(),
                created_at: args.at.clone(),
                deadline_at: args.deadline_at.clone(),
            };
            msg.validate()
                .map_err(|err| StoreError::InvalidState(err.to_string()))?;
            let recipients = resolve_recipients(
                &msg.to,
                msg.sender_id.as_ref(),
                &msg.sender_name,
                &all_agents,
            );
            append_wire_message(txn, &msg)?;
            let mut receipts = Vec::with_capacity(recipients.len());
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
                doc::put_json(
                    txn,
                    WIRE_INBOX,
                    &doc::pair(recipient_id.as_str(), msg.id.as_str()),
                    &rcpt,
                )?;
                append_wire_receipt(txn, &rcpt)?;
                receipts.push(rcpt);
            }
            Ok((msg, receipts))
        })
    }

    fn get_wire_message(&self, id: &MessageId) -> Result<Option<WireMessage>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| doc::get_json(txn, WIRE_MESSAGES, id.as_str()))
    }

    fn list_wire_inbox(&self, filter: &WireFilter) -> Result<Vec<WireInboxItem>, StoreError> {
        let filter = filter.clone();
        self.with_read(|txn| {
            let candidates = match &filter.recipient {
                Some(query) => Some(recipient_candidates(txn, &self.label, query)?),
                None => None,
            };
            let mut items = Vec::new();
            for (_recipient, _message, rcpt) in inbox_rows(txn)? {
                if let (Some(query), Some((agent_id, name_key))) = (&filter.recipient, &candidates)
                {
                    let rcpt_resolved = doc::resolve_iri(txn, &self.label, &rcpt.recipient_id)?;
                    if !receipt_matches(
                        &rcpt,
                        &rcpt_resolved,
                        agent_id.as_ref(),
                        name_key.as_deref(),
                        query,
                    ) {
                        continue;
                    }
                }
                if filter.unread_only
                    && matches!(rcpt.status, DeliveryStatus::Acked | DeliveryStatus::Expired)
                {
                    continue;
                }
                let Some(msg) =
                    doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, rcpt.message_id.as_str())?
                else {
                    continue;
                };
                if let Some(ref thread) = filter.thread_id
                    && &msg.thread_id != thread
                {
                    continue;
                }
                if let Some(ref correlation) = filter.correlation_id
                    && msg.correlation_id.as_ref() != Some(correlation)
                {
                    continue;
                }
                items.push(WireInboxItem {
                    message: msg,
                    receipt: rcpt,
                });
            }
            items.sort_by(|a, b| {
                (&a.message.id, &a.receipt.recipient_id)
                    .cmp(&(&b.message.id, &b.receipt.recipient_id))
            });
            if let Some(limit) = filter.limit
                && items.len() > limit
            {
                items.truncate(limit);
            }
            Ok(items)
        })
    }

    fn list_wire_thread(&self, thread_id: &ThreadId) -> Result<Vec<WireMessage>, StoreError> {
        let thread_id = thread_id.clone();
        self.with_read(|txn| messages_for_pair(txn, WIRE_THREADS, thread_id.as_str()))
    }

    fn list_wire_replies(
        &self,
        correlation_id: &MessageId,
    ) -> Result<Vec<WireMessage>, StoreError> {
        let correlation_id = correlation_id.clone();
        self.with_read(|txn| messages_for_pair(txn, WIRE_REPLIES, correlation_id.as_str()))
    }

    fn list_wire_receipts(
        &self,
        message_id: &MessageId,
    ) -> Result<Vec<DeliveryReceipt>, StoreError> {
        let message_id = message_id.clone();
        self.with_read(|txn| {
            if doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, message_id.as_str())?.is_none() {
                return Err(StoreError::MessageNotFound(message_id));
            }
            let mut out = Vec::new();
            for (_recipient, message, rcpt) in inbox_rows(txn)? {
                if message == message_id.as_str() {
                    out.push(rcpt);
                }
            }
            out.sort_by(|a, b| a.recipient_id.cmp(&b.recipient_id));
            Ok(out)
        })
    }

    fn update_wire_receipt(
        &self,
        message_id: &MessageId,
        recipient: &str,
        status: DeliveryStatus,
        increment_attempts: bool,
        at: At,
    ) -> Result<Option<DeliveryReceipt>, StoreError> {
        let message_id = message_id.clone();
        let recipient = recipient.to_string();
        self.with_write(|txn| {
            if doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, message_id.as_str())?.is_none() {
                return Err(StoreError::MessageNotFound(message_id.clone()));
            }
            let (agent_id, name_key) = recipient_candidates(txn, &self.label, &recipient)?;
            let mut matched = None;
            for (rec_key, message, rcpt) in inbox_rows(txn)? {
                if message != message_id.as_str() {
                    continue;
                }
                let rcpt_resolved = doc::resolve_iri(txn, &self.label, &rcpt.recipient_id)?;
                if receipt_matches(
                    &rcpt,
                    &rcpt_resolved,
                    agent_id.as_ref(),
                    name_key.as_deref(),
                    &recipient,
                ) {
                    matched = Some((rec_key, rcpt));
                    break;
                }
            }
            let Some((rec_key, existing)) = matched else {
                return Ok(None);
            };
            doc::raise_format(txn, FORMAT_WITH_WIRE)?;
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
                    Some(at.clone())
                } else {
                    None
                },
            };
            let merged = existing.merge(&incoming);
            doc::put_json(
                txn,
                WIRE_INBOX,
                &doc::pair(&rec_key, message_id.as_str()),
                &merged,
            )?;
            append_wire_receipt(txn, &merged)?;
            Ok(Some(merged))
        })
    }

    fn wire_messages_after(&self, after_seq: u64) -> Result<Vec<(u64, WireMessage)>, StoreError> {
        self.with_read(|txn| {
            let mut out = Vec::new();
            for (seq, id) in doc::scan_u64::<String>(txn, WIRE_MSG_LOG)? {
                if seq <= after_seq {
                    continue;
                }
                if let Some(msg) = doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, &id)? {
                    out.push((seq, msg));
                }
            }
            Ok(out)
        })
    }

    fn wire_receipts_after(
        &self,
        after_seq: u64,
    ) -> Result<Vec<(u64, DeliveryReceipt)>, StoreError> {
        self.with_read(|txn| {
            let mut out = Vec::new();
            for (seq, rcpt) in doc::scan_u64::<DeliveryReceipt>(txn, WIRE_RCPT_LOG)? {
                if seq <= after_seq {
                    continue;
                }
                if doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, rcpt.message_id.as_str())?
                    .is_some()
                {
                    out.push((seq, rcpt));
                }
            }
            Ok(out)
        })
    }

    fn import_wire_batch(
        &self,
        messages: &[WireMessage],
        receipts: &[DeliveryReceipt],
    ) -> Result<WireImportReport, StoreError> {
        if messages.is_empty() && receipts.is_empty() {
            return Ok(WireImportReport::default());
        }
        let messages = messages.to_vec();
        let receipts = receipts.to_vec();
        self.with_write(|txn| {
            let mut report = WireImportReport::default();
            doc::raise_format(txn, FORMAT_WITH_WIRE)?;
            let all_agents = doc::scan_json::<AgentRecord>(txn, AGENTS)?;
            for msg in &messages {
                // Each message stands or falls alone: a skip writes nothing for it.
                if let Err(err) = msg.validate() {
                    report.reject(
                        &msg.id,
                        false,
                        WireImportRejectReason::InvalidMessage,
                        err.to_string(),
                    );
                    continue;
                }
                if let Some(existing) =
                    doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, msg.id.as_str())?
                {
                    if &existing != msg {
                        report.reject(
                            &msg.id,
                            false,
                            WireImportRejectReason::TamperedMessage,
                            "imported wire message differs from stored copy",
                        );
                    }
                    continue;
                }
                match index_message_id(txn, msg.id.iri()) {
                    Ok(()) => {}
                    Err(err @ StoreError::WrongKind { .. }) => {
                        report.reject(
                            &msg.id,
                            false,
                            WireImportRejectReason::IdCollision,
                            err.to_string(),
                        );
                        continue;
                    }
                    Err(err) => return Err(err),
                }
                append_wire_message(txn, msg)?;
                report.imported_messages += 1;
                let resolved = resolve_recipients(
                    &msg.to,
                    msg.sender_id.as_ref(),
                    &msg.sender_name,
                    &all_agents,
                );
                for (rec_id, rec_name) in resolved {
                    let in_incoming = receipts.iter().any(|rcpt| {
                        rcpt.message_id == msg.id
                            && (rcpt.recipient_id == rec_id || rcpt.recipient_name == rec_name)
                    });
                    let key = doc::pair(rec_id.as_str(), msg.id.as_str());
                    if !in_incoming && txn.get(WIRE_INBOX, &key).is_none() {
                        let rcpt = DeliveryReceipt {
                            message_id: msg.id.clone(),
                            recipient_id: rec_id.clone(),
                            recipient_name: rec_name,
                            status: DeliveryStatus::Pending,
                            attempts: 0,
                            updated_at: msg.created_at.clone(),
                            acked_at: None,
                        };
                        doc::put_json(txn, WIRE_INBOX, &key, &rcpt)?;
                        append_wire_receipt(txn, &rcpt)?;
                    }
                }
            }
            for rcpt in &receipts {
                if doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, rcpt.message_id.as_str())?
                    .is_none()
                {
                    report.reject(
                        &rcpt.message_id,
                        true,
                        WireImportRejectReason::UnknownMessage,
                        "no such wire message",
                    );
                    continue;
                }
                let key = doc::pair(rcpt.recipient_id.as_str(), rcpt.message_id.as_str());
                match doc::get_json::<DeliveryReceipt>(txn, WIRE_INBOX, &key)? {
                    Some(existing) => {
                        let merged = existing.merge(rcpt);
                        if merged != existing {
                            doc::put_json(txn, WIRE_INBOX, &key, &merged)?;
                            append_wire_receipt(txn, &merged)?;
                            report.imported_receipts += 1;
                        }
                    }
                    None => {
                        doc::put_json(txn, WIRE_INBOX, &key, rcpt)?;
                        append_wire_receipt(txn, rcpt)?;
                        report.imported_receipts += 1;
                    }
                }
            }
            Ok(report)
        })
    }

    fn prune_wire(&self, before: At) -> Result<usize, StoreError> {
        self.with_write(|txn| {
            let all_messages: Vec<WireMessage> = doc::scan_json(txn, WIRE_MESSAGES)?;
            let all_receipts = inbox_rows(txn)?;
            let to_prune: Vec<WireMessage> = all_messages
                .into_iter()
                .filter(|msg| !msg.durable && msg.created_at < before)
                .filter(|msg| {
                    all_receipts
                        .iter()
                        .filter(|(_, _, rcpt)| rcpt.message_id == msg.id)
                        .all(|(_, _, rcpt)| {
                            matches!(rcpt.status, DeliveryStatus::Acked | DeliveryStatus::Expired)
                        })
                })
                .collect();
            let count = to_prune.len();
            if count == 0 {
                return Ok(0);
            }
            doc::raise_format(txn, FORMAT_WITH_WIRE)?;
            let pruned_ids: std::collections::BTreeSet<MessageId> =
                to_prune.iter().map(|msg| msg.id.clone()).collect();
            for msg in &to_prune {
                txn.delete(WIRE_MESSAGES, msg.id.as_str());
                txn.delete(IDS, msg.id.as_str());
                txn.delete(
                    WIRE_THREADS,
                    doc::pair(msg.thread_id.as_str(), msg.id.as_str()),
                );
                if let Some(ref corr) = msg.correlation_id {
                    txn.delete(WIRE_REPLIES, doc::pair(corr.as_str(), msg.id.as_str()));
                }
                if let Some(seq) = doc::take_json::<u64>(txn, WIRE_MSG_SEQ, msg.id.as_str())? {
                    txn.delete(WIRE_MSG_LOG, doc::key_u64(seq));
                }
                let recipient_keys: Vec<String> = all_receipts
                    .iter()
                    .filter(|(_, _, rcpt)| rcpt.message_id == msg.id)
                    .map(|(recipient, _, _)| recipient.clone())
                    .collect();
                for recipient in recipient_keys {
                    txn.delete(WIRE_INBOX, doc::pair(&recipient, msg.id.as_str()));
                }
            }
            let stale: Vec<u64> = doc::scan_u64::<DeliveryReceipt>(txn, WIRE_RCPT_LOG)?
                .into_iter()
                .filter_map(|(seq, rcpt)| pruned_ids.contains(&rcpt.message_id).then_some(seq))
                .collect();
            for seq in stale {
                txn.delete(WIRE_RCPT_LOG, doc::key_u64(seq));
            }
            Ok(count)
        })
    }
}

fn messages_for_pair(
    txn: &super::Txn,
    collection: &str,
    first: &str,
) -> Result<Vec<WireMessage>, StoreError> {
    let mut out = Vec::new();
    for (id, _) in txn.scan(collection) {
        let Some((left, message_id)) = doc::split_pair(&id) else {
            continue;
        };
        if left != first {
            continue;
        }
        if let Some(msg) = doc::get_json::<WireMessage>(txn, WIRE_MESSAGES, message_id)? {
            out.push(msg);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}
