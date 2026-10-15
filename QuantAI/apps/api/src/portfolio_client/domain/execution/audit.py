import hashlib
import json
from dataclasses import dataclass
from typing import Any


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True)
class AuditEntry:
    chain_id: str
    seq: int
    previous_hash: str
    event_hash: str
    payload_json: str


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    checked_events: int
    head_hash: str
    reason: str | None = None


def audit_hash(previous_hash: str, payload_json: str) -> str:
    return hashlib.sha256((previous_hash + payload_json).encode("utf-8")).hexdigest()


def verify_audit(entries: list[AuditEntry], head_seq: int, head_hash: str) -> AuditVerification:
    previous = "0" * 64
    for seq, entry in enumerate(entries, 1):
        try:
            payload = json.loads(entry.payload_json)
            valid = (
                entry.seq == seq
                and entry.previous_hash == previous
                and payload["schema_version"] == 1
                and payload["seq"] == seq
                and payload["chain_id"] == entry.chain_id
                and canonical_json(payload) == entry.payload_json
                and audit_hash(previous, entry.payload_json) == entry.event_hash
            )
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            return AuditVerification(False, seq - 1, head_hash, "hash_or_sequence_mismatch")
        previous = entry.event_hash
    if len(entries) != head_seq or previous != head_hash:
        return AuditVerification(False, len(entries), head_hash, "head_mismatch")
    return AuditVerification(True, len(entries), head_hash)
