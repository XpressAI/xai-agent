"""Fixture-backed OmniBrain MCP gateway authorization/retrieval stub.

This module implements the first deterministic server-side slice from
ADR-001: opaque-token introspection, per-tool authorization, permission-safe
retrieval filters, safe denial responses, and redacted audit logging.

It is intentionally dependency-free so the contract can be used by tests,
Xircuits components, or a future HTTP/stdio MCP transport adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_SAFE_SEARCH_MESSAGE = "No accessible OmniBrain results are available for this request."
DEFAULT_SAFE_DOCUMENT_MESSAGE = "That document is not accessible with your current OmniBrain grant."
DEFAULT_SAFE_EXPLANATION = (
    "Your current OmniBrain grant does not include this content. "
    "Ask an admin if you believe this is incorrect."
)


class OmniBrainAuthError(Exception):
    """Base class for transport-level MCP gateway authorization failures."""

    status_code = 403
    reason_code = "FORBIDDEN"
    safe_message = "The OmniBrain request is not authorized."

    def __init__(self, reason_code: Optional[str] = None, safe_message: Optional[str] = None):
        self.reason_code = reason_code or self.reason_code
        self.safe_message = safe_message or self.safe_message
        super().__init__(self.safe_message)


class OmniBrainAuthenticationError(OmniBrainAuthError):
    """Missing, invalid, expired, or revoked bearer token."""

    status_code = 401
    reason_code = "TOKEN_INACTIVE"
    safe_message = "The OmniBrain token is missing or invalid."


class OmniBrainForbiddenError(OmniBrainAuthError):
    """Valid token/session but requested tool or operation is not allowed."""

    status_code = 403
    reason_code = "TOOL_NOT_ALLOWED"
    safe_message = "The requested OmniBrain tool is not allowed for this grant."


class OmniBrainValidationError(OmniBrainAuthError):
    """Malformed request that should not proceed to retrieval."""

    status_code = 400
    reason_code = "INVALID_REQUEST"
    safe_message = "The OmniBrain request is invalid."


@dataclass(frozen=True)
class TokenIntrospection:
    active: bool
    token_id: Optional[str] = None
    tenant_id: Optional[str] = None
    subject_principal_id: Optional[str] = None
    subject_display: Optional[str] = None
    grant_ids: Tuple[str, ...] = ()
    scope_ids: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()
    tool_scopes: Tuple[str, ...] = ()
    expires_at: Optional[datetime] = None
    rate_limit_policy: Optional[str] = None
    audit_policy: str = "log-redacted"
    reason_code: Optional[str] = None


@dataclass
class QueryLogEntry:
    audit_id: str
    timestamp: str
    tenant_id: Optional[str]
    token_id: Optional[str]
    grant_ids: List[str]
    subject_principal_id: Optional[str]
    subject_display: Optional[str]
    tool_name: str
    request_summary: Dict[str, Any]
    decision: str
    result_ids: List[Dict[str, str]] = field(default_factory=list)
    denied_counts: Dict[str, int] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_query_terms(query: str) -> Set[str]:
    terms = set()
    for raw in query.lower().replace("/", " ").replace("-", " ").split():
        term = "".join(ch for ch in raw if ch.isalnum())
        if len(term) >= 3:
            terms.add(term)
    return terms


def _redacted_request_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query", "")) if "query" in payload else ""
    summary = {
        "keys": sorted(payload.keys()),
        "payload_bytes": len(json.dumps(payload, sort_keys=True, default=str)),
    }
    if query:
        summary["query_hash"] = _hash_text(query)
        summary["query_length"] = len(query)
    filters = payload.get("filters")
    if isinstance(filters, dict):
        summary["filters"] = {key: filters.get(key) for key in sorted(filters.keys())}
    if "document_id" in payload:
        summary["document_id_hash"] = _hash_text(str(payload["document_id"]))
    return summary


def _safe_scope_names(scopes: Dict[str, Dict[str, Any]], scope_ids: Iterable[str]) -> List[str]:
    names = []
    for scope_id in scope_ids:
        scope = scopes.get(scope_id)
        if scope and scope.get("status") == "active":
            names.append(scope["name"])
    return names


def _has_acl_read(document: Dict[str, Any], token: TokenIntrospection, principals: Dict[str, Dict[str, Any]]) -> bool:
    principal = principals.get(token.subject_principal_id or "", {})
    allowed_principals = set(document.get("acl", {}).get("read_principals", []))
    allowed_groups = set(document.get("acl", {}).get("read_groups", []))
    subject_groups = set(principal.get("groups", []))
    return (token.subject_principal_id in allowed_principals) or bool(subject_groups & allowed_groups)


def build_demo_fixture_store() -> Dict[str, Any]:
    """Return deterministic demo state for OmniBrain permission-safety tests."""

    now = _utc_now()
    future = (now + timedelta(days=30)).isoformat()
    past = (now - timedelta(days=1)).isoformat()
    stale_acl = (now - timedelta(days=45)).isoformat()
    fresh_acl = (now - timedelta(hours=2)).isoformat()

    return {
        "principals": {
            "user_mia": {
                "principal_id": "user_mia",
                "display": "mia@acme.example",
                "tenant_id": "tenant_acme",
                "groups": ["group_support"],
            },
            "user_eric": {
                "principal_id": "user_eric",
                "display": "eric@acme.example",
                "tenant_id": "tenant_acme",
                "groups": ["group_engineering"],
            },
        },
        "scopes": {
            "scope_support_safe": {
                "scope_id": "scope_support_safe",
                "tenant_id": "tenant_acme",
                "name": "support-safe",
                "description": "Support playbooks and customer-facing policy",
                "status": "active",
                "document_ids": ["doc_refund_policy", "doc_support_macro"],
                "max_results": 5,
            },
            "scope_engineering_safe": {
                "scope_id": "scope_engineering_safe",
                "tenant_id": "tenant_acme",
                "name": "engineering-safe",
                "description": "Engineering runbooks and public architecture notes",
                "status": "active",
                "document_ids": ["doc_release_runbook"],
                "max_results": 5,
            },
            "scope_exec_confidential": {
                "scope_id": "scope_exec_confidential",
                "tenant_id": "tenant_acme",
                "name": "exec-confidential",
                "description": "Executive confidential planning material",
                "status": "active",
                "document_ids": ["doc_exec_acquisition_plan"],
                "max_results": 3,
            },
        },
        "sources": {
            "src_confluence_support": {
                "source_id": "src_confluence_support",
                "tenant_id": "tenant_acme",
                "kind": "confluence",
                "status": "active",
                "acl_last_synced_at": fresh_acl,
                "acl_ttl_days": 7,
            },
            "src_gdrive_exec": {
                "source_id": "src_gdrive_exec",
                "tenant_id": "tenant_acme",
                "kind": "google_drive",
                "status": "active",
                "acl_last_synced_at": fresh_acl,
                "acl_ttl_days": 7,
            },
            "src_sharepoint_eng": {
                "source_id": "src_sharepoint_eng",
                "tenant_id": "tenant_acme",
                "kind": "sharepoint",
                "status": "active",
                "acl_last_synced_at": fresh_acl,
                "acl_ttl_days": 7,
            },
            "src_confluence_stale": {
                "source_id": "src_confluence_stale",
                "tenant_id": "tenant_acme",
                "kind": "confluence",
                "status": "active",
                "acl_last_synced_at": stale_acl,
                "acl_ttl_days": 7,
            },
            "src_notion_other_tenant": {
                "source_id": "src_notion_other_tenant",
                "tenant_id": "tenant_umbrella",
                "kind": "notion",
                "status": "active",
                "acl_last_synced_at": fresh_acl,
                "acl_ttl_days": 7,
            },
        },
        "grants": {
            "grant_mia_support": {
                "grant_id": "grant_mia_support",
                "tenant_id": "tenant_acme",
                "principal_id": "user_mia",
                "scope_ids": ["scope_support_safe"],
                "expires_at": future,
                "revoked_at": None,
            },
            "grant_eric_engineering": {
                "grant_id": "grant_eric_engineering",
                "tenant_id": "tenant_acme",
                "principal_id": "user_eric",
                "scope_ids": ["scope_engineering_safe"],
                "expires_at": future,
                "revoked_at": None,
            },
            "grant_mia_revoked": {
                "grant_id": "grant_mia_revoked",
                "tenant_id": "tenant_acme",
                "principal_id": "user_mia",
                "scope_ids": ["scope_support_safe"],
                "expires_at": future,
                "revoked_at": past,
            },
        },
        "tokens": {
            "demo-mia-support-token": {
                "token_id": "mcp_tok_mia_support",
                "tenant_id": "tenant_acme",
                "subject_principal_id": "user_mia",
                "grant_ids": ["grant_mia_support"],
                "scope_ids": ["scope_support_safe"],
                "allowed_tools": [
                    "omnibrain.search",
                    "omnibrain.get_document",
                    "omnibrain.list_scopes",
                    "omnibrain.explain_access",
                ],
                "tool_scopes": ["search:read", "document:read", "scope:read", "access:explain"],
                "expires_at": future,
                "revoked_at": None,
                "rate_limit_policy": "employee-demo-default",
                "audit_policy": "log-redacted",
            },
            "demo-mia-search-only-token": {
                "token_id": "mcp_tok_mia_search_only",
                "tenant_id": "tenant_acme",
                "subject_principal_id": "user_mia",
                "grant_ids": ["grant_mia_support"],
                "scope_ids": ["scope_support_safe"],
                "allowed_tools": ["omnibrain.search"],
                "tool_scopes": ["search:read"],
                "expires_at": future,
                "revoked_at": None,
                "rate_limit_policy": "employee-demo-default",
                "audit_policy": "log-redacted",
            },
            "demo-eric-engineering-token": {
                "token_id": "mcp_tok_eric_engineering",
                "tenant_id": "tenant_acme",
                "subject_principal_id": "user_eric",
                "grant_ids": ["grant_eric_engineering"],
                "scope_ids": ["scope_engineering_safe"],
                "allowed_tools": [
                    "omnibrain.search",
                    "omnibrain.get_document",
                    "omnibrain.list_scopes",
                    "omnibrain.explain_access",
                ],
                "tool_scopes": ["search:read", "document:read", "scope:read", "access:explain"],
                "expires_at": future,
                "revoked_at": None,
                "rate_limit_policy": "employee-demo-default",
                "audit_policy": "log-redacted",
            },
            "demo-revoked-token": {
                "token_id": "mcp_tok_revoked",
                "tenant_id": "tenant_acme",
                "subject_principal_id": "user_mia",
                "grant_ids": ["grant_mia_support"],
                "scope_ids": ["scope_support_safe"],
                "allowed_tools": ["omnibrain.search"],
                "tool_scopes": ["search:read"],
                "expires_at": future,
                "revoked_at": past,
                "rate_limit_policy": "employee-demo-default",
                "audit_policy": "log-redacted",
            },
            "demo-grant-revoked-token": {
                "token_id": "mcp_tok_grant_revoked",
                "tenant_id": "tenant_acme",
                "subject_principal_id": "user_mia",
                "grant_ids": ["grant_mia_revoked"],
                "scope_ids": ["scope_support_safe"],
                "allowed_tools": ["omnibrain.search"],
                "tool_scopes": ["search:read"],
                "expires_at": future,
                "revoked_at": None,
                "rate_limit_policy": "employee-demo-default",
                "audit_policy": "log-redacted",
            },
        },
        "documents": {
            "doc_refund_policy": {
                "document_id": "doc_refund_policy",
                "tenant_id": "tenant_acme",
                "source_id": "src_confluence_support",
                "source_kind": "confluence",
                "status": "active",
                "title": "Refund Exceptions",
                "citation": "confluence://Support/Refund Exceptions#v12",
                "modified_at": "2026-06-28T18:00:00Z",
                "acl": {"read_principals": [], "read_groups": ["group_support"]},
                "text": "Enterprise annual plan cancellations require account owner and finance approval. Agents must record the approval ticket before issuing exceptions.",
                "chunks": [
                    {
                        "chunk_id": "chk_refund_policy_003",
                        "status": "active",
                        "text": "Enterprise annual plan cancellations require account owner and finance approval before refund exceptions are issued.",
                    }
                ],
            },
            "doc_support_macro": {
                "document_id": "doc_support_macro",
                "tenant_id": "tenant_acme",
                "source_id": "src_confluence_support",
                "source_kind": "confluence",
                "status": "active",
                "title": "Support Escalation Macro",
                "citation": "confluence://Support/Escalation Macro#v4",
                "modified_at": "2026-06-27T12:00:00Z",
                "acl": {"read_principals": [], "read_groups": ["group_support"]},
                "text": "Use the customer-impact escalation macro when a billing policy exception needs finance review.",
                "chunks": [
                    {
                        "chunk_id": "chk_support_macro_001",
                        "status": "active",
                        "text": "Use the customer-impact escalation macro for billing policy exceptions that need finance review.",
                    }
                ],
            },
            "doc_exec_acquisition_plan": {
                "document_id": "doc_exec_acquisition_plan",
                "tenant_id": "tenant_acme",
                "source_id": "src_gdrive_exec",
                "source_kind": "google_drive",
                "status": "active",
                "title": "Project Falcon Acquisition Plan",
                "citation": "gdrive://Exec/Project Falcon Acquisition Plan",
                "modified_at": "2026-06-29T09:30:00Z",
                "acl": {"read_principals": ["user_ceo"], "read_groups": ["group_exec"]},
                "text": "Project Falcon acquisition target list, deal timing, board memo, and confidential valuation model.",
                "chunks": [
                    {
                        "chunk_id": "chk_exec_plan_001",
                        "status": "active",
                        "text": "Project Falcon acquisition target list and confidential valuation model.",
                    }
                ],
            },
            "doc_release_runbook": {
                "document_id": "doc_release_runbook",
                "tenant_id": "tenant_acme",
                "source_id": "src_sharepoint_eng",
                "source_kind": "sharepoint",
                "status": "active",
                "title": "Release Runbook",
                "citation": "sharepoint://Engineering/Release Runbook#v8",
                "modified_at": "2026-06-25T11:00:00Z",
                "acl": {"read_principals": [], "read_groups": ["group_engineering"]},
                "text": "Production releases require two approvals, green integration tests, and rollback owner assignment.",
                "chunks": [
                    {
                        "chunk_id": "chk_release_runbook_002",
                        "status": "active",
                        "text": "Production releases require two approvals, green integration tests, and rollback owner assignment.",
                    }
                ],
            },
            "doc_stale_acl_policy": {
                "document_id": "doc_stale_acl_policy",
                "tenant_id": "tenant_acme",
                "source_id": "src_confluence_stale",
                "source_kind": "confluence",
                "status": "active",
                "title": "Legacy Support Refund Policy",
                "citation": "confluence://Support/Legacy Refund Policy#v1",
                "modified_at": "2026-05-01T11:00:00Z",
                "acl": {"read_principals": [], "read_groups": ["group_support"]},
                "text": "Old policy text that must not be returned because ACLs are stale.",
                "chunks": [
                    {"chunk_id": "chk_stale_acl_001", "status": "active", "text": "Old stale refund policy."}
                ],
            },
            "doc_other_tenant_refund_policy": {
                "document_id": "doc_other_tenant_refund_policy",
                "tenant_id": "tenant_umbrella",
                "source_id": "src_notion_other_tenant",
                "source_kind": "notion",
                "status": "active",
                "title": "Refund Exceptions",
                "citation": "notion://Umbrella/Refund Exceptions",
                "modified_at": "2026-06-28T18:00:00Z",
                "acl": {"read_principals": ["user_mia"], "read_groups": ["group_support"]},
                "text": "Other tenant refund policy must never be visible to Acme tokens.",
                "chunks": [
                    {"chunk_id": "chk_other_tenant_001", "status": "active", "text": "Other tenant refund policy."}
                ],
            },
        },
    }


class OmniBrainMcpGateway:
    """Permission-enforcing MCP gateway facade backed by deterministic fixtures."""

    TOOL_SCOPES = {
        "omnibrain.search": "search:read",
        "omnibrain.get_document": "document:read",
        "omnibrain.list_scopes": "scope:read",
        "omnibrain.explain_access": "access:explain",
    }

    def __init__(self, store: Optional[Dict[str, Any]] = None, *, now=None):
        self.store = copy.deepcopy(store or build_demo_fixture_store())
        self._now = now or _utc_now
        self.query_logs: List[QueryLogEntry] = []

    def introspect_mcp_token(self, bearer_token: Optional[str]) -> TokenIntrospection:
        if not bearer_token:
            return TokenIntrospection(active=False, reason_code="TOKEN_MISSING")

        token_record = self.store["tokens"].get(bearer_token)
        if not token_record:
            return TokenIntrospection(active=False, reason_code="TOKEN_INVALID")

        now = self._now()
        revoked_at = _parse_dt(token_record.get("revoked_at"))
        expires_at = _parse_dt(token_record.get("expires_at"))
        if revoked_at and revoked_at <= now:
            return TokenIntrospection(active=False, token_id=token_record.get("token_id"), reason_code="TOKEN_REVOKED")
        if expires_at and expires_at <= now:
            return TokenIntrospection(active=False, token_id=token_record.get("token_id"), reason_code="TOKEN_EXPIRED")

        grant_ids = tuple(token_record.get("grant_ids", []))
        for grant_id in grant_ids:
            grant = self.store["grants"].get(grant_id)
            if not grant:
                return TokenIntrospection(active=False, token_id=token_record.get("token_id"), reason_code="GRANT_MISSING")
            grant_revoked_at = _parse_dt(grant.get("revoked_at"))
            grant_expires_at = _parse_dt(grant.get("expires_at"))
            if grant_revoked_at and grant_revoked_at <= now:
                return TokenIntrospection(active=False, token_id=token_record.get("token_id"), reason_code="GRANT_REVOKED")
            if grant_expires_at and grant_expires_at <= now:
                return TokenIntrospection(active=False, token_id=token_record.get("token_id"), reason_code="GRANT_EXPIRED")

        principal = self.store["principals"].get(token_record.get("subject_principal_id"), {})
        return TokenIntrospection(
            active=True,
            token_id=token_record.get("token_id"),
            tenant_id=token_record.get("tenant_id"),
            subject_principal_id=token_record.get("subject_principal_id"),
            subject_display=principal.get("display"),
            grant_ids=grant_ids,
            scope_ids=tuple(token_record.get("scope_ids", [])),
            allowed_tools=tuple(token_record.get("allowed_tools", [])),
            tool_scopes=tuple(token_record.get("tool_scopes", [])),
            expires_at=expires_at,
            rate_limit_policy=token_record.get("rate_limit_policy"),
            audit_policy=token_record.get("audit_policy", "log-redacted"),
        )

    def require_tool(self, bearer_token: Optional[str], tool_name: str) -> TokenIntrospection:
        token = self.introspect_mcp_token(bearer_token)
        if not token.active:
            raise OmniBrainAuthenticationError(token.reason_code)
        required_scope = self.TOOL_SCOPES.get(tool_name)
        if not required_scope:
            raise OmniBrainForbiddenError("UNKNOWN_TOOL")
        if tool_name not in token.allowed_tools:
            raise OmniBrainForbiddenError("TOOL_NOT_ALLOWED")
        if required_scope not in token.tool_scopes:
            raise OmniBrainForbiddenError("TOOL_SCOPE_NOT_ALLOWED")
        return token

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None, *, bearer_token: Optional[str]) -> Dict[str, Any]:
        arguments = arguments or {}
        if tool_name == "omnibrain.search":
            return self.search(bearer_token, **arguments)
        if tool_name == "omnibrain.get_document":
            return self.get_document(bearer_token, **arguments)
        if tool_name == "omnibrain.list_scopes":
            return self.list_scopes(bearer_token)
        if tool_name == "omnibrain.explain_access":
            return self.explain_access(bearer_token, **arguments)
        raise OmniBrainForbiddenError("UNKNOWN_TOOL")

    def search(
        self,
        bearer_token: Optional[str],
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        token = self.require_tool(bearer_token, "omnibrain.search")
        filters = self._validate_filters(filters or {})
        requested_scope = filters.get("scope_id")
        if requested_scope and requested_scope not in token.scope_ids:
            return self._safe_denial(
                token,
                "omnibrain.search",
                {"query": query, "limit": limit, "filters": filters},
                ["REQUESTED_SCOPE_NOT_BOUND"],
                safe_message=DEFAULT_SAFE_SEARCH_MESSAGE,
            )

        usable_scope_ids = [requested_scope] if requested_scope else list(token.scope_ids)
        max_results = self._max_results_for_scopes(usable_scope_ids)
        effective_limit = max(0, min(int(limit or 0), max_results))
        terms = _normalize_query_terms(query)

        results = []
        denied_counts: Dict[str, int] = {}
        for document in self.store["documents"].values():
            doc_chunk_scores = [
                self._score(query, terms, document, chunk)
                for chunk in document.get("chunks", [])
                if chunk.get("status") == "active"
            ]
            candidate_matches_query = bool(doc_chunk_scores) and max(doc_chunk_scores) > 0
            allowed, reasons = self._document_allowed(document, token, usable_scope_ids, filters)
            if not allowed:
                if candidate_matches_query:
                    self._increment_reasons(denied_counts, reasons)
                continue
            for chunk in document.get("chunks", []):
                if chunk.get("status") != "active":
                    self._increment_reasons(denied_counts, ["CHUNK_INACTIVE"])
                    continue
                score = self._score(query, terms, document, chunk)
                if score <= 0:
                    continue
                results.append((score, document, chunk))

        results.sort(key=lambda item: (-item[0], item[1]["document_id"], item[2]["chunk_id"]))
        response_results = [self._result_payload(document, chunk) for _, document, chunk in results[:effective_limit]]
        decision = "allowed" if response_results else "denied"
        audit = self._log(
            token,
            "omnibrain.search",
            {"query": query, "limit": limit, "filters": filters},
            decision,
            result_ids=[{"document_id": r["document_id"], "chunk_id": r["chunk_id"]} for r in response_results],
            denied_counts=denied_counts,
            reason_codes=[] if response_results else ["NO_ACCESSIBLE_RESULTS"],
        )
        if not response_results:
            return {"decision": "denied", "results": [], "safe_message": DEFAULT_SAFE_SEARCH_MESSAGE, "audit_id": audit.audit_id}
        return {
            "decision": "allowed",
            "scope_names": _safe_scope_names(self.store["scopes"], usable_scope_ids),
            "results": response_results,
            "audit_id": audit.audit_id,
        }

    def get_document(self, bearer_token: Optional[str], document_id: str) -> Dict[str, Any]:
        token = self.require_tool(bearer_token, "omnibrain.get_document")
        document = self.store["documents"].get(document_id)
        allowed = False
        reasons = ["DOCUMENT_NOT_ACCESSIBLE"]
        if document:
            allowed, reasons = self._document_allowed(document, token, token.scope_ids, {})

        if not allowed:
            return self._safe_denial(
                token,
                "omnibrain.get_document",
                {"document_id": document_id},
                reasons,
                safe_message=DEFAULT_SAFE_DOCUMENT_MESSAGE,
            )

        audit = self._log(
            token,
            "omnibrain.get_document",
            {"document_id": document_id},
            "allowed",
            result_ids=[{"document_id": document["document_id"]}],
        )
        return {
            "decision": "allowed",
            "document": {
                "document_id": document["document_id"],
                "title": document["title"],
                "text": document["text"],
                "citation": document["citation"],
                "source_kind": document["source_kind"],
                "modified_at": document["modified_at"],
            },
            "audit_id": audit.audit_id,
        }

    def list_scopes(self, bearer_token: Optional[str]) -> Dict[str, Any]:
        token = self.require_tool(bearer_token, "omnibrain.list_scopes")
        scopes = []
        for scope_id in token.scope_ids:
            scope = self.store["scopes"].get(scope_id)
            if scope and scope.get("tenant_id") == token.tenant_id and scope.get("status") == "active":
                scopes.append(
                    {
                        "scope_id": scope["scope_id"],
                        "name": scope["name"],
                        "description": scope["description"],
                    }
                )
        audit = self._log(
            token,
            "omnibrain.list_scopes",
            {},
            "allowed" if scopes else "denied",
            result_ids=[{"scope_id": item["scope_id"]} for item in scopes],
            reason_codes=[] if scopes else ["NO_ACTIVE_BOUND_SCOPES"],
        )
        return {"decision": "allowed" if scopes else "denied", "scopes": scopes, "audit_id": audit.audit_id}

    def explain_access(self, bearer_token: Optional[str], document_id: Optional[str] = None) -> Dict[str, Any]:
        token = self.require_tool(bearer_token, "omnibrain.explain_access")
        document = self.store["documents"].get(document_id or "")
        allowed = False
        internal_reasons = ["NOT_ACCESSIBLE_WITH_CURRENT_GRANT"]
        if document:
            allowed, internal_reasons = self._document_allowed(document, token, token.scope_ids, {})
        decision = "allowed" if allowed else "denied"
        audit = self._log(
            token,
            "omnibrain.explain_access",
            {"document_id": document_id},
            decision,
            result_ids=[{"document_id": document_id}] if allowed and document_id else [],
            reason_codes=[] if allowed else internal_reasons,
        )
        return {
            "decision": decision,
            "subject": token.subject_display,
            "token_scopes": _safe_scope_names(self.store["scopes"], token.scope_ids),
            "reason_codes": [] if allowed else ["NOT_ACCESSIBLE_WITH_CURRENT_GRANT"],
            "safe_explanation": (
                "This content is accessible with your current OmniBrain grant."
                if allowed
                else DEFAULT_SAFE_EXPLANATION
            ),
            "audit_id": audit.audit_id,
        }

    def revoke_token(self, bearer_token: str) -> None:
        if bearer_token in self.store["tokens"]:
            self.store["tokens"][bearer_token]["revoked_at"] = self._now().isoformat()

    def revoke_grant(self, grant_id: str) -> None:
        if grant_id in self.store["grants"]:
            self.store["grants"][grant_id]["revoked_at"] = self._now().isoformat()

    def set_source_status(self, source_id: str, status: str) -> None:
        if source_id in self.store["sources"]:
            self.store["sources"][source_id]["status"] = status

    def set_document_status(self, document_id: str, status: str) -> None:
        if document_id in self.store["documents"]:
            self.store["documents"][document_id]["status"] = status

    def _validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(filters, dict):
            raise OmniBrainValidationError("INVALID_FILTERS")
        allowed_keys = {"source_kind", "scope_id"}
        unknown = set(filters) - allowed_keys
        if unknown:
            raise OmniBrainValidationError("UNSUPPORTED_FILTER")
        for key, value in filters.items():
            if value is not None and not isinstance(value, str):
                raise OmniBrainValidationError("INVALID_FILTER_VALUE")
        return dict(filters)

    def _max_results_for_scopes(self, scope_ids: Sequence[str]) -> int:
        max_results = 5
        for scope_id in scope_ids:
            scope = self.store["scopes"].get(scope_id)
            if scope:
                max_results = min(max_results, int(scope.get("max_results", max_results)))
        return max_results

    def _document_allowed(
        self,
        document: Dict[str, Any],
        token: TokenIntrospection,
        scope_ids: Sequence[str],
        filters: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        reasons = []
        source = self.store["sources"].get(document.get("source_id"))
        if document.get("tenant_id") != token.tenant_id:
            reasons.append("TENANT_MISMATCH")
        if not source or source.get("tenant_id") != token.tenant_id:
            reasons.append("SOURCE_TENANT_MISMATCH")
        if source and source.get("status") != "active":
            reasons.append("SOURCE_INACTIVE")
        if document.get("status") != "active":
            reasons.append("DOCUMENT_INACTIVE")
        if filters.get("source_kind") and filters["source_kind"] != document.get("source_kind"):
            reasons.append("SOURCE_KIND_FILTER_MISMATCH")
        if source and self._source_acl_is_stale(source):
            reasons.append("ACL_STALE")
        if not self._document_in_active_scope(document["document_id"], token.tenant_id, scope_ids):
            reasons.append("DOCUMENT_NOT_IN_SCOPE")
        if not _has_acl_read(document, token, self.store["principals"]):
            reasons.append("SOURCE_ACL_NO_READ")
        return not reasons, reasons

    def _document_in_active_scope(self, document_id: str, tenant_id: Optional[str], scope_ids: Sequence[str]) -> bool:
        for scope_id in scope_ids:
            scope = self.store["scopes"].get(scope_id)
            if not scope:
                continue
            if scope.get("tenant_id") != tenant_id or scope.get("status") != "active":
                continue
            if document_id in scope.get("document_ids", []):
                return True
        return False

    def _source_acl_is_stale(self, source: Dict[str, Any]) -> bool:
        last_synced = _parse_dt(source.get("acl_last_synced_at"))
        if not last_synced:
            return True
        ttl_days = int(source.get("acl_ttl_days", 0))
        return self._now() - last_synced > timedelta(days=ttl_days)

    def _score(self, query: str, terms: Set[str], document: Dict[str, Any], chunk: Dict[str, Any]) -> int:
        haystack = f"{document.get('title', '')} {chunk.get('text', '')}".lower()
        if not terms:
            return 0
        matched_terms = {term for term in terms if term in haystack}
        # Deterministic stub retrieval intentionally requires enough overlap to
        # avoid surfacing unrelated authorized docs for restricted-only queries.
        required_matches = min(2, len(terms))
        if len(matched_terms) < required_matches:
            return 0
        score = len(matched_terms) * 3
        if query.lower() in haystack:
            score += 10
        return score

    def _result_payload(self, document: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "document_id": document["document_id"],
            "chunk_id": chunk["chunk_id"],
            "title": document["title"],
            "snippet": chunk["text"],
            "citation": document["citation"],
            "source_kind": document["source_kind"],
            "modified_at": document["modified_at"],
        }

    def _safe_denial(
        self,
        token: TokenIntrospection,
        tool_name: str,
        payload: Dict[str, Any],
        reason_codes: Sequence[str],
        *,
        safe_message: str,
    ) -> Dict[str, Any]:
        audit = self._log(
            token,
            tool_name,
            payload,
            "denied",
            result_ids=[],
            denied_counts={code: 1 for code in reason_codes},
            reason_codes=list(reason_codes),
        )
        if tool_name == "omnibrain.search":
            return {"decision": "denied", "results": [], "safe_message": safe_message, "audit_id": audit.audit_id}
        return {"decision": "denied", "safe_message": safe_message, "audit_id": audit.audit_id}

    def _log(
        self,
        token: TokenIntrospection,
        tool_name: str,
        payload: Dict[str, Any],
        decision: str,
        *,
        result_ids: Optional[List[Dict[str, str]]] = None,
        denied_counts: Optional[Dict[str, int]] = None,
        reason_codes: Optional[List[str]] = None,
    ) -> QueryLogEntry:
        audit_id = f"qry_{len(self.query_logs) + 1:03d}"
        entry = QueryLogEntry(
            audit_id=audit_id,
            timestamp=self._now().isoformat(),
            tenant_id=token.tenant_id,
            token_id=token.token_id,
            grant_ids=list(token.grant_ids),
            subject_principal_id=token.subject_principal_id,
            subject_display=token.subject_display,
            tool_name=tool_name,
            request_summary=_redacted_request_summary(payload),
            decision=decision,
            result_ids=result_ids or [],
            denied_counts=denied_counts or {},
            reason_codes=reason_codes or [],
        )
        self.query_logs.append(entry)
        return entry

    @staticmethod
    def _increment_reasons(denied_counts: Dict[str, int], reasons: Sequence[str]) -> None:
        for reason in reasons:
            denied_counts[reason] = denied_counts.get(reason, 0) + 1


def discover_omnibrain_mcp_tools() -> List[Dict[str, Any]]:
    """Return stable tool metadata for MCP discovery/adapters."""

    return [
        {
            "name": "omnibrain.search",
            "description": "Search authorized OmniBrain chunks/snippets with hard tenant/scope/source/document/ACL filtering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "source_kind": {"type": "string"},
                            "scope_id": {"type": "string"},
                        },
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "omnibrain.get_document",
            "description": "Return full text for one authorized document; returns safe denial for inaccessible IDs.",
            "input_schema": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        },
        {
            "name": "omnibrain.list_scopes",
            "description": "List non-sensitive scopes bound to the current OmniBrain MCP token.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "omnibrain.explain_access",
            "description": "Explain current grant access safely without leaking restricted document metadata.",
            "input_schema": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
            },
        },
    ]


__all__ = [
    "OmniBrainAuthenticationError",
    "OmniBrainForbiddenError",
    "OmniBrainMcpGateway",
    "OmniBrainValidationError",
    "TokenIntrospection",
    "build_demo_fixture_store",
    "discover_omnibrain_mcp_tools",
]
