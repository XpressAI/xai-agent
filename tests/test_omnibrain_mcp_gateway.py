import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnibrain_mcp_gateway import (
    OmniBrainAuthenticationError,
    OmniBrainForbiddenError,
    OmniBrainMcpGateway,
    OmniBrainValidationError,
    discover_omnibrain_mcp_tools,
)


MIA_TOKEN = "demo-mia-support-token"
SEARCH_ONLY_TOKEN = "demo-mia-search-only-token"
REVOKED_TOKEN = "demo-revoked-token"
GRANT_REVOKED_TOKEN = "demo-grant-revoked-token"
RESTRICTED_STRINGS = [
    "Project Falcon",
    "Acquisition Plan",
    "gdrive://Exec",
    "valuation model",
    "doc_exec_acquisition_plan",
    "chk_exec_plan_001",
]


def serialized(value):
    return json.dumps(value, sort_keys=True)


def assert_no_restricted_leak(response):
    body = serialized(response)
    for restricted in RESTRICTED_STRINGS:
        assert restricted not in body


def test_tool_discovery_exposes_expected_contracts():
    tools = discover_omnibrain_mcp_tools()
    names = {tool["name"] for tool in tools}
    assert names == {
        "omnibrain.search",
        "omnibrain.get_document",
        "omnibrain.list_scopes",
        "omnibrain.explain_access",
    }


def test_allowed_search_returns_authorized_support_result_and_audit_log():
    gateway = OmniBrainMcpGateway()

    response = gateway.search(MIA_TOKEN, query="current refund exception policy", limit=5)

    assert response["decision"] == "allowed"
    assert response["scope_names"] == ["support-safe"]
    assert response["results"]
    first = response["results"][0]
    assert first["document_id"] == "doc_refund_policy"
    assert first["chunk_id"] == "chk_refund_policy_003"
    assert first["title"] == "Refund Exceptions"
    assert "finance approval" in first["snippet"]
    assert response["audit_id"] == "qry_001"
    assert gateway.query_logs[-1].decision == "allowed"
    assert {tuple(sorted(item.items())) for item in gateway.query_logs[-1].result_ids} >= {
        tuple(sorted({"document_id": "doc_refund_policy", "chunk_id": "chk_refund_policy_003"}.items()))
    }
    assert "query_hash" in gateway.query_logs[-1].request_summary


def test_search_denied_for_restricted_exec_content_leaks_no_metadata():
    gateway = OmniBrainMcpGateway()

    response = gateway.search(MIA_TOKEN, query="Project Falcon acquisition plan valuation model", limit=5)

    assert response["decision"] == "denied"
    assert response["results"] == []
    assert response["safe_message"] == "No accessible OmniBrain results are available for this request."
    assert_no_restricted_leak(response)
    log = gateway.query_logs[-1]
    assert log.decision == "denied"
    assert "SOURCE_ACL_NO_READ" in log.denied_counts or "DOCUMENT_NOT_IN_SCOPE" in log.denied_counts


def test_get_document_denied_for_restricted_known_id_leaks_no_title_text_or_citation():
    gateway = OmniBrainMcpGateway()

    response = gateway.get_document(MIA_TOKEN, "doc_exec_acquisition_plan")

    assert response["decision"] == "denied"
    assert response["safe_message"] == "That document is not accessible with your current OmniBrain grant."
    assert_no_restricted_leak(response)
    assert gateway.query_logs[-1].decision == "denied"


def test_explain_access_collapses_sensitive_reasons_and_leaks_no_metadata():
    gateway = OmniBrainMcpGateway()

    response = gateway.explain_access(MIA_TOKEN, "doc_exec_acquisition_plan")

    assert response["decision"] == "denied"
    assert response["subject"] == "mia@acme.example"
    assert response["token_scopes"] == ["support-safe"]
    assert response["reason_codes"] == ["NOT_ACCESSIBLE_WITH_CURRENT_GRANT"]
    assert "Ask an admin" in response["safe_explanation"]
    assert_no_restricted_leak(response)


def test_list_scopes_returns_only_bound_safe_scope_metadata():
    gateway = OmniBrainMcpGateway()

    response = gateway.list_scopes(MIA_TOKEN)

    assert response["decision"] == "allowed"
    assert response["scopes"] == [
        {
            "scope_id": "scope_support_safe",
            "name": "support-safe",
            "description": "Support playbooks and customer-facing policy",
        }
    ]
    assert_no_restricted_leak(response)


def test_tool_authorization_denies_get_document_for_search_only_token():
    gateway = OmniBrainMcpGateway()

    with pytest.raises(OmniBrainForbiddenError) as exc:
        gateway.get_document(SEARCH_ONLY_TOKEN, "doc_refund_policy")

    assert exc.value.status_code == 403
    assert exc.value.reason_code == "TOOL_NOT_ALLOWED"


def test_missing_invalid_revoked_and_grant_revoked_tokens_raise_401():
    gateway = OmniBrainMcpGateway()

    for token, reason in [
        (None, "TOKEN_MISSING"),
        ("not-a-real-token", "TOKEN_INVALID"),
        (REVOKED_TOKEN, "TOKEN_REVOKED"),
        (GRANT_REVOKED_TOKEN, "GRANT_REVOKED"),
    ]:
        with pytest.raises(OmniBrainAuthenticationError) as exc:
            gateway.search(token, query="refund")
        assert exc.value.status_code == 401
        assert exc.value.reason_code == reason


def test_immediate_token_revocation_removes_previous_search_access():
    gateway = OmniBrainMcpGateway()
    assert gateway.search(MIA_TOKEN, query="refund exception")["decision"] == "allowed"

    gateway.revoke_token(MIA_TOKEN)

    with pytest.raises(OmniBrainAuthenticationError) as exc:
        gateway.search(MIA_TOKEN, query="refund exception")
    assert exc.value.reason_code == "TOKEN_REVOKED"


def test_source_revocation_removes_previous_search_access_without_leak():
    gateway = OmniBrainMcpGateway()
    assert gateway.search(MIA_TOKEN, query="refund exception")["decision"] == "allowed"

    gateway.set_source_status("src_confluence_support", "revoked")
    response = gateway.search(MIA_TOKEN, query="refund exception")

    assert response["decision"] == "denied"
    assert response["results"] == []
    assert "Refund Exceptions" not in serialized(response)
    assert "confluence://Support/Refund Exceptions" not in serialized(response)
    assert gateway.query_logs[-1].denied_counts["SOURCE_INACTIVE"] >= 1


def test_requested_unbound_scope_filter_is_safely_denied():
    gateway = OmniBrainMcpGateway()

    response = gateway.search(
        MIA_TOKEN,
        query="release runbook",
        filters={"scope_id": "scope_engineering_safe"},
    )

    assert response["decision"] == "denied"
    assert response["results"] == []
    assert "Release Runbook" not in serialized(response)
    assert "sharepoint://Engineering" not in serialized(response)


def test_stale_acl_and_cross_tenant_candidates_are_filtered_before_response():
    gateway = OmniBrainMcpGateway()
    # Add stale doc into support scope to verify ACL freshness still denies.
    gateway.store["scopes"]["scope_support_safe"]["document_ids"].append("doc_stale_acl_policy")
    # Add cross-tenant doc into support scope to verify tenant filter still denies.
    gateway.store["scopes"]["scope_support_safe"]["document_ids"].append("doc_other_tenant_refund_policy")

    response = gateway.search(MIA_TOKEN, query="refund exception policy", limit=5)

    assert response["decision"] == "allowed"
    body = serialized(response)
    assert "Legacy Support Refund Policy" not in body
    assert "Old stale refund policy" not in body
    assert "Other tenant refund policy" not in body
    assert "notion://Umbrella" not in body
    log = gateway.query_logs[-1]
    assert log.denied_counts["ACL_STALE"] >= 1
    assert log.denied_counts["TENANT_MISMATCH"] >= 1


def test_get_document_allowed_for_authorized_document():
    gateway = OmniBrainMcpGateway()

    response = gateway.get_document(MIA_TOKEN, "doc_refund_policy")

    assert response["decision"] == "allowed"
    assert response["document"]["document_id"] == "doc_refund_policy"
    assert response["document"]["title"] == "Refund Exceptions"
    assert "account owner and finance approval" in response["document"]["text"]


def test_invalid_filters_raise_validation_error_before_retrieval():
    gateway = OmniBrainMcpGateway()

    with pytest.raises(OmniBrainValidationError):
        gateway.search(MIA_TOKEN, query="refund", filters={"source_id": "src_gdrive_exec"})

