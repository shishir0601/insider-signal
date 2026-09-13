"""
parser.py

Parses a single Form 4 (ownershipDocument) XML filing into raw
transaction records — one dict per non-derivative transaction, with the
filing-level issuer and reporting-owner fields repeated onto each row
(a filing can report several transactions for the same insider).

Deliberately reads only <nonDerivativeTransaction> elements, not
<nonDerivativeHolding> (a pre-existing position with no transaction —
nothing to score) or <derivativeTransaction> (options/RSUs, which
convert into a corresponding non-derivative "M" line when exercised —
that's the line this project's schema cares about; the derivative side
records the same event from the option's point of view, not a second
transaction). See normalize.py for how transaction codes and the
acquired/disposed flag become this project's BUY/SELL schema.

Uses the standard library's xml.etree.ElementTree — no new dependency
for something this project only needs to walk once per filing.
"""

import xml.etree.ElementTree as ET


def _text(node, path, default=None):
    """Read `node.find(path).text`, or `default` if the element is
    absent or empty. Handles the many places Form 4 XML nests a bare
    value one level down (e.g. <transactionShares><value>100</value></transactionShares>)."""
    if node is None:
        return default
    el = node.find(path)
    if el is None or el.text is None:
        return default
    text = el.text.strip()
    return text if text else default


def _owner_title(reporting_owner_el) -> str:
    """officerTitle if present (most specific, e.g. "Chief Financial
    Officer"); otherwise a comma-joined list built from the relationship
    flags (Director / Officer / 10% Owner / Other)."""
    rel = reporting_owner_el.find("reportingOwnerRelationship") if reporting_owner_el is not None else None
    if rel is None:
        return ""
    officer_title = _text(rel, "officerTitle")
    if officer_title:
        return officer_title
    roles = []
    if _text(rel, "isDirector") == "1":
        roles.append("Director")
    if _text(rel, "isOfficer") == "1":
        roles.append("Officer")
    if _text(rel, "isTenPercentOwner") == "1":
        roles.append("10% Owner")
    if _text(rel, "isOther") == "1":
        roles.append("Other")
    return ", ".join(roles)


def parse_form4_xml(xml_text: str) -> list:
    """
    Returns a list of dicts, one per <nonDerivativeTransaction> in the
    filing. Returns an empty list for a filing with no non-derivative
    transactions (e.g. one reporting only holdings, or only derivative
    transactions) — that's a valid, common filing shape, not an error.

    Raises ValueError for XML that doesn't parse, or that isn't a Form 4
    ownershipDocument at all — those ARE errors, and the caller (an
    ingestion loop pulling many filings) decides whether to skip and log
    or halt; this function never silently returns an empty list for bad
    input the way it does for a well-formed filing with nothing to report.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"could not parse Form 4 XML: {e}") from e

    if root.tag != "ownershipDocument":
        raise ValueError(f"not a Form 4 ownershipDocument (root element was <{root.tag}>)")

    issuer = root.find("issuer")
    issuer_cik = _text(issuer, "issuerCik")
    issuer_name = _text(issuer, "issuerName")
    issuer_ticker = _text(issuer, "issuerTradingSymbol")

    reporting_owner = root.find("reportingOwner")
    owner_id = reporting_owner.find("reportingOwnerId") if reporting_owner is not None else None
    insider_cik = _text(owner_id, "rptOwnerCik")
    insider_name = _text(owner_id, "rptOwnerName")
    insider_title = _owner_title(reporting_owner)

    document_type = _text(root, "documentType")

    rows = []
    non_derivative_table = root.find("nonDerivativeTable")
    if non_derivative_table is not None:
        for tx in non_derivative_table.findall("nonDerivativeTransaction"):
            coding = tx.find("transactionCoding")
            amounts = tx.find("transactionAmounts")
            ownership = tx.find("ownershipNature")

            rows.append({
                "issuer_cik": issuer_cik,
                "issuer_name": issuer_name,
                "issuer_ticker": issuer_ticker,
                "insider_cik": insider_cik,
                "insider_name": insider_name,
                "insider_title": insider_title,
                "document_type": document_type,
                "transaction_date": _text(tx, "transactionDate/value"),
                "transaction_code": _text(coding, "transactionCode"),
                "shares": _text(amounts, "transactionShares/value"),
                "price_per_share": _text(amounts, "transactionPricePerShare/value"),
                "acquired_disposed_code": _text(amounts, "transactionAcquiredDisposedCode/value"),
                "ownership_type": _text(ownership, "directOrIndirectOwnership/value"),
                "security_title": _text(tx, "securityTitle/value"),
            })
    return rows
