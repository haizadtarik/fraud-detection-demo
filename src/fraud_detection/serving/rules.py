from fraud_detection.serving.schemas import Transaction

# In production these come from config / a managed list, not literals.
HARD_AMOUNT_CAP = 10_000_000  # kill-switch ceiling
NEW_PAYEE_REVIEW_AMOUNT = 1_000_000  # large transfer to a first-time payee → review

KNOWN_MULE_ACCOUNTS: set[str] = set()  # populated from fraud-ops feed in prod


def apply_rules(txn: Transaction, dest_is_new: int) -> tuple[str | None, str | None]:
    """Return (decision, rule_name) if a rule fires, else (None, None)."""
    if txn.amount > HARD_AMOUNT_CAP:
        return "BLOCK", "hard_amount_cap"
    if txn.name_dest in KNOWN_MULE_ACCOUNTS:
        return "BLOCK", "known_mule_recipient"
    if dest_is_new == 1 and txn.amount > NEW_PAYEE_REVIEW_AMOUNT:
        return "HOLD", "new_payee_large_amount"  # BNM cooling-off analogue
    return None, None
