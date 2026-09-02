"""DECISION SIDE — owned by the decision-layer teammate.

`reference_verify.py` is the reference implementation of the receipt
reconciliation described in docs/CONTRACT_v1.1.md §5. It is deliberately not
imported by the vision pipeline's production path; scripts/pipeline.py only
calls it in its demo self-test.

PR #1 (feature/unpaid-item-verification) lands VerificationService,
PaymentRepository and the payment JSON adapter in this package. Keep this
module free of vision imports so the boundary stays JSON-only.
"""
