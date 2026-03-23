# PINNProof

PINNProof is a **validation and verification toolbox for power-system component surrogate models**.

This repository is now organized like a standard Python package with `src/` layout and runnable Python examples.

## Installation

```bash
pip install -e .
```

## Package layout

```text
src/pinnproof/
  validation/
    metrics.py          # RMSE/MAE/NRMSE and trajectory-level summaries
  verification/
    residuals.py        # residual calculations (e.g., swing equation)
    report.py           # high-level pass/fail verification reports
examples/python/
  validation_quickstart.py
  verification_quickstart.py
```

`src/pinnproof/` is the real package. The examples add `src/` to `sys.path` only so they can be run directly from a local checkout before installation.

## Quickstart

Validation:

```bash
python examples/python/validation_quickstart.py
```

Verification:

```bash
python examples/python/verification_quickstart.py
```

## Scope

The toolbox is intended for surrogate models of dynamic power-system components (e.g., synchronous machines and related models), where:

- **Validation** quantifies surrogate-vs-reference trajectory errors.
- **Verification** checks physical consistency using differential-equation residuals.
