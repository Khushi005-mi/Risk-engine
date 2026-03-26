# Credit Risk Engine

## Overview

The Credit Risk Engine is a deterministic, version-controlled batch scoring system
designed to estimate Probability of Default (PD) for unsecured retail loans.

The system is built to:
- Enforce strict data contracts
- Guarantee deterministic inference
- Maintain reproducibility across environments
- Support regulatory-grade auditability

---

## Architecture Layers

1. Data Contract Layer
2. Validation Engine
3. Feature Engineering Layer
4. Modeling Engine
5. Calibration Layer
6. Inference Orchestration
7. Governance & Model Registry

Each layer is isolated and single-responsibility.

---

## Core Principles

- No silent data coercion
- No implicit type casting
- Deterministic model inference
- Train / inference parity
- Full versioning of artifacts
- Environment pinning
- Reproducible scoring

---

## Directory Structure

credit-risk-engine/
│
├── pyproject.toml
├── README.md
├── src/
│   └── credit_risk_engine/
│       ├── data_contract/
│       ├── validation/
│       ├── features/
│       ├── modeling/
│       ├── calibration/
│       ├── inference/
│       ├── governance/
│       └── utils/
│
├── models/
├── configs/
├── tests/
└── scripts/

---

## Execution Modes

### Train Mode
Trains model and saves:
- model artifact
- calibration artifact
- feature schema hash
- training metadata

### Inference Mode
Validates input file and outputs:
- scored CSV with calibrated PD
- batch metadata log

---

## Reproducibility

- All random seeds fixed
- Dependencies pinned
- Model artifacts hashed
- Feature order locked

---

## Compliance Objectives

Designed to support:
- Audit replay
- Deterministic scoring validation
- Model version governance
- Regulatory review documentation

---

## Status

Phase 5A – Core Spine Implementation