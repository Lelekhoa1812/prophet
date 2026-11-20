# Notice

## TimesFM 3 weights are non-commercial

Prophet loads the TimesFM 3.0 checkpoint (`google/timesfm-3.0-pytorch`) as its forecasting model. Those weights are
published under the TimesFM Non-Commercial License v1.0. Use of this project is therefore **research and evaluation
only**:

- Do not use its outputs for commercial decisions, client deliverables or paid services.
- Do not redistribute the weights with this project.
- Review the licence at the model card before any use beyond research.

This notice is intentionally kept out of the web console and out of generated files; it lives here and in the README.

## Data labelling

Figures produced from generated test data (for example the synthetic demand twin in `src/prophet_agent/synth/`) must be
labelled as synthetic wherever they are reported in documentation or evaluation. Real-data results are labelled with
their source. The console reports whatever the person supplied as their own data.

## Secrets

`.env` holds NVIDIA NIM keys and is git-ignored. Keys are never printed, logged, stored in the ledger or committed.
