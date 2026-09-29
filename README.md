# ModelForge 3.0

ModelForge is split into two experiences backed by the same model-inspection and research concepts.

## 1. Beginner Mode — comprehensive interactive model education

`beginner/` is static and can be hosted on GitHub Pages / Cloudflare Pages. It does not require an account or backend.

### Real CNN lab

The old dense 8×8 digit toy has been replaced with a trained spatial CNN:

`28×28 input → Conv2d(1→4) → ReLU → MaxPool → Conv2d(4→8) → ReLU → MaxPool → Flatten → Linear(392→10) → Softmax`

The bundled CNN reaches about **98% held-out accuracy** on its digit validation split. Training uses translated/noisy examples to improve robustness. Browser preprocessing crops the drawn/uploaded digit, preserves aspect ratio, centers it, rescales it to the model's 28×28 representation, and detects/inverts light backgrounds.

Beginner Mode supports:

- draw-a-digit input
- image upload
- real browser-side CNN inference
- prediction probabilities
- feature-map visualization after every convolution/ReLU/pooling stage
- layer-by-layer tensor flow
- ablate one map/unit
- scale activations
- add activation noise
- before/after comparison
- ten guided CNN lessons
- local progress only (`localStorage`)

### Block Academy

A second beginner workspace teaches the reusable blocks that appear in Advanced Mode. Each block includes plain-language explanation, visual/animated demonstration, shape/math intuition, PyTorch equivalent, research motivation, and a check question.

Included blocks:

- Conv2d
- ReLU
- GELU
- MaxPool2d
- AvgPool2d
- AdaptiveAvgPool2d
- BatchNorm2d
- LayerNorm
- Flatten
- Linear
- Softmax
- Dropout
- Embedding
- MultiheadAttention
- ResidualBlock

## 2. Advanced Mode — research workbench

All v2 research/model features remain, including model hierarchy inspection, probes, interventions, real-input runs, causal patching primitives, persistent runs, code export, multiple model tabs, architecture canvas, model outputs and activation inspection.

### Dataset workspace

Advanced Mode now has a dedicated **Data** rail.

Users can:

- create an account and sign in
- upload datasets privately
- use up to **1 GiB of dataset storage per user**
- upload CSV, JSON, JSONL/NDJSON and Parquet
- preview rows and inferred dtypes
- inspect missing-value counts, uniqueness and numeric statistics
- filter rows with equals / contains / numeric comparisons / missing checks
- create cleaned derived copies
- remove duplicate rows
- drop missing rows
- drop selected columns
- visualize numeric columns as histograms
- visualize categorical columns as frequency bars
- delete datasets and immediately recover quota

Uploads are streamed in chunks; a 1 GiB upload is not read into application RAM as one giant byte string.

### Local development authentication/storage

The runnable local package uses:

- SQLite for users, sessions, dataset metadata, runs and experiments
- PBKDF2-SHA256 password hashing with per-user salts
- HTTP-only session cookies
- private per-user dataset directories under `~/.modelforge/datasets/`

This lets the full workflow be tested before AWS credentials exist.

## 3. AWS production infrastructure

`iac/aws/` contains Terraform for the production data/auth layer:

- Amazon Cognito User Pool + web client
- private S3 dataset bucket
- public access blocking
- S3 server-side encryption
- S3 versioning
- multipart-upload cleanup
- configurable CORS
- DynamoDB dataset metadata table
- DynamoDB per-user usage table

The application enforces the 1 GiB quota because S3 does not provide native per-prefix/user quotas.

No AWS credentials are included in the project. `modelforge.cloud` is prepared to use standard AWS credential resolution later (AWS profile, environment, ECS/EC2 role, etc.).

Install optional AWS support:

```bash
pip install -e '.[aws]'
```

Terraform:

```bash
cd iac/aws
terraform init
terraform plan
terraform apply
```

After deployment, configure the returned values through environment variables rather than hardcoding credentials:

```bash
export MODELFORGE_AWS_MODE=1
export MODELFORGE_AWS_REGION=us-west-2
export MODELFORGE_DATASET_BUCKET=...
export MODELFORGE_COGNITO_USER_POOL_ID=...
export MODELFORGE_COGNITO_CLIENT_ID=...
export MODELFORGE_DATASET_TABLE=...
export MODELFORGE_USAGE_TABLE=...
```

The AWS resources are intentionally not deployed in this package because credentials/account details have not been supplied yet.

## Run Advanced Mode

```bash
pip install -e .
modelforge serve
```

or:

```bash
python -m uvicorn modelforge.api:app --reload
```

Open:

- Advanced: `http://127.0.0.1:8000/`
- Beginner: `http://127.0.0.1:8000/beginner/`

## Static Beginner hosting

The `beginner/` directory can be served independently as static files. Ensure `index.html` and `cnn_model.json` are published together.

## Validation

Current automated suite: **19 tests passing**.

Additional validation performed:

- Advanced browser JavaScript syntax check
- Beginner browser JavaScript syntax check
- browser-equivalent CNN inference checked directly against bundled weights
- signup/session workflow
- 1 GiB quota metadata
- dataset CSV upload
- preview
- statistics
- row filtering
- cleaned-copy creation
- data visualization endpoint
- auth-required access control

## Production security note

The local auth system is for the local package/development workflow. For a public hosted Advanced deployment, switch authentication to Cognito, use HTTPS-only secure cookies/JWT validation, presigned S3 uploads, malware/content scanning if accepting arbitrary files, rate limits, audit logging, and normal cloud IAM least-privilege controls.


## v3.1 UX changes

### Advanced research results
Every action launched from the Research workbench now opens a dedicated modal result workspace rather than dumping raw JSON into the sidebar. Results use summary metrics, structured tables, code views where appropriate, and an optional raw-payload disclosure for exact debugging/reproducibility data. Errors use the same result window with a clear failure explanation.

### Beginner layout
CNN Lab was reorganized to keep the learning loop visible on a normal desktop screen: input on the left of the main workspace, network path and activations in the center, and prediction/experiments/lesson guidance in the right panel. The page itself no longer requires long scrolling at typical desktop widths. Feature maps scroll locally when there are many of them. Block Academy uses the same pattern: interactive animation in the center and researcher-oriented explanation, PyTorch code, and quiz in the right panel. Responsive fallback restores normal document scrolling on narrow screens.


## v3.2 UX note
Advanced mode now uses a more schematic, Simulink-inspired symbolic rendering for model operators, so Conv/ReLU/Pool/Linear and related modules appear as operator symbols with captions rather than large rectangular name blocks.

## v3.2.1 schematic symbol correction
This patch replaces the remaining card-like operator illustrations with true schematic-style operator symbols and adds routing primitives (Add, Split, Concat, Input, Output).


## v3.3 Advanced canvas style
The Advanced canvas now uses a compact Simulink-inspired block language: uniform engineering blocks, small operator glyphs, compact captions, square signal ports, thin wiring, and reduced decorative chrome. The operator glyph supports recognition, while the block/port geometry remains consistent for dense research schematics.

## v3.4 paper-style architecture notation

Advanced mode now uses deliberately plain, blocky architecture-diagram conventions rather than custom pictograms. Examples include `Conv 7×7, 64 /2`, `BN 64`, `ReLU`, `MaxPool 3×3 /2`, `BasicBlock ×2`, and `FC 512→1000`. Imported-model labels and inspector details are generated from the actual PyTorch module attributes. Custom design blocks now carry editable constructor-style configuration (channels, kernel, stride, padding, dilation, groups, bias, feature sizes, dropout, attention heads, and related fields) and update their diagram labels immediately.


## v3.5 usability
The Advanced canvas now uses a dotted drafting background and auto-fit/auto-center when opening tabs, models, hierarchy levels, and unit views. Paper-style blocks keep their standard notation with a small restrained category mark for quick scanning. Beginner Mode is guided step-by-step by default.


## v3.6 canvas fixes
- Dotted drafting background now belongs to the viewport, so it remains visible in Design and Inspect modes.
- Automatic model zoom-to-fit is disabled; inspected models stay at a readable 100% unless the user changes zoom.
- Hierarchy drill-down preserves the current readable scale instead of shrinking the architecture.
- Design-wire SVG is hidden and cleared in Inspect mode, eliminating stale pink connection curves.
- Fit controls now center at the current zoom instead of resizing content.
