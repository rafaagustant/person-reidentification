# Refactor plan: error-driven demonstration

## Baseline audit

- `ui/app.py` is a 2,848-line mixed UI/controller module. It duplicates the per-case recommendation data already present in `config/presets.py` and exposes several scene-category presets.
- `config/presets.py` has both current case recommendations and legacy stage/visual presets. The UI reads both, so the runtime source is not unambiguous.
- Tracking already writes a BoT-SORT YAML with `with_reid: false`, instantiates a YOLO model for every camera, and records most tracking inputs.
- Crops are collected during tracking and embeddings are built only from valid tracks. However, crop sampling uses a global filter configuration during Re-ID, losing the originating camera configuration.
- GT matching selects the best GT independently for every prediction. It is not one-to-one within a camera/source-frame and can over-count TP.
- Tracking standard metrics clamp `mota_simple` although `raw_mota_simple` is available. Global-ID purity is currently a pure-group rate but is named `mean_global_id_purity` in pairwise output.
- Cross-camera MNN uses all cameras in one nearest-neighbour pool; it must instead operate separately for each camera pair. Global-ID component order is not explicitly deterministic.
- Embeddings are reused based only on two cache files; no cache manifest validates model, crops, sampling, or feature dimensions.

## Files to change

- `config/presets.py`: retain a single case-recommendation source; add one permissive error-analysis tuning configuration and strict normalization/validation.
- `core/evaluation.py`: deterministic one-to-one IoU matching, raw MOTA, and correctly defined Global-ID purity.
- `core/association.py`: per-camera-pair MNN and deterministic union-find output.
- `core/reid.py`, `core/pipeline.py`: camera-aware crop sampling and manifest-validated embedding cache; persist a run manifest.
- `ui/app.py`, plus focused UI modules where useful: two configuration modes, staged actions, concise results and run comparison.
- `tests/`: small synthetic tests for critical processing paths.
- `REFACTOR_REPORT.md`: implementation and verification record.

## Intended behaviour

1. “Rekomendasi Case” uses only `CASE_RECOMMENDATIONS`.
2. “Tuning Analisis Kesalahan” starts from one permissive detection/tracking/filter setup and conservative Re-ID thresholds, then permits manual edits.
3. Runtime configuration is validated, used unchanged by the pipeline, and recorded in `config_used.json`, per-camera configuration, Re-ID configuration, tracker YAML, and `run_manifest.json`.
4. Embeddings are only reused when their inputs and model fingerprint match the embedding manifest.
5. Ground-truth matching and academic metrics are one-to-one and reproducible.

## Risks and checks

- Existing output folders can lack the new cache manifest; they will be safely recomputed.
- UI restructuring can affect only presentation/state; pipeline functions remain independently callable.
- Tests use synthetic DataFrames and temporary files; no video, GPU, or model weights are required.
- Run `pytest`, `python -m compileall`, imports for core modules and the Streamlit entry point.
