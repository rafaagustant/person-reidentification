# Refactor report: error-driven demonstration

## Summary

The application now presents the thesis pipeline as three explicit stages: local tracking, post-tracking OSNet Re-ID/Global ID association, and rendering. BoT-SORT remains local-ID-only (`with_reid: false`), and OSNet still extracts crop embeddings after valid-track filtering.

## Structure

- `ui/app.py` is the small Streamlit entry point and stage controller.
- `ui/config_panel.py` contains the two configuration modes and manual parameter form.
- `ui/results_panel.py` contains the four result views.
- Core tracking, Re-ID, association, evaluation, gallery, render, and pipeline modules remain separate.

## UI and tuning

Only two modes are exposed:

1. **Rekomendasi Case** reads the final per-case configuration solely from `config.presets.CASE_RECOMMENDATIONS`.
2. **Tuning Analisis Kesalahan** starts from the one permissive `ERROR_ANALYSIS_TUNING_CONFIG`, with conservative Re-ID thresholds, and exposes manual controls grouped by stage.

The process buttons enforce Tracking → Re-ID → Render. Results are limited to Ringkasan, Video, Gallery, and Analisis. Diagnostics from the pipeline are retained as short tuning suggestions; parameters are never changed automatically.

## Processing and evaluation changes

- Every camera gets a fresh YOLO/tracker instance; tracker YAML explicitly disables internal Re-ID.
- Crop sampling now uses the filter configuration for the track's origin camera.
- Crop embeddings are L2-normalized, mean-pooled per track, then normalized again; cosine similarity uses those normalized vectors.
- GT matching is deterministic one-to-one IoU matching within the same camera and source frame.
- MOTA is no longer clipped; negative values are preserved.
- Cross-camera MNN is applied separately for each camera pair. Union-find roots and Global ID ordering are deterministic and allow valid transitive associations.
- Pairwise `global_id_purity` is now weighted dominant-GT composition across evaluated Global-ID members. The older boolean group-rate is not labelled as mean purity.
- Re-ID cache reuse requires `embedding_manifest.json` to match OSNet architecture/weight fingerprint, valid tracks, selected crops, sampling, preprocessing, and embedding dimension.

## Runtime configuration and outputs

Each run records `config_used.json`, `camera_config_used.json` per camera, `reid_config_used.json`, generated tracker YAML, tracking runtime YAML, `embedding_manifest.json`, and `run_manifest.json`.

| Parameter group | UI/config source | Runtime consumer | Recorded output |
|---|---|---|---|
| YOLO (`yolo_conf`, `yolo_iou`, `imgsz`) | `ui/config_panel.py` → normalized config | `core.tracking.run_track_once` | tracker/runtime YAML, config/manifest |
| BoT-SORT thresholds/buffer | same | `core.tracking.make_botsort_yaml` | tracker YAML |
| Valid-track filters | same, including per-camera config | `core.tracking.filter_valid_tracks` | camera config, config/manifest |
| Crop sampling | same, per origin camera | `core.reid.build_sampled_track_crop_df` | sampled CSV, embedding manifest |
| Re-ID association | same | `core.association.assign_global_ids` | Re-ID config, pair CSV, manifest |

## Tests and verification

- Synthetic tests added in `tests/test_refactor_core.py`: one-to-one matching, no cross-frame/camera match, negative MOTA, per-camera sampling, per-pair MNN/transitive union, cache invalidation, config validation, and BoT-SORT YAML Re-ID disablement.
- `python -m compileall -q core config ui tests app.py`: passed using the project virtual environment.
- Import check for `ui.app`, config panel, results panel, and core pipeline: passed.
- `pytest -q --basetemp=.\.pytest_tmp`: 7 tests passed.
- `pytest` ditambahkan melalui `requirements-dev.txt` sebagai development dependency.

## Run command

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Thesis implications and remaining limitations

All Bab IV tables derived from GT matching must be recalculated: TP/FP/FN, precision, recall, F1, MOTA, ID switch, fragmentation, pairwise Re-ID metrics, false merge/split, and Global ID purity. Previous values may have over-counted matches or shown clipped MOTA.

Full video/GPU execution was not performed because the case video and model assets are not available in the repository workspace. The Streamlit interaction itself also still requires a browser run with those assets.
