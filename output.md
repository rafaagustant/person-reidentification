# Ringkasan Output Run Terakhir

Dokumen ini merangkum output yang pernah dihasilkan dari keenam case. Folder `outputs/` bersifat artefak lokal dan tidak dipush ke GitHub, tetapi struktur dan interpretasi hasilnya dijelaskan di sini agar hasil eksperimen tetap bisa dibaca.

Nilai di bawah berasal dari run lokal terakhir sebelum cleanup repo.

## Cara Membaca Output

Setiap case menghasilkan folder:

```text
outputs/<case_id>/<timestamp>/
```

Isi penting:

- `config_used.json`: config aktif level run.
- `<camera>/camera_config_used.json`: config tracking dan filter kamera.
- `<camera>/generated_tracker_config_used.yaml`: YAML BoT-SORT aktual.
- `<camera>/tracking_runtime_manifest.yaml`: argumen runtime tracking.
- `<camera>/local_tracks.csv`: semua hasil tracking.
- `<camera>/local_tracks_valid.csv`: row dari track valid.
- `<camera>/valid_track_summary.csv`: summary track, pass/fail filter, dan alasan filter.
- `valid_tracks_all.csv`: gabungan valid track semua kamera.
- `pair_similarity.csv`: semua kandidat pair Re-ID.
- `merged_pairs.csv`: pair yang digabung.
- `global_track_meta.csv`: metadata Global ID.
- `flow1_tracking_score.csv`: skor tracking.
- `flow2_global_id_score.csv`: skor Re-ID.
- `final_pipeline_score.csv`: skor final pipeline.
- `tracking_diagnostics.csv`: diagnostik tracking.
- `reid_diagnostics.csv`: diagnostik Re-ID.
- `reid_config_used.json`: config Re-ID aktual.
- `reid_stage_log.json` dan `reid_stage_log.txt`: stage yang reuse/recompute.

## Case 1 - Single-Camera Normal Success

Run terakhir:

```text
outputs/case_1_normal_success/20260623_181354/
```

Ringkasan tracking:

- `local_tracks.csv`: 2400 rows.
- `valid_tracks_all.csv`: 2400 rows.
- `valid_track_summary.csv`: 3 tracks.
- `raw_rows`: 2400.
- `valid_rows`: 2400.
- `avg_conf_mean`: sekitar 0.7354.
- `avg_area_mean`: sekitar 39755.84.

Ringkasan Re-ID:

- `global_track_meta.csv`: 3 tracks.
- Jumlah Global ID: 3.
- `merged_pairs.csv`: 0 rows.
- Tidak ada merge karena kondisi normal single-camera dan track sudah stabil.

Skor:

- Tracking score: 1.000.
- Tracking precision: 1.000.
- Tracking recall: 1.000.
- Tracking F1: 1.000.
- Re-ID score: 1.000.
- Expected Global ID: 3.
- Local tracks: 3.
- Global IDs: 3.
- Merged pairs: 0.

Interpretasi:

Case 1 berhasil sempurna. Tracking stabil, tidak ada fragment yang perlu digabung, dan Global ID sama dengan jumlah identitas target.

## Case 2 - Crowded / False Positive Sensitive

Run terakhir:

```text
outputs/case_2_crowded_fragmentation/20260623_182039/
```

Recommended config final yang sekarang dipakai:

```json
{
  "tracking": {
    "yolo_conf": 0.2,
    "yolo_iou": 0.5,
    "imgsz": 640,
    "track_high_thresh": 0.2,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.2,
    "match_thresh": 0.8,
    "track_buffer": 45
  },
  "filter": {
    "min_frames": 80,
    "min_crops": 80,
    "min_avg_conf": 0.4,
    "min_avg_area": 3000.0,
    "max_samples_per_track": 32,
    "crop_selection_strategy": "quality"
  },
  "reid": {
    "enable_cross_camera": false,
    "enable_strict_intra": true,
    "cross_threshold": 0.75,
    "intra_threshold": 0.78,
    "intra_max_gap": 30,
    "intra_max_overlap": 0,
    "use_mnn": true
  }
}
```

Ringkasan tracking:

- `local_tracks.csv`: 7677 rows.
- `valid_tracks_all.csv`: 6247 rows.
- `valid_track_summary.csv`: 21 tracks.
- `raw_rows`: 7677.
- `valid_rows`: 6247.
- `avg_conf_mean`: sekitar 0.6973.
- `avg_area_mean`: sekitar 35940.74.
- `id_switch_count`: 1.

Ringkasan Re-ID:

- `global_track_meta.csv`: 9 tracks.
- Jumlah Global ID: 8.
- `merged_pairs.csv`: 1 row.
- Merge reason: `intra_fragment_recovery` sebanyak 1.

Skor:

- Tracking score: sekitar 0.9394.
- Tracking precision: sekitar 0.9979.
- Tracking recall: sekitar 0.8745.
- Tracking F1: sekitar 0.9321.
- Re-ID score: 1.000.
- Expected Global ID: 8.
- Local tracks: 9.
- Global IDs: 8.
- Reduction required: 1.
- Reduction achieved: 1.
- Pairwise precision: 1.000.
- Pairwise recall: 1.000.
- Pairwise F1: 1.000.

Interpretasi:

Case 2 berhasil mencapai fragment recovery yang dibutuhkan. Tracking masih menghadapi crowded/occlusion, tetapi filter dan Re-ID menghasilkan satu merge intra-camera yang tepat. Dengan config final, `track_buffer=45` dipakai untuk menjaga track saat occlusion.

## Case 3 - Small/Distant Person Limitation

Run terakhir:

```text
outputs/case_3_failure_limitation/20260623_182403/
```

Ringkasan tracking:

- `local_tracks.csv`: 2429 rows.
- `valid_tracks_all.csv`: 2389 rows.
- `valid_track_summary.csv`: 14 tracks.
- `global_track_meta.csv`: 9 tracks.
- `raw_rows`: 2429.
- `valid_rows`: 2389.
- `fragmentation_avg`: 2.6.
- `id_switch_count`: 27.
- `mixed_track_count`: 3.

Ringkasan Re-ID:

- Jumlah Global ID: 6.
- `merged_pairs.csv`: 3 rows.
- Semua merge reason: `intra_fragment_recovery`.

Skor:

- Tracking score: sekitar 0.7982.
- Tracking precision: sekitar 0.9334.
- Tracking recall: sekitar 0.6561.
- Tracking F1: sekitar 0.7706.
- Re-ID score: 0.5625.
- Expected Global ID: 5.
- Local tracks: 9.
- Global IDs: 6.
- Reduction required: 4.
- Reduction achieved: 3.
- Pairwise precision: 1.000.
- Pairwise recall: 0.600.
- Pairwise F1: 0.750.
- False split rate: 0.400.

Interpretasi:

Case 3 adalah limitation case. Sistem berhasil melakukan beberapa fragment recovery tanpa false merge, tetapi recall association masih rendah. Banyak ID switch dan fragmentasi menunjukkan objek kecil/jauh masih sulit untuk detector, tracker, dan Re-ID.

## Case 4 - Multi-Camera Same-Time Re-ID

Run terakhir:

```text
outputs/case_4_multicamera_success/20260623_182939/
```

Ringkasan tracking:

- `local_tracks.csv`: 1306 rows.
- `valid_tracks_all.csv`: 1272 rows.
- `valid_track_summary.csv`: 11 tracks.
- `global_track_meta.csv`: 6 tracks.

Ringkasan Re-ID:

- Jumlah Global ID: 3.
- `merged_pairs.csv`: 3 rows.
- Merge reasons:
  - `intra_fragment_recovery`: 1.
  - `cross_camera_mnn`: 2.

Skor:

- Tracking score: sekitar 0.9485.
- Tracking precision: sekitar 0.9866.
- Tracking recall: sekitar 0.8474.
- Tracking F1: sekitar 0.9117.
- Re-ID score: 1.000.
- Expected Global ID: 3.
- Local tracks: 6.
- Global IDs: 3.
- Reduction required: 3.
- Reduction achieved: 3.
- Pairwise precision: 1.000.
- Pairwise recall: 1.000.
- Pairwise F1: 1.000.

Interpretasi:

Case 4 sukses untuk multi-camera association. Cross-camera MNN menggabungkan track lintas kamera dengan benar, sementara satu fragment intra-camera juga berhasil disambung.

## Case 5 - Multi-Camera 3-Video Stress Test

Run terakhir:

```text
outputs/case_5_multicamera_3_video_stress/20260623_183248/
```

Ringkasan tracking:

- `local_tracks.csv`: 11053 rows.
- `valid_tracks_all.csv`: 10068 rows.
- `valid_track_summary.csv`: 33 tracks.
- `global_track_meta.csv`: 17 tracks.

Ringkasan Re-ID:

- Jumlah Global ID: 11.
- `merged_pairs.csv`: 7 rows.
- Merge reasons:
  - `intra_fragment_recovery`: 4.
  - `cross_camera_mnn`: 3.

Skor:

- Tracking score: sekitar 0.9477.
- Tracking precision: sekitar 0.9988.
- Tracking recall: sekitar 0.8835.
- Tracking F1: sekitar 0.9376.
- Re-ID score: sekitar 0.9167.
- Expected Global ID: 8.
- Local tracks: 17.
- Global IDs: 11.
- Reduction required: 9.
- Reduction achieved: 6.
- Pairwise precision: 1.000.
- Pairwise recall: sekitar 0.5385.
- Pairwise F1: 0.700.
- False split rate: sekitar 0.4615.

Interpretasi:

Case 5 adalah stress test. Sistem konservatif: precision association tetap tinggi dan tidak banyak false merge, tetapi recall turun karena beberapa identitas yang seharusnya digabung masih terpisah. Ini terlihat dari false split rate yang cukup tinggi.

## Case 6 - Multi-Camera Temporal Handoff

Run terakhir:

```text
outputs/case_6_multicamera_temporal_handoff/20260623_183757/
```

Ringkasan tracking:

- `local_tracks.csv`: 441 rows.
- `valid_tracks_all.csv`: 409 rows.
- `valid_track_summary.csv`: 9 tracks.
- `global_track_meta.csv`: 6 tracks.

Ringkasan Re-ID:

- Jumlah Global ID: 3.
- `merged_pairs.csv`: 3 rows.
- Semua merge reason: `cross_camera_mnn`.

Skor:

- Tracking score: sekitar 0.9848.
- Tracking precision: sekitar 0.9927.
- Tracking recall: sekitar 0.9486.
- Tracking F1: sekitar 0.9701.
- Re-ID score: 1.000.
- Expected Global ID: 3.
- Local tracks: 6.
- Global IDs: 3.
- Reduction required: 3.
- Reduction achieved: 3.
- Pairwise precision: 1.000.
- Pairwise recall: 1.000.
- Pairwise F1: 1.000.

Interpretasi:

Case 6 sukses untuk temporal handoff. Track lintas kamera yang muncul pada waktu berbeda berhasil digabung dengan cross-camera MNN.

## Ringkasan Perbandingan Case

| Case | Local Tracks | Global IDs | Merged Pairs | Tracking Score | Re-ID Score | Catatan |
|---|---:|---:|---:|---:|---:|---|
| Case 1 | 3 | 3 | 0 | 1.000 | 1.000 | Baseline normal sukses |
| Case 2 | 9 | 8 | 1 | 0.939 | 1.000 | Fragment recovery sukses |
| Case 3 | 9 | 6 | 3 | 0.798 | 0.563 | Limitation, false split masih tinggi |
| Case 4 | 6 | 3 | 3 | 0.948 | 1.000 | Multi-camera sukses |
| Case 5 | 17 | 11 | 7 | 0.948 | 0.917 | Stress test, konservatif |
| Case 6 | 6 | 3 | 3 | 0.985 | 1.000 | Temporal handoff sukses |

## Kesimpulan Output

Case 1, 2, 4, dan 6 menunjukkan pipeline bekerja sesuai target. Case 3 sengaja menjadi limitation case untuk menunjukkan dampak orang kecil/jauh terhadap tracking dan Re-ID. Case 5 menunjukkan stress test multi-kamera: precision tetap tinggi, tetapi recall association masih dapat ditingkatkan jika ingin mengurangi false split.
