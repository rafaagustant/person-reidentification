# Person Re-Identification Multi-Kamera

Aplikasi Streamlit untuk demonstrasi person re-identification pada satu atau
beberapa kamera. Sistem membentuk local track per kamera, menyaring track yang
tidak memenuhi kualitas minimum, lalu mengasosiasikan track menjadi Global ID.

## Arsitektur

```text
YOLO11n
  → BoT-SORT
  → filtering local track
  → OSNet crop embedding
  → mean track-level embedding
  → cosine similarity
  → intra/cross-camera association
  → Global ID
  → evaluasi ground truth
  → tabel, gallery, dan video
```

- YOLO menerima `yolo_conf`, `yolo_iou`, dan `imgsz` sebagai argumen langsung
  `model.track()`.
- Threshold BoT-SORT ditulis ke YAML tracker yang dibuat untuk setiap kamera.
- Tracking dan filter dapat berbeda untuk setiap kamera.
- Association/Re-ID berlaku pada tingkat case.
- Perubahan Re-ID mempertahankan hasil tracking. Perubahan tracking/filter
  hanya membuat kamera yang konfigurasinya berubah menjadi stale.

## Struktur Repository

```text
.
├── app.py                 # entry point Streamlit
├── config/
│   ├── cases.py           # enam case, urutan kamera, dan path video
│   ├── gt_cases.py        # metadata annotation dan source frame
│   └── presets.py         # default, rekomendasi, normalisasi, validasi
├── core/
│   ├── pipeline.py        # orkestrasi tracking, Re-ID, evaluasi, render
│   ├── tracking.py        # YOLO, BoT-SORT, crop, filter, track count
│   ├── reid.py            # sampling dan embedding OSNet
│   ├── association.py     # similarity, MNN, temporal rule, Global ID
│   ├── evaluation.py      # parser GT dan metrik
│   ├── gallery.py
│   ├── render.py
│   ├── models.py
│   └── paths.py
├── ui/
│   ├── app.py
│   ├── config_panel.py
│   ├── state.py
│   ├── result_adapter.py
│   ├── results_panel.py
│   └── components.py
├── utils/
├── tests/
├── assets/
│   ├── cases/
│   ├── annotations/
│   └── weights/
├── outputs/               # hasil runtime; di-ignore
└── uploads/               # file runtime; di-ignore
```

Script pencarian konfigurasi satu kali tidak menjadi bagian runtime dan tidak
menjadi sumber preset. Satu-satunya sumber rekomendasi aplikasi adalah
`config/presets.py::CASE_RECOMMENDATIONS`.

## Versi Python dan Instalasi

Gunakan Python 3.10 atau 3.11.

### Windows dan NVIDIA CUDA

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows_cuda.ps1
python check_cuda.py
```

Script tersebut membuat atau menggunakan `.venv`, memasang dependency runtime,
kemudian memasang PyTorch/torchvision CUDA 12.6 dari `requirements-cuda.txt`.

### Environment manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` tidak memasang PyTorch secara langsung. Pasang build PyTorch
yang sesuai perangkat, lalu pasang dependency pengujian:

```powershell
python -m pip install -r requirements-dev.txt
```

## Model Weight

Letakkan bobot berikut di `assets/weights/`:

```text
yolo11n.pt
osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth
```

Bobot model di-ignore oleh Git. Jika bobot YOLO lokal tidak ada, Ultralytics
dapat mencoba mengambil `yolo11n.pt`; bobot OSNet wajib tersedia secara lokal.

## Enam Case

| Case | Kamera |
|---|---|
| `case_1_normal_success` | `camera_2` |
| `case_2_crowded_fragmentation` | `camera_3` |
| `case_3_failure_limitation` | `camera_1` |
| `case_4_multicamera_success` | `camera_1`, `camera_3` |
| `case_5_multicamera_3_video_stress` | `camera_1`, `camera_2`, `camera_3` |
| `case_6_multicamera_temporal_handoff` | `camera_6`, `camera_5` |

Definisi case dan path video berada di `config/cases.py`. Metadata annotation
serta rentang source frame berada di `config/gt_cases.py`.

## Mode Konfigurasi

UI menyediakan dua mode:

1. **Konfigurasi Rekomendasi**
   - read-only;
   - dibangun oleh `build_recommended_config()`;
   - langsung menjadi konfigurasi runtime aktif.
2. **Konfigurasi Awal Analisis**
   - dapat diedit melalui form;
   - perubahan dapat diterapkan ke semua kamera atau satu kamera;
   - nilai dinormalisasi dan divalidasi sebelum digunakan.

Tracking/filter disimpan dalam `camera_configs.<camera>`. Re-ID disimpan pada
bagian `reid` tingkat case. Nilai top-level tracking/filter hanya menjadi
representasi umum dan fallback jika override kamera tidak tersedia.

## Menjalankan Aplikasi

```powershell
python -m streamlit run app.py
```

Alur UI:

1. Buka tab case.
2. Pilih mode konfigurasi.
3. Jalankan tracking untuk satu kamera atau semua kamera.
4. Pastikan seluruh kamera berstatus siap.
5. Jalankan Re-ID dan pembentukan Global ID.
6. Jalankan render jika video hasil diperlukan.

Tab Ringkasan membedakan:

- **Raw Track**: pasangan unik `(camera, track_id)` sebelum filtering.
- **Valid Track**: pasangan unik yang lolos seluruh filter.
- **Track Terfilter**: `Raw Track - Valid Track`.
- **False Merge Rate** dan **False Split Rate**: rate 0–1 dari evaluasi
  pairwise, bukan jumlah error mentah.

Jika artefak lama tidak menyediakan data yang cukup untuk menghitung suatu
nilai, UI menampilkan `–` dan tidak menyalin count lain sebagai pengganti.

## Output Runtime

Setiap run berada di `outputs/<case_id>/<run_id>/`. Artefak utama:

```text
config_used.json
run_manifest.json
track_count_summary_by_camera.csv
local_tracks.csv
valid_tracks_all.csv
valid_track_summary.csv
tracking_standard_metrics.csv
pair_similarity.csv
merged_pairs.csv
global_track_meta.csv
reid_pairwise_evaluation.csv
reid_config_used.json
embedding_manifest.json
gallery_local/
gallery_global/
merge_gallery/
*_rendered.mp4
```

Setiap folder kamera juga menyimpan:

```text
camera_config_used.json
generated_tracker_config_used.yaml
tracking_runtime_manifest.yaml
local_tracks.csv
local_tracks_valid.csv
valid_track_summary.csv
```

`run_manifest.json` baru menggunakan `output_schema_version: 2` dan menyimpan
`raw_track_count`, `valid_track_count`, serta `filtered_track_count`.

## Pengujian

```powershell
python -m compileall app.py config core ui utils
python -c "import app; import ui.app; import core.pipeline; import config.presets"
python -m pytest -q --basetemp=.pytest_tmp
```

Test mencakup schema/preset konfigurasi, override kamera, tracker YAML, direct
YOLO args, filtering, track counts, association, pairwise rates, cache,
manifest compatibility, state invalidation, adapter UI, dan Streamlit AppTest.

## File Lokal dan Git

`.gitignore` mengecualikan virtual environment, cache Python/pytest, model
weight, video demo, output run, embedding cache, log, serta file IDE. Folder
placeholder tetap disimpan melalui `.gitkeep` atau README kecil.

Sebelum menjalankan aplikasi setelah clone, sediakan video case dan bobot model
secara lokal sesuai path di atas.
