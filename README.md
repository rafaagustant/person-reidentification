# Person Re-ID Multi-Kamera

Aplikasi Streamlit untuk demo pipeline person re-identification multi-kamera. Pipeline memakai YOLO11n untuk deteksi orang, BoT-SORT dari Ultralytics untuk local tracking, OSNet untuk embedding Re-ID, lalu association untuk membentuk Global ID.

Dokumentasi lengkap ada di [dokumentasi.md](dokumentasi.md). Ringkasan hasil run terakhir per case ada di [output.md](output.md).

## Fitur Utama

- 6 demo case: single-camera normal, crowded fragmentation, limitation case, multi-camera success, 3-video stress test, dan temporal handoff.
- Config tracking dan filter dapat diatur per kamera.
- Config Re-ID diatur pada level case dan diterapkan lewat form supaya perubahan parameter Re-ID tidak menjalankan ulang tracking.
- Snapshot config runtime disimpan ke `config_used.json`, `camera_config_used.json`, `generated_tracker_config_used.yaml`, `tracking_runtime_manifest.yaml`, dan `reid_config_used.json`.
- Output diagnostik filter track menampilkan threshold aktual, pass/fail tiap syarat, dan `filter_reason`.
- Output Re-ID menampilkan pair similarity, merged pairs, global ID metadata, pairwise evaluation, gallery, dan render video.

## Struktur Repo

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- dokumentasi.md
|-- output.md
|-- assets/
|   |-- cases/
|   |-- weights/
|   `-- annotations/
|-- config/
|-- core/
|-- ui/
|-- utils/
|-- outputs/
`-- uploads/
```

`outputs/`, `uploads/`, video demo, model weight, virtual environment, dan cache Python tidak dipush ke GitHub. Folder placeholder tetap disimpan dengan `.gitkeep` atau README kecil.

## Instalasi

Gunakan Python 3.10 atau 3.11.

### Windows + NVIDIA GPU

Untuk laptop/PC NVIDIA, gunakan installer CUDA PyTorch khusus agar `torch` tidak terpasang sebagai CPU-only build dari PyPI default.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows_cuda.ps1
python -m streamlit run app.py
```

Jalankan Streamlit dengan `python -m streamlit run app.py`, bukan langsung `streamlit run app.py`, supaya executable yang dipakai pasti berasal dari environment Python aktif.

Validasi CUDA:

```powershell
python check_cuda.py
```

Target output:

- `torch.__version__` mengandung `+cu126`.
- `torch.version.cuda` bernilai `12.6`.
- `torch.cuda.is_available()` bernilai `True`.
- Device name menampilkan GPU NVIDIA, misalnya `NVIDIA GeForce RTX 4060 Laptop GPU`.

Jika koneksi putus saat download wheel PyTorch CUDA yang besar, ulangi dari venv aktif:

```powershell
python -m pip install --timeout 1000 --retries 20 --prefer-binary -r requirements-cuda.txt
python check_cuda.py
```

Jika `torch.__version__` masih mengandung `+cpu`, uninstall build CPU dulu:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --timeout 1000 --retries 20 --prefer-binary -r requirements-cuda.txt
python check_cuda.py
```

### CPU atau Environment Manual

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` sengaja tidak berisi `torch` dan `torchvision`, supaya PyTorch CPU-only tidak terpasang diam-diam dari PyPI default. Untuk GPU, install PyTorch dari `requirements-cuda.txt`.

## Model dan Dataset

Letakkan file model di:

```text
assets/weights/yolo11n.pt
assets/weights/osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth
```

Letakkan video demo sesuai case:

```text
assets/cases/case_1_normal_success/camera_2.mp4
assets/cases/case_2_crowded_fragmentation/camera_3.mp4
assets/cases/case_3_failure_limitation/camera_1.mp4
assets/cases/case_4_multicamera_success/camera_1.mp4
assets/cases/case_4_multicamera_success/camera_3.mp4
assets/cases/case_5_multicamera_3_video_stress/camera_1.mp4
assets/cases/case_5_multicamera_3_video_stress/camera_2.mp4
assets/cases/case_5_multicamera_3_video_stress/camera_3.mp4
assets/cases/case_6_multicamera_temporal_handoff/camera_6.mp4
assets/cases/case_6_multicamera_temporal_handoff/camera_5.mp4
```

## Menjalankan App

```bash
python -m streamlit run app.py
```

Flow umum:

1. Pilih tab case.
2. Cek status video, annotation, CUDA, dan model.
3. Atur Tracking + Filter Config jika perlu.
4. Klik `Jalankan tracking` atau `Jalankan tracking semua kamera`.
5. Atur Re-ID config di form, lalu klik `Terapkan config Re-ID`.
6. Klik `Jalankan Re-ID`.
7. Klik `Render video` jika ingin output video final.

## Case 2 Recommended Config

Recommended config final Case 2 saat ini:

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

## GitHub Notes

File besar dan output lokal di-ignore:

- `.venv/`
- `__pycache__/`
- `outputs/`
- `uploads/`
- `assets/weights/*.pt`
- `assets/weights/*.pth`
- `assets/cases/**/*.mp4`
- file cache dan temporary lain

Repo ini siap dipush sebagai source code. Model, video, dan output run dibuat atau ditempatkan secara lokal setelah clone.
