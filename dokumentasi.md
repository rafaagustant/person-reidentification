# Dokumentasi Lengkap Person Re-ID Multi-Kamera

Dokumen ini menjelaskan cara memakai aplikasi, alur data, konfigurasi, tombol UI, file output, dan cara membaca hasil. Aplikasi dibuat untuk eksperimen person re-identification multi-kamera berbasis YOLO11n, BoT-SORT, dan OSNet.

## 1. Gambaran Sistem

Pipeline terdiri dari tiga tahap utama:

1. Tracking lokal per kamera
   - Video dibaca frame by frame.
   - YOLO11n mendeteksi objek kelas person.
   - BoT-SORT memberi `track_id` lokal untuk setiap kamera.
   - Crop orang disimpan untuk kandidat Re-ID.
   - Track disaring memakai threshold filter.

2. Re-ID dan association
   - Crop valid dipilih per track.
   - OSNet membuat embedding visual.
   - Similarity antar track dihitung dengan cosine similarity.
   - Track yang memenuhi aturan intra-camera atau cross-camera digabung menjadi Global ID.

3. Render dan ekspor hasil
   - Local track dan Global ID divisualisasikan ke video.
   - Tabel CSV, JSON, YAML, gallery, dan diagnostics disimpan di folder run.

Tahap Re-ID sengaja dipisah dari tracking. Mengubah config Re-ID tidak menjalankan ulang tracking, crop extraction, atau embedding yang sudah ada. Re-ID baru dihitung ulang setelah user menekan tombol `Jalankan Re-ID`.

## 2. Struktur Folder

```text
app.py                         entry point Streamlit
config/
  cases.py                     daftar demo case dan mapping video
  presets.py                   preset tracking, filter, Re-ID, dan rekomendasi case
  gt_cases.py                  mapping annotation ground truth
  scoring.py                   policy scoring operasional
core/
  tracking.py                  YOLO + BoT-SORT, filter track, tracker YAML
  reid.py                      ekstraksi embedding OSNet
  association.py               merge track dan Global ID
  evaluation.py                evaluasi tracking dan Re-ID berbasis GT
  diagnostics.py               ringkasan diagnostik
  pipeline.py                  orkestrasi tracking, Re-ID, render, snapshot config
  render.py                    render video hasil
  paths.py                     path asset, output, upload, weight
  models.py                    load YOLO dan OSNet
  gallery.py                   pembuatan gallery crop
ui/
  app.py                       UI utama Streamlit
  components.py                komponen tabel, gallery, download
  state.py                     session state, cleanup output, cache
  styles.py                    CSS ringan
utils/
  helpers.py                   helper filesystem dan JSON
assets/
  cases/                       video demo lokal
  weights/                     model weight lokal
  annotations/                 annotation ground truth opsional
outputs/                       hasil run lokal
uploads/                       upload lokal jika dipakai
```

## 3. Case Demo

### Case 1 - Single-Camera Normal Success

Satu kamera, kondisi normal, orang terlihat jelas. Targetnya local track tidak perlu digabung lintas kamera. Re-ID tetap bisa dijalankan untuk membuat Global ID, tetapi biasanya 1 local track menjadi 1 Global ID.

Use case:

- Baseline normal.
- Mengecek tracking precision.
- Mengecek bahwa pipeline tidak melakukan merge yang tidak perlu.

### Case 2 - Crowded / False Positive Sensitive

Satu kamera, kondisi ramai dan rawan fragmentasi. Target utamanya adalah mempertahankan track valid sambil mengurangi false positive. Strict intra-camera Re-ID aktif untuk menyambung fragmen track yang masih identitas sama.

Use case:

- Tuning `track_buffer`, `match_thresh`, dan threshold filter.
- Mengecek apakah fragment recovery menyambung track yang terputus.
- Menganalisis false positive dengan `valid_track_summary.csv`.

Recommended config final Case 2:

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

### Case 3 - Small/Distant Person Limitation

Satu kamera dengan orang kecil atau jauh. Case ini dipakai sebagai limitation case. Hasil tidak harus sempurna; yang penting sistem memperlihatkan keterbatasan secara jujur lewat fragmentation, ID switch, dan false split.

Use case:

- Menguji batas kemampuan detector/tracker.
- Membandingkan konfigurasi high recall.
- Melihat efek `imgsz`, `yolo_conf`, dan threshold filter yang lebih longgar.

### Case 4 - Multi-Camera Same-Time Re-ID

Dua kamera, orang yang sama bisa muncul di kamera berbeda. Cross-camera Re-ID aktif. Targetnya local tracks dari beberapa kamera digabung menjadi Global ID yang benar.

Use case:

- Validasi cross-camera association.
- Mengecek `cross_threshold` dan `use_mnn`.
- Mengecek gallery Global ID lintas kamera.

### Case 5 - Multi-Camera 3-Video Stress Test

Tiga kamera, lebih ramai, lebih banyak fragmentasi, dan lebih sulit. Case ini menguji kestabilan Re-ID multi-kamera pada kondisi stress.

Use case:

- Mengukur trade-off precision dan recall association.
- Melihat false split ketika sistem terlalu konservatif.
- Menilai robustness pipeline pada jumlah track lebih banyak.

### Case 6 - Multi-Camera Temporal Handoff

Dua kamera dengan perpindahan temporal. Cross-camera aktif, strict intra bisa dimatikan sesuai rekomendasi case. Targetnya orang yang muncul di kamera berbeda pada waktu berbeda tetap menjadi Global ID yang sama.

Use case:

- Handoff antar kamera.
- Menguji temporal gap lintas kamera.
- Mengecek cross-camera MNN ketika kemunculan tidak bersamaan.

## 4. Sidebar UI

### Status Perangkat

Menampilkan apakah CUDA tersedia. Jika CUDA tersedia, YOLO dan OSNet bisa memakai GPU. Jika tidak, pipeline tetap berjalan di CPU tetapi lebih lambat.

### Detail CUDA

Expander berisi detail device, nama GPU, dan status runtime. Gunakan ini saat ingin memastikan app benar-benar memakai GPU.

### Model

Menampilkan status YOLO11n dan OSNet. Jika model belum lengkap, letakkan weight di `assets/weights/`.

### Detail Model

Menampilkan path YOLO dan OSNet yang dibaca app.

### Reset Semua Case

Menghapus session state semua case dari Streamlit. Tombol ini tidak otomatis menghapus semua folder output kecuali flow cleanup lain dipakai.

Akibat klik:

- UI kembali seperti belum ada run.
- Config state case dibuat ulang dari rekomendasi saat case dibuka lagi.
- Output fisik lama masih ada kecuali dibersihkan lewat cleanup output.

### Simpan Run Terbaru per Case

Input angka untuk menentukan berapa run terbaru per case yang dipertahankan ketika menjalankan cleanup run lama.

### Bersihkan Run Lama

Aktif setelah checkbox konfirmasi dicentang. Menghapus folder run lama di `outputs/<case_id>/` dan menyisakan N run terbaru per case.

### Bersihkan Cache Streamlit

Aktif setelah checkbox konfirmasi dicentang. Membersihkan `st.cache_data`, `st.cache_resource`, dan session run state. Gunakan jika localhost terasa berat atau model/config tampak stale.

## 5. Panel Ringkasan Case

Setiap tab case menampilkan:

- Judul case.
- Deskripsi skenario.
- Expected result.
- Kamera yang terlibat.
- Metadata video dan annotation.
- Preview frame.

Jika video belum ada, status kamera akan menunjukkan video tidak tersedia dan tracking tidak bisa menghasilkan output.

## 6. Tracking + Filter Config

Config tracking dan filter berada sebelum tahap tracking. Untuk multi-kamera, setiap kamera memiliki expander config sendiri.

### Sumber Config

Pilihan umumnya:

- `Rekomendasi case`: memakai konfigurasi final yang disediakan di `config/presets.py`.
- Preset visual seperti normal, crowded, small/distant, occlusion, atau blur jika tersedia.
- Konfigurasi manual: terjadi otomatis ketika user mengubah nilai angka/selectbox dari nilai rekomendasi.

Jika user memilih ulang rekomendasi atau preset, widget akan diprime kembali dengan nilai preset tersebut dan hasil tracking lama akan dianggap tidak cocok.

### Parameter Tracking

`yolo_conf`

Threshold confidence deteksi YOLO. Makin rendah berarti lebih banyak deteksi masuk, tetapi risiko false positive naik.

`yolo_iou`

Threshold IoU NMS YOLO. Mengontrol penggabungan box yang overlap.

`imgsz`

Resolusi input YOLO. Nilai lebih besar membantu objek kecil, tetapi lebih berat.

`track_high_thresh`

Threshold deteksi high confidence untuk BoT-SORT.

`track_low_thresh`

Threshold deteksi low confidence untuk tahap matching tambahan.

`new_track_thresh`

Threshold minimal untuk membuat track baru.

`match_thresh`

Threshold matching track. Nilai lebih tinggi lebih ketat.

`track_buffer`

Jumlah frame track dipertahankan saat deteksi hilang. Nilai lebih besar membantu occlusion/fragmentasi tetapi dapat membuat track lama bertahan lebih panjang.

Catatan penting: app membuat `generated_tracker_config_used.yaml` dan memakainya pada `model.track(...)`. Detection args seperti `conf`, `iou`, dan `imgsz` masuk sebagai argumen `model.track`, sedangkan threshold BoT-SORT masuk ke YAML runtime.

### Parameter Filter

`min_frames`

Track valid jika `num_frames >= min_frames`.

`min_crops`

Track valid jika `num_crops >= min_crops`. Nilai 0 diperbolehkan untuk diagnosis track tanpa crop. Gunakan 0 hanya untuk analisis, karena Re-ID tetap membutuhkan crop/embedding agar bisa membandingkan identitas.

`min_avg_conf`

Track valid jika `avg_conf >= min_avg_conf`.

`min_avg_area`

Track valid jika `avg_area >= min_avg_area`.

`max_samples_per_track`

Jumlah maksimum crop yang dipakai untuk embedding per track.

`crop_selection_strategy`

`quality` memilih crop berdasarkan kualitas/confidence/area. `uniform` mengambil sampel lebih merata sepanjang track.

### Apa yang Terjadi Jika Config Tracking/Filter Berubah

Perubahan tracking/filter menginvalidasi:

- hasil tracking,
- hasil Re-ID,
- hasil render,
- score,
- run aktif di session.

Ini perlu karena tracking/filter menentukan isi `local_tracks.csv`, `valid_tracks_all.csv`, crop, dan embedding.

## 7. Tombol Tracking

### Jalankan Tracking

Dipakai pada single-camera case.

Akibat klik:

- Membuat folder run baru di `outputs/<case_id>/<timestamp>/`.
- Menulis `config_used.json`.
- Untuk setiap kamera, menulis `camera_config_used.json`.
- Membuat `generated_tracker_config_used.yaml`.
- Membuat `tracking_runtime_manifest.yaml`.
- Menjalankan YOLO + BoT-SORT.
- Menyimpan crop.
- Membuat `local_tracks.csv`.
- Membuat `local_tracks_valid.csv`.
- Membuat `valid_track_summary.csv`.
- Membuat gabungan `valid_tracks_all.csv`.
- Menghitung diagnostics dan score jika GT tersedia.

### Jalankan Tracking Semua Kamera

Dipakai pada multi-camera case. Efeknya sama, tetapi dilakukan untuk seluruh kamera dalam case. Re-ID baru terbuka setelah semua kamera memiliki valid track dan `valid_tracks_all.csv` tersedia.

### Jalankan Kamera Ini

Tombol per kamera di workspace kamera. Dipakai untuk menjalankan atau meninjau satu kamera. Untuk flow Re-ID multi-kamera, seluruh kamera tetap harus siap.

### Edit Config

Tombol bantuan yang mengarahkan user ke expander config kamera terkait.

## 8. Output Tahap Tracking di UI

Tab `Ringkasan tracking` menampilkan:

- jumlah raw rows,
- valid rows,
- valid tracks,
- runtime,
- tracking score,
- metrik precision/recall/F1 jika GT tersedia,
- status apakah Re-ID sudah siap.

Tab `Hasil per kamera` menampilkan ringkasan setiap kamera.

Tab `Galeri local track` menampilkan crop representatif local track.

Tab `Cakupan Ground Truth` menampilkan coverage GT jika annotation tersedia.

Tab `Diagnostik` menampilkan ID switch, fragmentation, mixed track, dan ringkasan lain.

Tab `Unduhan` menyediakan CSV/JSON/YAML detail.

## 9. Re-ID Config

Re-ID config berada di tahap 2 dan berlaku pada level case, bukan per kamera. Ini karena Global ID dibentuk dari seluruh valid track dalam satu case.

### Sumber Konfigurasi Re-ID

Dropdown saat ini menampilkan rekomendasi case sebagai sumber awal. Jika user mengubah nilai form dan menekan `Terapkan config Re-ID`, config berubah menjadi manual/custom secara internal:

- `custom["reid"] = true`
- `custom["reid_config_source"] = "Konfigurasi manual"`
- `custom["reid_config_dirty"] = true`

Nilai manual tidak boleh ditimpa rekomendasi case setelah diterapkan.

### Parameter Re-ID

`enable_cross_camera`

Jika true, track dari kamera berbeda boleh digabung jika memenuhi threshold dan aturan MNN.

`enable_strict_intra`

Jika true, track dari kamera yang sama boleh digabung sebagai fragment recovery dengan aturan temporal yang ketat.

`cross_threshold`

Cosine similarity minimum untuk merge lintas kamera.

`intra_threshold`

Cosine similarity minimum untuk merge dalam kamera yang sama.

`intra_max_gap`

Gap frame maksimum untuk merge intra-camera.

`intra_max_overlap`

Overlap frame maksimum yang masih diperbolehkan untuk merge intra-camera. Jika pair overlap 10 dan `intra_max_overlap=15`, pair tidak boleh ditolak karena overlap.

`use_mnn`

Mutual nearest neighbor. Jika true, pair harus saling menjadi kandidat terbaik agar lebih aman dari false merge.

## 10. Tombol Re-ID

### Reset Re-ID

Mengembalikan config Re-ID ke rekomendasi case.

Akibat klik:

- Widget Re-ID diprime ulang dari rekomendasi.
- Config aktif kembali ke rekomendasi.
- Hanya hasil Re-ID dan render lama di session yang dihapus.
- Tracking, crop, valid tracks, dan output tracking tidak dihapus.

### Terapkan Config Re-ID

Tombol submit form Re-ID.

Akibat klik:

- Nilai form menjadi config Re-ID aktif.
- Jika nilai berbeda dari rekomendasi, mode internal menjadi custom/manual.
- `runtime_config` diperbarui.
- Hanya output Re-ID lama dan render lama yang diinvalidasi.
- Tracking tidak dijalankan ulang.
- Crop dan embedding yang sudah ada boleh dipakai ulang.

### Jalankan Re-ID

Menjalankan tahap association.

Akibat klik:

- Membaca `valid_tracks_all.csv`.
- Memakai `track_embedding_df.pkl` dan `track_features.npy` jika sudah ada.
- Jika embedding belum ada, OSNet mengekstrak embedding dari crop valid.
- Menulis `reid_config_used.json`.
- Memperbarui `config_used.json` agar `reid`, `reid_config`, `reid_config_used`, dan `raw_config["reid"]` konsisten.
- Menghitung `pair_similarity.csv`.
- Menulis `merged_pairs.csv`.
- Menulis `global_track_meta.csv`.
- Menulis evaluation dan diagnostics.
- Menampilkan stage log seperti `Tracking reused`, `Embeddings reused`, dan `Re-ID recomputed`.

## 11. Render Video

### Render Video

Tersedia setelah Re-ID selesai.

Akibat klik:

- Membaca hasil Global ID.
- Membuat video render per kamera.
- Jika lebih dari satu kamera, membuat `combined_multicamera_rendered.mp4`.
- Menampilkan preview video dan tombol download.

Render dipisah dari Re-ID supaya user bisa mengevaluasi tabel lebih dulu tanpa langsung membuat file video besar.

## 12. Cleanup Output

Panel cleanup per case memiliki:

- `Bersihkan output run aktif`
- `Bersihkan semua output case ini`

Keduanya membutuhkan checkbox konfirmasi. Setelah dihapus, state case direset dan UI rerun.

Gunakan cleanup saat:

- output lama terlalu banyak,
- disk mulai penuh,
- ingin memastikan run berikutnya benar-benar baru,
- ingin membersihkan artefak sebelum push repo.

## 13. Cara Set Config

### Cara memakai rekomendasi case

1. Buka tab case.
2. Biarkan sumber config pada `Rekomendasi case`.
3. Jalankan tracking.
4. Terapkan Re-ID default jika tidak ada perubahan.
5. Jalankan Re-ID.

### Cara membuat config manual tracking/filter

1. Buka expander Tracking + Filter Config.
2. Ubah parameter tracking atau filter.
3. Status config berubah menjadi manual.
4. Jalankan tracking ulang.

Perubahan tracking/filter harus menjalankan ulang tracking karena file local track dan crop berubah.

### Cara membuat config manual Re-ID

1. Jalankan tracking sampai valid tracks tersedia.
2. Buka panel Re-ID.
3. Ubah parameter Re-ID di form.
4. Klik `Terapkan config Re-ID`.
5. Klik `Jalankan Re-ID`.

Perubahan Re-ID tidak menjalankan ulang tracking.

### Cara diagnosis track tanpa crop

Set `min_crops=0` pada filter. Logika filter tetap:

```text
num_crops >= min_crops
```

Dengan `min_crops=0`, track dengan `num_crops=0` bisa valid jika syarat lain lolos. Gunakan ini untuk diagnosis tracking/filter. Untuk Re-ID, track tanpa crop tidak memiliki embedding visual yang kuat, jadi hasil association bisa terbatas.

## 14. Cara Membaca Config Runtime

### `config_used.json`

Snapshot config level run. Berisi:

- tracking config global,
- filter config global,
- reid config aktif,
- camera configs,
- raw config,
- custom flags,
- path tracker YAML,
- fingerprint tracking.

Gunakan ini untuk menjawab pertanyaan: "config apa yang dipakai run ini?"

### `camera_config_used.json`

Snapshot config per kamera. Berisi tracking dan filter aktual untuk kamera tertentu. Gunakan ini pada multi-kamera karena setiap kamera bisa memiliki threshold berbeda.

### `generated_tracker_config_used.yaml`

YAML BoT-SORT aktual yang diberikan ke `model.track(...)`. File ini harus berisi parameter:

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `match_thresh`
- `track_buffer`

Komentar header juga mencatat `yolo_conf`, `yolo_iou`, dan `imgsz`.

### `tracking_runtime_manifest.yaml`

Manifest runtime tracking. Berisi:

- tracking fingerprint,
- path tracker YAML,
- argumen `model.track`: `conf`, `iou`, `imgsz`, `classes`, `persist`,
- tracking config lengkap.

Gunakan ini untuk memastikan UI benar-benar masuk ke runtime, bukan hanya tersimpan di JSON.

### `reid_config_used.json`

Snapshot Re-ID aktual yang dipakai backend saat `Jalankan Re-ID`. Jika UI memakai:

```json
{
  "use_mnn": true,
  "intra_threshold": 0.66,
  "intra_max_overlap": 15
}
```

maka file ini harus menampilkan nilai yang sama.

## 15. File Output Utama

`local_tracks.csv`

Semua row hasil tracking lokal, termasuk frame, bbox, confidence, area, track id, track key, dan crop path.

`local_tracks_valid.csv`

Subset row dari track valid setelah filter.

`valid_tracks_all.csv`

Gabungan valid tracks semua kamera. Ini input utama Re-ID.

`valid_track_summary.csv`

Satu row per track. Kolom penting:

- `num_frames`
- `num_crops`
- `avg_conf`
- `avg_area`
- `min_frames_used`
- `min_crops_used`
- `min_avg_conf_used`
- `min_avg_area_used`
- `pass_min_frames`
- `pass_min_crops`
- `pass_min_avg_conf`
- `pass_min_avg_area`
- `is_valid`
- `status`
- `filter_reason`

Jika track filtered, baca `filter_reason` untuk tahu threshold mana yang gagal.

`track_embedding_df.pkl` dan `track_features.npy`

Cache embedding per track. Re-ID bisa reuse file ini supaya perubahan config Re-ID tidak mengekstrak ulang embedding.

`pair_similarity.csv`

Semua kandidat pair track dengan similarity, temporal gap, temporal overlap, status merge, reason merge, dan reject reason.

`merged_pairs.csv`

Pair yang benar-benar digabung menjadi Global ID.

`global_track_meta.csv`

Metadata Global ID: anggota local track, kamera, frame range, jumlah track, dominant GT, purity, dan flag mixed jika tersedia.

`reid_pairwise_evaluation.csv`

Metrik precision, recall, F1, false merge, dan false split berbasis GT.

`reid_pairwise_detail.csv`

Detail evaluasi setiap pair.

`tracking_diagnostics.csv`

Diagnostik tracking seperti raw rows, valid rows, confidence/area, ID switch, fragmentation, dan mixed track jika GT tersedia.

`reid_diagnostics.csv`

Diagnostik Re-ID seperti jumlah pair, jumlah merge, global ids, dan kualitas association.

`flow1_tracking_score.csv`

Score operasional tahap tracking.

`flow2_global_id_score.csv`

Score operasional tahap Re-ID/Global ID.

`final_pipeline_score.csv`

Gabungan score tracking dan Re-ID.

`*_rendered.mp4` dan `combined_multicamera_rendered.mp4`

Video hasil render.

## 16. Cara Membaca Hasil Filtering

Contoh logika:

```text
num_frames >= min_frames
num_crops >= min_crops
avg_conf >= min_avg_conf
avg_area >= min_avg_area
```

Track valid jika semua pass bernilai true.

Contoh:

```text
num_frames = 777
num_crops = 0
avg_conf = 0.386
avg_area = 37539
min_frames = 180
min_crops = 0
min_avg_conf = 0.36
min_avg_area = 3000
```

Hasil yang benar:

```text
pass_min_frames = true
pass_min_crops = true
pass_min_avg_conf = true
pass_min_avg_area = true
is_valid = true
```

Jika masih filtered, cek `min_*_used` di `valid_track_summary.csv` untuk melihat threshold aktual yang dipakai.

## 17. Cara Membaca Hasil Re-ID

Mulai dari `pair_similarity.csv`:

- `cosine_similarity` tinggi berarti visual mirip.
- `same_camera=true` berarti pair intra-camera.
- `temporal_gap` adalah jarak frame antar track.
- `temporal_overlap` adalah jumlah overlap waktu.
- `merge_status=true` berarti pair digabung.
- `merge_reason` menjelaskan alasan merge.
- `reject_reason` menjelaskan alasan reject.

Jika pair dengan overlap 10 ditolak sebagai `rejected_temporal_overlap`, cek `reid_config_used.json`. Jika `intra_max_overlap=15`, pair tersebut tidak boleh ditolak karena overlap.

## 18. Flow yang Bisa Dilakukan

### Flow A - Baseline rekomendasi case

1. Buka case.
2. Jangan ubah config.
3. Jalankan tracking.
4. Klik `Terapkan config Re-ID` jika ingin memastikan snapshot Re-ID aktif.
5. Jalankan Re-ID.
6. Render video.
7. Download score dan diagnostics.

### Flow B - Tuning tracking

1. Ubah `yolo_conf`, `track_buffer`, atau `match_thresh`.
2. Jalankan tracking ulang.
3. Bandingkan `tracking_fingerprint`, `generated_tracker_config_used.yaml`, dan metrics.
4. Jalankan Re-ID ulang jika valid track berubah.

### Flow C - Tuning filter

1. Ubah `min_frames`, `min_crops`, `min_avg_conf`, atau `min_avg_area`.
2. Jalankan tracking ulang.
3. Baca `valid_track_summary.csv`.
4. Cek `filter_reason` untuk track yang filtered.

### Flow D - Tuning Re-ID tanpa tracking ulang

1. Tracking sudah selesai.
2. Ubah `intra_threshold`, `cross_threshold`, `use_mnn`, `intra_max_gap`, atau `intra_max_overlap`.
3. Klik `Terapkan config Re-ID`.
4. Klik `Jalankan Re-ID`.
5. Baca stage log. Seharusnya muncul `Tracking reused`, `Embeddings reused`, dan `Re-ID recomputed`.

### Flow E - Diagnosis runtime tracker YAML

1. Ubah `track_buffer` atau `match_thresh`.
2. Jalankan tracking.
3. Buka `generated_tracker_config_used.yaml`.
4. Pastikan nilai YAML sama dengan UI.
5. Buka `tracking_runtime_manifest.yaml`.
6. Pastikan `model_track_args` memuat `conf`, `iou`, dan `imgsz` yang benar.

### Flow F - Cleanup sebelum eksperimen baru

1. Gunakan cleanup per case untuk menghapus output case tertentu.
2. Gunakan sidebar cleanup untuk menghapus run lama.
3. Gunakan cache cleanup jika state UI terasa stale.

## 19. Troubleshooting

### Model belum tersedia

Pastikan file weight ada di `assets/weights/`. YOLO dapat fallback ke nama model Ultralytics jika tersedia, tetapi untuk reproducibility letakkan `yolo11n.pt` secara lokal.

### CUDA tidak tersedia di UI

Jika sidebar menampilkan `torch_version: ...+cpu`, artinya environment Streamlit sedang memakai PyTorch CPU-only. Ini biasanya terjadi karena `torch` terinstall dari PyPI default, bukan dari wheel CUDA PyTorch.

Solusi cepat dari environment aktif:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --timeout 1000 --retries 20 --prefer-binary -r requirements-cuda.txt
python check_cuda.py
```

Target validasi:

- `torch.__version__` mengandung `+cu126`.
- `torch.version.cuda` bernilai `12.6`.
- `torch.cuda.is_available()` bernilai `True`.
- `torch.cuda.get_device_name(0)` menampilkan GPU NVIDIA.

Untuk setup bersih Windows + NVIDIA GPU:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows_cuda.ps1
python -m streamlit run app.py
```

Gunakan `python -m streamlit run app.py` agar Streamlit memakai Python dari environment aktif. Jika memakai command global `streamlit run app.py`, UI bisa berjalan dari environment lain dan kembali membaca torch CPU-only.

### Video tidak tersedia

Pastikan file `.mp4` berada di path case yang benar. Lihat `config/cases.py`.

### Tracking tidak berubah walau config berubah

Cek:

- `config_used.json`
- `camera_config_used.json`
- `generated_tracker_config_used.yaml`
- `tracking_runtime_manifest.yaml`
- `tracking_fingerprint`

Jika fingerprint berbeda, run dianggap berbeda.

### Track valid yang seharusnya lolos tetap filtered

Cek `valid_track_summary.csv`, terutama:

- `min_*_used`
- `pass_*`
- `filter_reason`

Threshold aktual ada di kolom `min_*_used`, bukan hanya di UI.

### Re-ID memakai config lama

Cek:

- `reid_config_used.json`
- `config_used.json` bagian `reid`
- `config_used.json` bagian `reid_config`
- `config_used.json` bagian `reid_config_used`
- `config_used.json` bagian `raw_config.reid`

Semua harus konsisten.

### Mengubah Re-ID membuat app terasa rerun

Parameter Re-ID berada di `st.form`. Perubahan nilai tidak menjalankan stage berat sampai `Terapkan config Re-ID` dan `Jalankan Re-ID` ditekan.

### Re-ID tidak bisa dijalankan

Pastikan tracking selesai dan `valid_tracks_all.csv` ada. Untuk multi-kamera, semua kamera harus memiliki valid tracks.

## 20. Catatan GitHub

File berikut tidak dikomit:

- virtual environment,
- cache Python,
- output run,
- upload lokal,
- model weights,
- video demo besar,
- temporary logs.

Setelah clone, user perlu menyiapkan model dan video lokal sesuai README. Source code, config, dan dokumentasi tetap cukup untuk memahami dan menjalankan pipeline.
