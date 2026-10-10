# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Trần Xuân Tùng
- MSSV: 2A202602787
- Email: Không
- Link repo (fork): https://github.com/TranXuanTung/K4-L2L3-DAY23-TranXuanTung-2A202602787-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 7959306139f9de2f061d48d07e8173fe703d9e2f

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9586, 0.4021, 1019/44/1515
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.346, 1000, 119.848, 0, 1534, 5.025
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.371, 1000, 137.753, 0, 1534, 5.025
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Ở Lidar mode, RMSE là 0.346, trong khi Fused mode lại tăng nhẹ lên 0.371. Điều này có thể xảy ra vì mô hình camera (sử dụng nhiễu ngẫu nhiên) đã thêm sự bất định (uncertainty) và làm lệch trạng thái dự đoán, dẫn đến tăng RMSE. Các thông số matches (1000), ghost (0), và miss (1534) giống hệt nhau do Lidar chịu trách nhiệm quyết định khởi tạo và xoá track (lifecycle), còn Camera chỉ đóng vai trò cập nhật (update) EKF.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - Lidar 3D đo trực tiếp không gian thực nên hàm quan sát là tuyến tính, `R` (3x3) biểu diễn độ không chắc chắn theo (x,y,z).
   - Camera 2D đo điểm ảnh qua phép chiếu phối cảnh Pinhole (phi tuyến) nên cần sử dụng Jacobian `H` để tuyến tính hoá tại trạng thái dự đoán. `R` (2x2) là phương sai nhiễu đo đạc theo pixel (u,v).
2. Vì sao cần gating Mahalanobis trước khi gán?
   - Mahalanobis tính khoảng cách cân nhắc sự tương quan và độ bất định (covariance của state và nhiễu sensor). Gating giúp loại bỏ sớm các cặp không khả thi, giảm số lượng cạnh trong bài toán gán (như Greedy Matching), tăng tốc độ tính toán và tránh gán sai.
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline là **track-then-fuse** (còn gọi là sensor-level tracking with sequential update). Lidar phát hiện mục tiêu để track, sau đó thông tin từ Lidar và Camera được lần lượt hợp nhất vào track đó (tuần tự qua các bước update). Trong log của `fusion-run-lab`, trong cùng 1 frame, thuật toán chạy `predict` sau đó chạy `update(lidar)` rồi `update(camera)`.
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Lệch calibration (như sai Intrinsic/Extrinsic) làm phép chiếu 3D sang 2D bị sai, nên giá trị dự đoán từ state (z_hat) không khớp với giá trị đo (z) trên ảnh. Điều này làm cho innovation/residual (z - z_hat) bị lệch hẳn về một phía (có bias lớn) thay vì dao động quanh 0, và dễ vượt ngưỡng gating dẫn tới bị từ chối liên tục.
5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Hàm update cần gọi ngay cả khi không có đo đạc để giảm điểm (score) của track nếu sensor đó là sensor quyết định (Lidar). Lidar có thông tin chính xác về độ sâu và tọa độ 3D, ít nhiễu ảo hơn Camera, do đó Lidar được tin tưởng để quyết định khởi tạo hoặc xoá track (quản lý lifecycle). Camera không thể đo chiều sâu trực tiếp và dễ bị nhiễu do ánh sáng, dùng Camera khởi tạo dễ sinh track ảo (ghost), nên nó chỉ đóng vai trò cập nhật bổ sung (update state).
6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - Xác nhận: Khi score vượt ngưỡng (>3), track chuyển thành CONFIRMED.
   - Giữ sau miss: Nếu miss, track bị trừ điểm. Miễn là score vẫn >0, track vẫn ở trạng thái CONFIRMED (được giữ qua vùng bị che khuất).
   - Xoá track: Nếu miss nhiều tới mức score <= 0, track sẽ bị xoá khỏi danh sách.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Agent Antigravity (Google)
- Dùng cho phần nào (hàm, câu hỏi, debug): Triển khai thuật toán Kalman, Camera Fusion, Track Management, Data Association và trả lời câu hỏi lý thuyết, chạy pipeline.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Pass 100% các testcase bằng lệnh pytest cho Part E-H, chạy thành công pipeline với dữ liệu Waymo.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
