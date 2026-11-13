# Ghi chú thay đổi recorder — 2026-11-13

File sửa: `data_recording/record_teleop_data.py`.

## Cách dùng

Chạy từ thư mục gốc SmolVLA-Lite-Lab trong môi trường có LeRobot và OpenCV:

```bash
python -B -m data_recording.record_teleop_data --data-dir pick_place_smolvla_v1
```

Click vào cửa sổ dashboard để nhận phím. Bấm `t`, gõ instruction tiếng Anh/không dấu,
`Enter` xác nhận, `Esc` hủy chỉnh sửa, `Backspace` xóa ký tự. Tối đa 200 ký tự ASCII.
Task hiện tại được đưa vào ô nhập để có thể sửa tiếp bằng Backspace.

Bấm `r` để bắt đầu ghi, `s` để lưu, `x` để bỏ, `q` để thoát (lưu episode đang ghi).
Task phải được xác nhận và không rỗng trước khi ghi. Chỉ đổi task khi chưa recording.
Trong chế độ nhập, các chữ `r/s/x/q/t` đều được xem là nội dung, không kích hoạt phím tắt.
Teleoperation vẫn tiếp tục chạy trong lúc nhập task.

Có thể đặt task ban đầu bằng `--task "Pick up the red cube"`, sau đó sửa trên dashboard.

## Những đoạn đã thay đổi

- **Import / khởi tạo phần cứng:** dùng `low_control.hardware_constant` và
  `make_camera_configs()` để chuyển dict thành camera config của LeRobot.
  Đặt rõ `use_degrees=True` cho cả hai tay. Thêm `disconnect_arm()` tại recorder
  vì repo không có module `hardware_utils` mà bản copy ACT đang import.
- **`TaskEditor` mới:** quản lý task đã xác nhận và bản nháp; xử lý nhập/xóa/xác nhận/hủy,
  giới hạn độ dài và khóa đổi task trong lúc recording.
- **`DashboardRenderer.render()`:** thêm vùng instruction, trạng thái chỉnh sửa và
  hướng dẫn phím; tăng chiều cao cửa sổ từ 720 lên 840 để giữ chỗ cho các panel cũ.
- **`main()`:** thêm `--task`, chuyển tên dataset mặc định thành `pick_place_smolvla_v1`,
  sửa đường dẫn trong help thành `data_recording/data/`. Dùng `waitKeyEx()` và xử lý
  nhập task trước phím tắt; chặn ghi nếu chưa có task; truyền task vào writer khi bấm `r`.
- **`EpisodeWriter.__init__()` / `save()`:** chốt task riêng cho episode và ghi
  `metadata.json` khi lưu. Metadata gồm task, robot type, control Hz mục tiêu,
  video FPS, camera names, joint names, action type và số frame.
  `data.npz` và MP4 giữ schema cũ. Episode rỗng/bị discard không được giữ lại.

## Đầu ra

```text
data_recording/data/pick_place_smolvla_v1/episode_0000/
├── data.npz
├── front.mp4
├── wrist.mp4
└── metadata.json
```

**Giới hạn ở phần training:** loader hiện tại vẫn dùng một `--task` chung cho toàn bộ
dataset, chưa đọc task trong `metadata.json`. Muốn train nhiều task cùng lúc cần sửa
tiếp `EpisodicDataset` và luồng truyền task trong trainer. Thay đổi lần này chỉ ở phần thu data.

## Kiểm tra

Đã kiểm tra cú pháp và chạy luồng main bằng robot mô phỏng, với OpenCV ghi/đọc video thật:
nhập task chứa phím tắt, xác nhận/hủy/xóa, chặn task rỗng, giới hạn độ dài, khóa đổi task
khi ghi, lưu hai episode có task khác nhau, khớp số frame NPZ/MP4, xóa episode rỗng/discard.
Đã render và xem dashboard ở trạng thái nhập và recording. Chưa kiểm tra robot/camera
thật hoặc bàn phím cửa sổ OpenCV thật; môi trường kiểm tra hiện chưa có LeRobot.
