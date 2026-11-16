# SmolVLA-Lite-Lab

SmolVLA tự triển khai cho SO-ARM101: **chuẩn bị dữ liệu → chạy thử 1 epoch → train chính → đánh giá offline → rollout**.

Lệnh chính: `python -B -m smolvla_scratch.train_infer`. Chạy tất cả lệnh bên dưới từ thư mục gốc repo.

Train mới sẽ tải backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, đóng băng VLM và train action expert cùng các lớp projection. Expert được khởi tạo mới. Repo chưa hỗ trợ nạp trực tiếp policy `lerobot/smolvla_base`; `--resume` và `--finetune` dùng checkpoint do repo này tạo.

## 1. Chuẩn bị môi trường

Lần đầu, tạo môi trường Python 3.12 và cài dependency:

```bash
cd /Users/tt/Documents/GitHub/SmolVLA-Lite-Lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Requirements hiện ghim PyTorch `2.11.0`, Transformers `5.22.2` và LeRobot `0.6.1`. Lần train đầu cần mạng để tải backbone/config/tokenizer; những lần sau có thể dùng cache.

Mỗi lần mở terminal mới, chỉ cần:

```bash
cd /Users/tt/Documents/GitHub/SmolVLA-Lite-Lab
source .venv/bin/activate
```

Kiểm tra môi trường và danh sách tham số, chưa train hoặc kết nối robot:

```bash
python -c 'import sys, torch, lerobot; print(sys.executable); print(torch.__version__); print(lerobot.__file__)'
python -B -m smolvla_scratch.train_infer --help
```

Nếu dùng env cũ có file `smolvla_local_sources.pth`, bỏ liên kết source cũ để dùng LeRobot cài trong env:

```bash
rm -f .venv/lib/python3.12/site-packages/smolvla_local_sources.pth
```

**Chỉ khi cần dashboard thu dữ liệu:** dashboard dùng `cv2.imshow`, cần OpenCV có GUI. Kiểm tra các biến thể đang cài:

```bash
python -m pip list | python -c 'import sys; print("".join(line for line in sys.stdin if "opencv" in line.lower()))'
```

Nếu cả `opencv-python` và `opencv-python-headless` cùng có mặt, cài lại một biến thể có GUI:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install 'opencv-python>=4.10,<4.14'
```

LeRobot yêu cầu bản headless trong metadata nên `python -m pip check` có thể báo thiếu gói đó sau khi thay bằng bản GUI. Cài lại requirements có thể đưa bản headless trở lại.

## 2. Chuẩn bị dataset

Nếu đã có dữ liệu đúng định dạng, chuyển sang bước 3. **Train và đánh giá offline không cần cắm robot.**

Muốn thu dữ liệu mới, đặt cổng USB, calibration ID và camera trong [`low_control/hardware_constant.py`](low_control/hardware_constant.py) theo phần cứng đã setup/calibrate, rồi chạy:

```bash
python -B -m data_recording.record_teleop_data \
  --data-dir pick_place_smolvla_v1 \
  --task "Pick up the block"
```

Trong cửa sổ dashboard: `t` sửa task tiếng Anh → Enter xác nhận → `r` bắt đầu ghi → `s` lưu episode. `x` bỏ episode, `q` thoát. Thu một lần thực hiện task hoàn chỉnh cho mỗi episode; kiểm tra cả hai camera và kết quả thao tác trước khi đưa vào train.

Dataset mẫu dùng xuyên suốt README:

```text
data_recording/data/pick_place_smolvla_v1/
├── episode_0000/
│   ├── data.npz
│   ├── front.mp4
│   ├── wrist.mp4
│   └── metadata.json
├── episode_0001/
│   └── ...
└── ...
```

Yêu cầu loader:

- Ít nhất **2 episode** để tách train/validation. Đây là mức tối thiểu để chạy code, chưa đủ để kết luận chất lượng policy.
- `data.npz` có `states`, `actions` dạng `(T, 6)` và `joint_names`; `timestamps`, nếu có, phải tăng nghiêm ngặt và có `T` phần tử.
- Mỗi camera có đúng `T` frame; độ phân giải của từng camera phải nhất quán giữa các episode và hai split.
- `metadata.json` có instruction không rỗng, ví dụ `{"task": "Pick up the block"}`. Mặc định trainer đọc task riêng của từng episode.

Thứ tự 6 khớp chuẩn:

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
```

Loader sắp lại cột theo `joint_names`. Năm khớp thân dùng độ; gripper dùng thang riêng của LeRobot. Giữ cùng calibration, đơn vị đo và cách bố trí camera khi thu và rollout.

Nếu dữ liệu cũ thiếu task, thêm `--task "Pick up the block"` vào lệnh train. Option này **ghi đè instruction của mọi episode**, nên chỉ dùng khi toàn bộ dataset cùng task. Chỉ thêm `--legacy_joint_order` cho recording ACT cũ thiếu `joint_names` và thực sự lưu cột theo `sorted(joint_keys)`.

## 3. Chạy thử 1 epoch

Kiểm tra trọn luồng đọc dữ liệu, tải backbone, tính loss và lưu checkpoint:

```bash
python -B -m smolvla_scratch.train_infer \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --ckpt_dir outputs/pick_place_smoke \
  --camera_names front wrist \
  --batch_size 1 \
  --num_epochs 1 \
  --chunk_size 50 \
  --lr 1e-4 \
  --seed 0 \
  --num_workers 0 \
  --device auto
```

Dùng thư mục output mới hoặc rỗng. Nếu đã chạy thử vào thư mục trên, đổi tên thư mục cho lần thử tiếp theo.

`--device auto` chọn **CUDA → MPS → CPU** theo khả năng máy. Có thể chỉ định `--device mps` trên Mac hỗ trợ MPS, `--device cuda` trên máy NVIDIA có CUDA, hoặc `--device cpu`.

Chạy thử thành công khi hoàn tất 1 epoch và có `outputs/pick_place_smoke/policy_last.ckpt`. Checkpoint này chỉ phục vụ kiểm tra luồng; bước tiếp theo train một run mới.

## 4. Train chính

```bash
python -B -m smolvla_scratch.train_infer \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --ckpt_dir outputs/pick_place_run1 \
  --camera_names front wrist \
  --batch_size 4 \
  --num_epochs 100 \
  --chunk_size 50 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --seed 0 \
  --num_workers 0 \
  --save_every 10 \
  --device auto
```

Đổi `--dataset_dir` nếu dữ liệu nằm ở nơi khác. Nếu thiếu bộ nhớ, giảm `--batch_size 4` xuống `1` và chạy lại vào thư mục mới. Các giá trị trên là cấu hình khởi đầu; cần đánh giá trên dữ liệu và task thực tế.

| Tham số | Ý nghĩa |
| --- | --- |
| `--dataset_dir` | Thư mục chứa các `episode_*` |
| `--ckpt_dir` | Nơi lưu run; train mới/fine-tune cần thư mục mới hoặc rỗng |
| `--batch_size 4` | Số timestep mẫu trong mỗi batch |
| `--num_epochs 100` | Tổng số epoch mục tiêu |
| `--chunk_size 50` | Mỗi mẫu học dự đoán 50 action liên tiếp từ timestep hiện tại |
| `--lr 1e-4` | Peak learning rate; CLI yêu cầu LR hữu hạn và ≥ `2.5e-6` |
| `--val_ratio 0.2` | Tách khoảng 20% episode cho validation, ít nhất 1 episode |
| `--seed 0` | Seed cho split, khởi tạo, shuffle và noise |
| `--num_workers 0` | Đọc dữ liệu trong tiến trình chính |
| `--save_every 10` | Lưu thêm full checkpoint định kỳ mỗi 10 epoch |
| `--num_episodes N` | Tùy chọn: chỉ lấy N thư mục episode đầu theo tên đã sắp xếp |

Loader tách **theo episode** và chỉ tính normalization từ train split. Một epoch đi qua các timestep của train split; mỗi mẫu gồm ảnh/state tại `t`, instruction và action từ `t` đến `t + chunk_size - 1`. Phần action vượt cuối episode được padding và loại khỏi loss.

Khi bắt đầu, trainer kiểm tra dữ liệu và tính SHA-256 cho NPZ/video để phục vụ resume; bước này có thể mất thời gian. Khi train, theo dõi `train`, `val`, `best` trên thanh tiến độ. Best model được chọn bằng validation flow loss thấp nhất.

## 5. Đọc kết quả và chọn checkpoint

```text
outputs/pick_place_run1/
├── training_config.json
├── history.json
├── policy_best.ckpt
├── policy_last.ckpt
├── policy_last_model.ckpt
├── policy_epoch_9_seed_0.ckpt
└── ...
```

| File | Khi lưu / cách dùng |
| --- | --- |
| `policy_best.ckpt` | Model có validation loss thấp nhất; dùng cho eval, rollout hoặc fine-tune |
| `policy_last.ckpt` | Cuối mỗi epoch; chứa model, optimizer, scheduler, RNG, history và best snapshot để resume |
| `policy_last_model.ckpt` | Model cuối cùng, chỉ xuất khi train kết thúc bình thường; dùng cho eval/fine-tune |
| `policy_epoch_<index>_seed_<seed>.ckpt` | Full checkpoint định kỳ để resume từ mốc cũ; index bắt đầu từ 0, nên `epoch_9` nghĩa là đã xong 10 epoch |
| `history.json` | Train loss, validation loss, LR và global step theo epoch |
| `training_config.json` | Cấu hình model, dataset, split và stats của run |

Nếu có matplotlib, trainer còn lưu biểu đồ loss. Mọi checkpoint model đều chứa cả backbone và expert; full checkpoint lớn hơn vì có trạng thái training và best snapshot. `--save_every` chỉ điều khiển checkpoint định kỳ: `policy_last.ckpt` vẫn được lưu **mỗi epoch**.

## 6. Resume khi train bị ngắt

Tiếp tục run từ cuối epoch đã lưu gần nhất:

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/pick_place_run1/policy_last.ckpt \
  --device auto
```

Không cần nhập lại dataset, batch size, LR hoặc split: trainer khôi phục chúng từ checkpoint. Nếu bị ngắt giữa epoch thì phần epoch chưa hoàn tất được chạy lại. Chỉ có checkpoint sau khi hoàn tất ít nhất một epoch.

Nếu run đã hoàn tất 100 epoch và muốn chạy tới **tổng 120 epoch**:

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/pick_place_run1/policy_last.ckpt \
  --num_epochs 120 \
  --device auto
```

Lệnh này chạy thêm 20 epoch. Scheduler giữ kế hoạch LR gốc; sau horizon cũ, LR giữ ở floor, không warmup lại. Muốn đổi LR, batch size, task hoặc dataset thì dùng fine-tune ở bước 7.

Resume yêu cầu nội dung NPZ/video không đổi; có thể chuyển nguyên dataset bằng `--dataset_dir`. Muốn resume checkpoint định kỳ cũ, truyền checkpoint đó qua `--resume` và thêm `--ckpt_dir` trỏ tới thư mục mới/rỗng. Chỉ resume từ chính `policy_last.ckpt` của thư mục đích mới được ghi tiếp vào thư mục có sẵn.

## 7. Fine-tune với dữ liệu mới

Dùng weights đã học để tạo run mới:

```bash
python -B -m smolvla_scratch.train_infer \
  --finetune outputs/pick_place_run1/policy_best.ckpt \
  --dataset_dir data_recording/data/pick_place_smolvla_v2 \
  --ckpt_dir outputs/pick_place_finetune1 \
  --batch_size 4 \
  --num_epochs 50 \
  --lr 1e-5 \
  --device auto
```

Chuẩn bị dataset `pick_place_smolvla_v2` trước khi chạy. Fine-tune nạp weights/config nguồn, tạo optimizer và scheduler mới, bắt đầu epoch 0 và chọn best model lại. Camera, backbone và chunk size phải khớp checkpoint; task đọc từ metadata của dataset mới hoặc từ `--task`.

Mặc định giữ normalization của checkpoint (`--finetune_stats checkpoint`). Chỉ thêm `--finetune_stats dataset` khi chủ động muốn tính lại stats trên train split mới, vì thay stats sẽ đổi cách model diễn giải state và action.

## 8. Đánh giá offline

```bash
python -B -m smolvla_scratch.train_infer \
  --eval \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --checkpoint outputs/pick_place_run1/policy_best.ckpt \
  --device auto
```

Eval dùng lại validation split, task và normalization trong checkpoint; dữ liệu validation phải khớp nội dung đã lưu fingerprint. Kết quả in ra terminal và lưu vào:

```text
outputs/pick_place_run1/policy_best.eval.json
```

`flow_loss` là loss flow matching. `action_mse`, `action_mae` và `per_joint` đo sai số action sau denoise và đổi về đơn vị dữ liệu gốc, bỏ padding. Đây là đánh giá imitation offline; tỷ lệ hoàn thành task cần đo bằng rollout thực tế.

Đánh giá run fine-tune thì đổi sang dataset `pick_place_smolvla_v2` và checkpoint trong `outputs/pick_place_finetune1`.

## 9. Rollout trên robot

Rollout kết nối và di chuyển follower thật. Trước khi chạy, kiểm tra port, calibration ID, hai camera và `RESET_POSE` trong `low_control/hardware_constant.py` phù hợp với robot của mình; dừng chương trình thu dữ liệu để giải phóng port/camera.

```bash
python -B -m smolvla_scratch.train_infer \
  --rollout \
  --checkpoint outputs/pick_place_run1/policy_best.ckpt \
  --task "Pick up the block" \
  --episode_len 300 \
  --num_rollouts 1 \
  --device auto
```

Đặt instruction đúng task đã học; với checkpoint nhiều task cần chọn task cụ thể bằng `--task`. Có thể dùng `--env_kwargs path/to/env.json` để truyền `port`, `robot_id`, `reset_pose`, `max_relative_target` và `dt` khác mặc định. Camera phải khớp checkpoint.

Rollout chạy đồng bộ qua hàng đợi action; khi sinh chunk mới vẫn chờ inference, chưa đảm bảo 30 Hz và CLI hiện từ chối checkpoint bật guided RTC. `--n_action_steps` quy định số action thực thi trước khi lấy chunk mới, mặc định bằng `chunk_size`; `--num_euler_steps` quy định số bước denoise, mặc định 10. Khi eval/rollout, các giá trị này được lấy từ config checkpoint.

Kết quả lưu cạnh checkpoint thành `policy_best.rollout.json`. Môi trường chưa có reward/task-success detector, nên `return` có thể là `null`.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| Không import được module | Kích hoạt đúng env, cài requirements và chạy từ gốc repo |
| `Need at least two episodes` | Chuẩn bị ít nhất 2 episode hợp lệ |
| `missing task` | Bổ sung task vào metadata, hoặc dùng `--task` nếu cả dataset cùng task |
| `missing joint_names` | Kiểm tra nguồn recording; chỉ dùng `--legacy_joint_order` với đúng định dạng ACT cũ |
| Video lỗi hoặc số frame không khớp | Kiểm tra đủ `front.mp4`, `wrist.mp4` và số frame bằng số mẫu NPZ |
| `--ckpt_dir must be empty/new` | Đổi thư mục cho run mới; dùng `--resume` để tiếp tục run cũ |
| Thiếu bộ nhớ | Giảm batch size xuống 1; nếu đổi chunk size thì tạo run mới |
| Resume báo dataset đã thay đổi | Khôi phục dataset gốc, hoặc dùng `--finetune` cho dữ liệu mới |
| Không tải được model/tokenizer | Kiểm tra mạng hoặc cache; resume/eval vẫn cần config/tokenizer dù weights đã có trong checkpoint |
| Dashboard lỗi `cv2.imshow` | Kiểm tra bản OpenCV có GUI như bước 1 |

Chi tiết cơ chế checkpoint, resume và fine-tune: [`docs/TRAINING_VI.md`](docs/TRAINING_VI.md).
