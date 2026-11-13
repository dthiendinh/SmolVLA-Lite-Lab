# SmolVLA-Lite-Lab

SmolVLA tự triển khai: thu dữ liệu SO-ARM101 → train action expert trên SmolVLM pretrained → đánh giá offline → rollout. File chạy chính là `smolvla_scratch/train_infer.py`; `main.py` chứa các hàm hỗ trợ và parser.

## Môi trường

Dùng Python 3.12 và virtual env riêng. Từ thư mục gốc repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` chỉ liệt kê các dependency trực tiếp của dự án và `num2words` mà SmolVLM processor bắt buộc cần. `lerobot[feetech]` tự cài các dependency robot; không cần liệt kê lại chúng. `matplotlib` dùng cho biểu đồ loss và có thể bỏ nếu không cần biểu đồ.

Nếu `.venv` là env từng được tạo bằng script cũ, bỏ file liên kết source cũ trước khi dùng bản LeRobot cài bằng pip:

```bash
rm -f .venv/lib/python3.12/site-packages/smolvla_local_sources.pth
```

**Dashboard OpenCV:** LeRobot 0.6.2 khai báo `opencv-python-headless`, còn dashboard dùng `cv2.imshow` cần `opencv-python`. Hai gói dùng chung tên `cv2`; [OpenCV hướng dẫn chỉ giữ một biến thể](https://pypi.org/project/opencv-python/). Sau khi cài requirements, nếu cả hai gói cùng có mặt, chạy:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install 'opencv-python>=4.10,<4.14'
```

Đây là cách thay bản headless bằng bản có GUI cho dashboard. Sau đó `pip check` có thể báo LeRobot thiếu `opencv-python-headless` theo metadata của nó, dù `cv2` có GUI đã có mặt. Mỗi lần cài lại LeRobot/requirements cần kiểm tra lại điểm này; file requirements thuần không tự thực hiện bước thay thế gói.

Mỗi lần mở terminal mới:

```bash
cd /Users/tt/Documents/GitHub/SmolVLA-Lite-Lab
source .venv/bin/activate
python -B -m smolvla_scratch.train_infer --help
```

Kiểm tra Python và LeRobot đang dùng bằng:

```bash
python -c 'import sys, lerobot; print(sys.executable); print(lerobot.__file__)'
```

LeRobot cài bằng pip phải được import từ `.venv/.../site-packages/lerobot`. Thoát env bằng `deactivate`. `.venv/`, dữ liệu thu và outputs được bỏ qua trong Git.

## Kiểm tra môi trường không cần robot hoặc tải weights

```bash
python -m pip check
python -B -m smolvla_scratch.train_infer --help
python -B -m data_recording.record_teleop_data --help
python -c 'import dm_env; from low_control.low_level_control import RealEnv; print("Rollout imports OK")'
```

Các lệnh này kiểm tra dependency và import của ba luồng, không kết nối phần cứng hoặc đo chất lượng policy. Kết quả audit model trước đây nằm trong `docs/AUDIT_VI.md`.

## Thu dữ liệu

Đặt cổng USB, ID calibration và camera trong `low_control/hardware_constant.py` theo phần cứng thực tế, rồi chạy:

```bash
python -B -m data_recording.record_teleop_data --data-dir pick_place_smolvla_v1
```

Bấm `t` để nhập task tiếng Anh → Enter → `r` ghi → `s` lưu; `x` bỏ episode, `q` thoát. Chi tiết trong `data_recording/CHANGES.md`.

Mỗi `episode_XXXX/` gồm `data.npz`, `front.mp4`, `wrist.mp4`, `metadata.json`. Thứ tự 6 khớp được ghi trong NPZ và kiểm tra khi đọc. State/action dùng độ cho 5 khớp thân; gripper dùng thang riêng của LeRobot. Giữ cùng calibration khi thu và rollout.

## Train

```bash
python -B -m smolvla_scratch.train_infer \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --ckpt_dir outputs/pick_place_run1 \
  --batch_size 4 --num_epochs 100 --device auto
```

Cần ít nhất 2 episode để tách train/validation. Loader đọc task riêng từ `metadata.json`; dùng `--task "Pick up the block"` nếu muốn đặt một instruction chung cho toàn bộ dữ liệu, kể cả dữ liệu cũ thiếu metadata. Chỉ dùng `--legacy_joint_order` cho recording ACT cũ lưu các cột theo `sorted(joint_keys)`.

Lần đầu cần tải backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Action expert được khởi tạo mới; pretrained VLM được đóng băng. Đây là train expert của repo, không phải nạp sẵn checkpoint policy SmolVLA đã train action. Thư mục checkpoint phải mới hoặc rỗng. Trainer chưa hỗ trợ resume optimizer/scheduler.

Đầu ra: `policy_best.ckpt`, `policy_last.ckpt`, checkpoint định kỳ, `history.json`, `training_config.json`. Có matplotlib thì lưu thêm biểu đồ loss. Checkpoint chứa weights, cấu hình, normalization stats, split và task đã dùng.

## Đánh giá offline

```bash
python -B -m smolvla_scratch.train_infer --eval \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --checkpoint outputs/pick_place_run1/policy_best.ckpt --device auto
```

Dùng lại split, stats, seed và batch size validation của checkpoint. Kết quả gồm flow loss và MSE/MAE action ở đơn vị dữ liệu gốc; không phải tỷ lệ gắp thành công. Nạp checkpoint không tải lại weights pretrained đã có trong checkpoint, nhưng vẫn cần config/tokenizer của backbone trong cache hoặc từ Hub.

## Rollout

Sau khi xác nhận phần cứng và calibration phù hợp:

```bash
python -B -m smolvla_scratch.train_infer --rollout \
  --checkpoint outputs/pick_place_run1/policy_best.ckpt \
  --episode_len 300 --device auto
```

Với checkpoint nhiều task, thêm `--task "Pick up the block"`. Có thể truyền file JSON qua `--env_kwargs` để đổi `port`, `robot_id`, `reset_pose`, `max_relative_target`, `dt`. Camera dùng cùng tên với checkpoint.

Rollout hiện chạy đồng bộ bằng hàng đợi action. Lúc tạo chunk mới vẫn có khoảng chờ inference; chưa có worker RTC bất đồng bộ và chưa đảm bảo 30 Hz. Checkpoint bật guided RTC chỉ dùng được qua `predict_action_chunk`; CLI rollout hiện từ chối cấu hình đó. Reward thật chưa được đo, nên `return` có thể là `null`.

Xem kết quả rà soát và phần chưa xác minh trong `docs/AUDIT_VI.md`.
