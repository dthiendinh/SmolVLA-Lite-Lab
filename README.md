# SmolVLA-Lite-Lab

SmolVLA tự triển khai: thu dữ liệu SO-ARM101 → train action expert trên SmolVLM pretrained → đánh giá offline → rollout. File chạy chính là `smolvla_scratch/train_infer.py`; `main.py` chứa các hàm hỗ trợ và parser.

## Môi trường

Dùng Python 3.12 trong môi trường riêng. Từ thư mục gốc repo:

```bash
python -m pip install -r requirements.txt
python -B -m smolvla_scratch.train_infer --help
```

Thu dữ liệu và rollout cần thêm LeRobot với hỗ trợ Feetech. Nếu checkout LeRobot nằm cạnh repo này:

```bash
python -m pip install -e '../lerobot[feetech]'
python -c 'import transformers, cv2, dm_env, lerobot; print(lerobot.__file__)'
```

Kiểm tra đường dẫn in ra đúng checkout muốn dùng. Môi trường có sẵn trên máy khi audit import LeRobot từ `/Users/tt/lerobot`, còn checkout đang xem là `/Users/tt/Documents/GitHub/lerobot`.

Dashboard cần OpenCV có GUI. Kiểm tra dòng `GUI` bằng `python -c 'import cv2; print(cv2.getBuildInformation())'`; nếu là `NONE`, thay bản OpenCV headless bằng bản có GUI trong môi trường dùng để thu dữ liệu. Các bản OpenCV dùng chung tên import `cv2`, nên cần kiểm tra sau khi cài LeRobot.

## Kiểm tra không cần robot hoặc tải weights

```bash
HF_HUB_OFFLINE=1 python -B -m unittest discover -s tests -v
```

Test dùng các layer Transformers thật với model nhỏ ngẫu nhiên, dữ liệu NPZ/MP4 thật và môi trường robot giả. Test recorder cần cài LeRobot; nếu thiếu sẽ báo skip. Test này xác minh luồng chương trình, không đo chất lượng gắp vật hoặc tốc độ model 500M.

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
