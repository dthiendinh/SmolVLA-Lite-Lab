# Train, resume, fine-tuning và đánh giá SmolVLA

## 1. Chọn chế độ theo mục đích

| Mục đích | Lệnh | Weights ban đầu | Optimizer/LR/RNG | Epoch, best loss, history |
| --- | --- | --- | --- | --- |
| Train expert đầu tiên | Không thêm mode | VLM pretrained, expert mới | Tạo mới | Từ 0 |
| Tiếp tục run bị ngắt | `--resume FULL_CHECKPOINT` | Weights cuối epoch đã lưu | Khôi phục | Tiếp tục |
| Thích nghi model sang dữ liệu/task mới | `--finetune CHECKPOINT` | Toàn bộ policy nguồn | Tạo mới | Từ 0, chọn best lại |
| Đánh giá offline | `--eval --checkpoint CHECKPOINT` | Toàn bộ policy nguồn | Không cập nhật | Dùng holdout nguồn |

Fine-tuning chỉ lấy kiến thức trong weights, không mang momentum của AdamW hay vị trí scheduler sang run mới. Resume cần những trạng thái đó vì đây vẫn là cùng một quá trình tối ưu.

Repo giữ implementation SmolVLA riêng, với VLM đóng băng và expert/projection có thể học theo config. CLI hiện không chuyển đổi tên weights từ checkpoint chính thức `lerobot/smolvla_base`. Đường dẫn `--finetune`/`--resume` phải trỏ đến file checkpoint được trainer cục bộ tạo ra.

## 2. Chạy từ thư mục gốc repo

### Train đầu tiên

```bash
python -B -m smolvla_scratch.train_infer \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --ckpt_dir outputs/run1 \
  --num_epochs 100 --batch_size 4 --save_every 10 --device auto
```

`--num_epochs 100` lập kế hoạch scheduler cho 100 epoch. VLM pretrained được tải ở lần đầu, expert khởi tạo mới. Split theo episode để các frame cùng episode không xuất hiện ở cả train và validation. Normalization được tính trên timestep thật trong train, bỏ padding và không dùng validation.

### Resume run chưa hoàn thành

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/run1/policy_last.ckpt --device auto
```

Mặc định lấy dataset và tổng số epoch từ checkpoint, ghi tiếp vào thư mục chứa `policy_last.ckpt`. Có thể đổi `--device`, `--num_workers`, `--save_every`, hoặc chuyển nguyên dataset sang đường dẫn mới bằng `--dataset_dir`. Kết quả khớp chính xác với train liên tục chỉ được kiểm tra trên CPU cùng môi trường; thay phần cứng/backend có thể tạo sai khác số học và RNG.

Có thể đặt `--ckpt_dir outputs/run1_resumed` để ghi tiếp vào thư mục mới. Không cần copy `policy_best.ckpt` thủ công vì full checkpoint chứa best snapshot của thời điểm lưu.

Trainer lưu ở **cuối mỗi epoch**. Ví dụ checkpoint ghi `epoch=19` nghĩa là đã xong 20 epoch; resume bắt đầu `epoch=20`. Nếu bị ngắt giữa epoch 21, các batch đã chạy trong epoch đó chưa nằm trong checkpoint và sẽ được chạy lại.

### Resume một checkpoint định kỳ cũ

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/run1/policy_epoch_9_seed_0.ckpt \
  --ckpt_dir outputs/run1_from_epoch10 --device auto
```

Index trong tên file bắt đầu từ 0: `epoch_9` là đã hoàn tất 10 epoch. Thư mục đích phải mới/rỗng. Chỉ resume từ chính `policy_last.ckpt` của thư mục đích mới được ghi tiếp vào thư mục có sẵn, nhằm tránh ghi đè một nhánh train mới hơn.

### Kéo dài một run đã kết thúc

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/run1/policy_last.ckpt --num_epochs 120 --device auto
```

Nếu đã xong 100 epoch, lệnh trên chạy thêm 20 epoch. `--num_epochs` luôn chỉ tổng mục tiêu. Scheduler vẫn giữ horizon 100 epoch gốc. `LambdaLR.state_dict()` không lưu hàm lambda nên trainer lưu riêng `scheduler_total_steps` để dựng lại đúng hàm trước khi nạp trạng thái. Sau horizon cũ LR giữ floor; muốn warmup/schedule/LR mới thì tạo run fine-tuning.

Resume không cho thay batch size, seed, LR, chunk size, task, camera hoặc split. Các option đó được kế thừa nếu không gõ; nếu gõ giá trị khác, CLI báo lỗi để tránh dùng momentum/lịch cũ cho một bài toán đã đổi.

### Fine-tuning

```bash
python -B -m smolvla_scratch.train_infer \
  --finetune outputs/run1/policy_best.ckpt \
  --dataset_dir data_recording/data/pick_place_smolvla_v2 \
  --ckpt_dir outputs/finetune1 \
  --num_epochs 50 --batch_size 4 --lr 1e-5 --device auto
```

Run bắt đầu ở epoch 0, optimizer không có momentum cũ, scheduler có kế hoạch 50 epoch và history/best loss được tạo lại. Nạp weights bằng `strict=True` để không âm thầm bỏ qua layer không tương thích. Camera, backbone và chunk size được lấy từ checkpoint; CLI từ chối yêu cầu đổi chúng. Task có thể lấy từ metadata mới hoặc override bằng `--task`. Không truyền task thì không kế thừa task của dataset cũ.

Normalization mặc định là `--finetune_stats checkpoint`: giữ cách diễn giải đầu vào/đầu ra mà weights đã học. Nếu dữ liệu mới có phân phối khác và bạn chủ động muốn thay normalization:

```bash
python -B -m smolvla_scratch.train_infer \
  --finetune outputs/run1/policy_best.ckpt \
  --dataset_dir data_recording/data/pick_place_smolvla_v2 \
  --ckpt_dir outputs/finetune_new_stats \
  --num_epochs 50 --lr 1e-5 --finetune_stats dataset --device auto
```

Stats mới chỉ tính trên train split mới. Đổi stats sẽ đổi cả giá trị state đi vào model và ánh xạ action về đơn vị robot, nên cần đánh giá lại model. Chế độ fine-tuning cũng nhận full checkpoint hoặc model checkpoint format v1 cũ.

## 3. File nào dùng cho việc gì?

| File | Khi lưu | Resume? | Eval/fine-tuning? |
| --- | --- | --- | --- |
| `policy_best.ckpt` | Khi validation flow loss giảm | Không | Có |
| `policy_last_model.ckpt` | Khi vòng train kết thúc bình thường | Không | Có |
| `policy_last.ckpt` | Cuối mỗi epoch | Có | Có |
| `policy_epoch_<index>_seed_<seed>.ckpt` | Mỗi `save_every` epoch | Có | Có |
| `history.json` | Cuối mỗi epoch | Lịch sử được lấy từ full checkpoint | Dùng để xem loss/LR |
| `training_config.json` | Khi mở run mới | Mô tả cấu hình | Tham khảo |
| `resume_config.json` | Khi resume trong cùng thư mục | Mô tả lần resume gần nhất | Tham khảo |

Cả hai loại `.ckpt` lưu:

- `format_version`, `checkpoint_kind`: phiên bản và loại checkpoint.
- `model`: toàn bộ weights, gồm VLM và expert.
- `policy_config`: kiến trúc, features, normalization mode và tham số tối ưu.
- `dataset_stats`: stats để normalize state/action và đổi action về đơn vị gốc.
- `data`: split, task từng episode, camera, thứ tự khớp, validation seed/batch size và fingerprint dữ liệu.
- `epoch`, `best_val_loss`: epoch của weights và best loss đã gặp.

Full checkpoint thêm `training_state`:

- AdamW: momentum, phương sai, step count và parameter groups.
- Scheduler: step/LR hiện tại và horizon gốc để tái tạo hàm schedule.
- RNG: Python, NumPy, PyTorch CPU, CUDA/MPS nếu có.
- `global_step`, `steps_per_epoch`, CLI arguments và history đầy đủ.
- `best_epoch`, `best_model`: giữ model tốt nhất dù copy full checkpoint sang máy khác mà không có file best đi kèm.
- SHA-256 của NPZ và video trong split: resume từ chối dữ liệu bị thay đổi dù tên episode vẫn giống.

Full checkpoint lớn hơn model checkpoint vì chứa optimizer và bản best model. Checkpoint định kỳ được giữ lại, vì vậy số lượng file tăng theo run. Các file được ghi sang `.tmp` rồi thay thế file đích sau khi ghi xong. Nếu crash giữa lúc ghi, resume từ file `.ckpt` đã hoàn tất, không chọn `.tmp`.

`history.json` và config là file tiện đọc; nguồn chính xác để resume là full `.ckpt`. Thứ tự khôi phục là: dựng model → nạp weights → dựng optimizer/scheduler → nạp scheduler → nạp optimizer → khôi phục RNG sau cùng. Thứ tự này tránh khởi tạo model tiêu thụ mất chuỗi random đã lưu hoặc scheduler ghi đè LR được khôi phục. Xem [PyTorch Optimizer.load_state_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html).

## 4. Đánh giá offline và rollout

```bash
python -B -m smolvla_scratch.train_infer --eval \
  --dataset_dir data_recording/data/pick_place_smolvla_v2 \
  --checkpoint outputs/finetune1/policy_best.ckpt --device auto
```

Eval lấy đúng validation split, task, stats, seed và batch size trong checkpoint. Với format mới, nội dung NPZ/video validation phải khớp fingerprint; có thể chỉ giữ các episode validation trên máy eval. Task đã được lưu trong checkpoint nên metadata task bị sửa sau đó không thay đổi instruction của run cũ.

Kết quả ghi cạnh checkpoint vào `policy_best.eval.json`:

- `flow_loss`: loss flow matching, dùng noise/time có seed cố định; đây là tiêu chí chọn best model.
- `action_mse`, `action_mae`: sai số action sau denoise và denormalize, chỉ tính trên target thật, bỏ padding.
- `per_joint`: MSE/MAE cho từng tên khớp. MSE có đơn vị bình phương; MAE có cùng đơn vị dữ liệu thu.
- `num_samples`, `valid_action_timesteps`, `validation_seed`, `checkpoint_epoch`: ngữ cảnh để đọc kết quả.

Flow loss dùng trung bình số học loss của các batch như trainer hiện có. Action MSE/MAE dùng tổng lỗi trên tất cả giá trị action hợp lệ; đừng gộp hai cách tính này làm một.

Đây là đánh giá imitation trên dữ liệu giữ lại; MSE thấp không tự suy ra tỷ lệ gắp thành công. Đánh giá task thực tế cần rollout và tiêu chí thành công được xác định cho môi trường. CLI rollout hiện ghi steps và return nếu môi trường có reward; chưa tự chấm success rate.

```bash
python -B -m smolvla_scratch.train_infer --rollout \
  --checkpoint outputs/finetune1/policy_best.ckpt \
  --task "Pick up the block" --episode_len 300 --device auto
```

Checkpoint đã chứa weights backbone nên eval/fine-tuning/resume không tải lại pretrained weights. Config/tokenizer của VLM vẫn cần nằm trong cache hoặc tải từ Hub. Robot phải dùng cùng thứ tự khớp, đơn vị đo và calibration như khi thu dữ liệu.

## 5. Kiểm tra code

```bash
python -B -m unittest discover -s tests -v
```

Các test bao phủ resume với AdamW/LR/RNG/history, fine-tuning reset trạng thái nhưng giữ weights, kéo dài run với scheduler gốc, checkpoint cũ, dữ liệu bị sửa và CLI train/eval/resume/fine-tuning với NPZ/video giả. Một test dùng policy SmolVLA thật và các lớp Transformers nhỏ, chỉ thay tải config/tokenizer bằng đối tượng trong RAM. Nó yêu cầu weights/history sau resume khớp chính xác với run liên tục trên CPU và chạy cả denoise/đánh giá action.

Bộ test không tải model Hub, không kết nối robot và không chứng minh chất lượng model trên dữ liệu thật.
