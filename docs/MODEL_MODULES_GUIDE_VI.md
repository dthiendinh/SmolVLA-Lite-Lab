# Tách model thành năm module cho SmolVLA-Lite-Lab

Bản triển khai đã được tách vào **năm file thật trong repo** `/Users/tt/Documents/GitHub/SmolVLA-Lite-Lab/smolvla_scratch/models/`. Thư mục bài học có bản sao cùng nội dung để đọc cạnh source gốc. File triển khai gom chung `smolvlm_with_expert.py` đã được bỏ; file cùng tên trong `reference/` vẫn giữ nguyên để học và đối chiếu.

`flow_matching.py` mà m đang chỉnh trong repo được giữ nguyên. Không thay config hoặc processor trong lần tách này. [Tài liệu đọc hết 570 dòng gốc](</Users/tt/Documents/ChatGPT/SmolVLA plan/docs/SMOLVLM_WITH_EXPERT_GUIDE_VI.md>) vẫn giải thích thuật toán theo số dòng LeRobot; tài liệu này chỉ rõ **mỗi phần được chuyển vào đâu và gọi nhau thế nào**.

## 1. Ranh giới của từng file

| File | Phần thực sự triển khai | Không chịu trách nhiệm |
|---|---|---|
| `backbone.py` | tải VLM/processor, cắt text depth, freeze VLM, vision encoder + connector, lookup text token | tạo expert, action/time projections, solver |
| `action_expert.py` | tính FFN width, tạo text expert, sửa K/V projection cross, căn expert depth theo VLM | tải VLM pretrained, encode ảnh, tính loss |
| `embeddings.py` | RoPE, sinusoidal timestep, state/action/time projections, ghép prefix/suffix | tải model, tính attention matrix, tích phân Euler |
| `attention.py` | boolean block mask, GQA eager, self/cross attention từng layer, update/read/crop KV cache | khởi tạo transformer, tạo prefix/suffix từ input |
| `smolvla.py` | phối hợp hai nhánh, residual/MLP/final norm, action output projection, loss và sampling | chép lại công thức attention hay loader từ các file khác |

Không bắt buộc mỗi file đều có một class bọc model. `load_backbone()` trả VLM thật; `build_action_expert()` trả text expert thật. Core đăng ký chúng đúng một lần dưới `vlm` và `lm_expert`, để giữ nguyên tên weight của core LeRobot. `AttentionEngine` không có weight riêng; nó thao tác lên layer được core đưa vào.

## 2. backbone.py: VLM pretrained

- `load_backbone(...)`: nếu có VLM/processor truyền vào thì dùng trực tiếp; nếu không thì tải theo model ID. True tải VLM weights, False tạo VLM random từ config. Cắt N text layers đầu và trả `(vlm, processor)`.
- `set_backbone_requires_grad(...)`: thực thi freeze vision hoặc toàn VLM. Khi fine-tune VLM, giữ quy tắc freeze output head/final norm/tail layers của source và báo lỗi nếu mẫu tên không còn match.
- `keep_frozen_backbone_in_eval(...)`: gọi sau `.train(mode)` để VLM/vision frozen vẫn ở eval.
- `embed_image(vlm, image)`: ảnh đã chuẩn bị -> vision encoder -> connector -> image tokens `[B,Nimg,Dv]`.
- `embed_language_tokens(vlm, tokens)`: token IDs -> embedding table VLM -> `[B,Ntext,Dv]`.

Hai hàm embed ở đây chạy **layer pretrained thuộc backbone**. Phần ghép token và các projection riêng cho bài toán robot nằm trong `embeddings.py`.

## 3. action_expert.py: tạo expert

- `get_intermediate_size(...)`: giữ đúng thứ tự ép int và làm tròn FFN size lên bội số 256.
- `build_action_expert(text_config, num_vlm_layers, ...)`: deepcopy text config; thu nhỏ hidden width; giữ head geometry; chọn số layer; tạo expert mới bằng Transformers; đổi K/V input dimension ở layer cross; xóa vocabulary embedding không dùng; trả transformer thật.
- `align_model_layers(vlm_layers, expert_layers)`: ví dụ 4 VLM/2 expert -> `[E0,None,E1,None]`. Không nhân đôi expert và không đăng ký thêm weight.

Cross/self K/V được chọn theo **global VLM layer index**, nên nhánh reduced expert depth + interleaving không mắc lỗi index local của source gốc.

Expert không có `forward()` độc lập đọc ảnh hoặc token text. Ở mỗi global layer, core đưa layer VLM/expert tương ứng cho `AttentionEngine`, rồi cập nhật residual/MLP. Điều này giữ đúng cách hai nhánh trao đổi K/V của LeRobot.

## 4. embeddings.py: token position, flow time, prefix và suffix

- `apply_rope(x, positions, ...)`: quay Q/K theo **vị trí token**, input `[B,L,H,Dh]`. Attention gọi hàm này.
- `create_sinusoidal_pos_embedding(time, dimension, ...)`: mã hóa **flow time**, hỗ trợ `[B]` và `[B,S]`; dùng float32 trên MPS/XPU, float64 trên CPU trước khi cast sang action dtype.
- `PrefixSuffixEmbeddings.__init__`: tạo `state_proj`, `action_in_proj`, `action_time_mlp_in`, `action_time_mlp_out`. Không đăng ký VLM vào class này.
- `.embed_prefix(core, image, tokens, language_pad, state)`: gọi image/text embedding của backbone qua core, scale sqrt(Dv), pad/project state, ghép `[image;language;state]`; trả embedding, padding mask và causal boundaries.
- `.embed_suffix(x_t, time)`: project noisy action, embed timestep, broadcast/ghép theo action tokens, chạy hai Linear với SiLU; trả suffix `[B,S,De]`.

`action_out_proj` đặt trong `SmolVLA`: nó chuyển **output expert** thành velocity, không tạo embedding đầu vào.

Ví dụ baseline đang hỗ trợ một camera và một state token. Ảnh phải được resize/normalize trước; output action phải được inverse-normalize sau. Multi-camera/robot queue/RTC không được ngầm coi là đã triển khai.

## 5. attention.py: toàn bộ attention và cache

- `block_mask(pad, boundaries)`: tạo bool `[B,Q,K]` từ nhóm causal và padding; True nghĩa được đọc.
- `AttentionEngine.__init__(Hq,Hkv)`: giữ metadata và kiểm tra query heads chia hết KV heads. Không tạo parameter.
- `.forward_attn_layer(...)`: norm hai nhánh, chiếu Q/K/V bằng weight tương ứng, ghép token, RoPE, cache update và eager attention. Trả output attention chung.
- `.forward_cross_attn_layer(...)`: prefix tự attention khi train; expert Q đọc prefix K/V đã chiếu lại. Khi denoise đọc prefix cache, không append suffix ở cross layer.
- `.eager_attention_forward(...)`: repeat KV heads đúng thứ tự GQA, float32 QK score, scale theo Dh, mask, softmax và nhân V. Chưa chạy o_proj.
- `crop_prefix_cache(cache,P)`: bỏ phần suffix vừa append **theo từng layer**, giữ prefix ở cross layers. Transformers 5.22 nhận số âm là số token cần bỏ.

File này import `apply_rope` từ embeddings. Nó không import `smolvla`, vì vậy không tạo vòng import.

## 6. smolvla.py: hai mức điều phối

### SmolVLMWithExpertModel: core tương đương file gốc

Tên class được giữ để dễ so sánh, nhưng **implementation đã phân chia**:

- `__init__`: gọi loader backbone, builder expert và tạo attention engine.
- `get_vlm_model`, `embed_image`, `embed_language_tokens`: interface ngắn dẫn vào backbone.
- `set_requires_grad`, `train`: gọi freeze/mode helpers; train trả self.
- `get_model_layers`: gọi layer alignment của action_expert.
- `forward`: kiểm tra input; tạo prefix cache khi cần; chọn self/cross theo global layer; gọi attention engine; chạy o_proj/residual/MLP và final norm riêng hai nhánh.
- `get_attention_interface`, property `eager_attention_forward`: giữ interface nhỏ cho test/so sánh; công thức thực tế chỉ tồn tại trong attention.py.

Core vẫn nhận `[prefix,suffix]` và trả `([prefix_out,suffix_out],cache)`. `state_dict` core vẫn dùng `vlm.*` và `lm_expert.*`; có thể strict-load core weights của LeRobot khi cùng kiến trúc.

### SmolVLA: ghép core vào flow matching

- `__init__`: nhận core, tạo `PrefixSuffixEmbeddings` và `action_out_proj`.
- `prefix(...)` / `suffix(...)`: gọi embedding module.
- `forward(prefix,pad,boundaries,x_t,time)`: tạo suffix và full mask, gọi core, chiếu suffix output thành velocity `[B,S,padded_dim]`.
- `training_loss(...)`: noise/time sampling nếu caller chưa đưa vào; tạo x_t/target theo NOISE_AT_ONE; tính MSE trên action_dim thật; loại timestep padding; all-padded batch trả differentiable zero.
- `sample(...)`: Euler sampling; cache prefix một lần và crop sau mỗi suffix call. `use_cache=False` chạy full prefix/suffix mỗi bước để có đường đối chiếu đúng.
- `positions(pad)`: cumsum padding mask -1 để token padding không tăng vị trí.

`SmolVLA` hiện là baseline phục vụ model/flow matching, chưa phải bản sao toàn bộ API policy LeRobot. Constructor đang dùng cùng padded width cho state/action, phù hợp config dự án hiện tại đều bằng 32.

Weight của policy nay có các nhóm `core.vlm.*`, `core.lm_expert.*`, `embeddings.*`, `action_out_proj.*`. Checkpoint **toàn policy** theo layout cũ cần remap key trước khi load; không nhầm tính tương thích core với toàn policy.

## 7. Tất cả function gốc nay nằm đâu?

| Function gốc | Nơi triển khai hiện tại |
|---|---|
| apply_rope | embeddings.py |
| get_intermediate_size | action_expert.py |
| __init__ | smolvla core gọi backbone.load_backbone và action_expert.build_action_expert |
| get_vlm_model | smolvla core |
| set_requires_grad | backbone.set_backbone_requires_grad; expert guard trong builder |
| train | smolvla core + backbone.keep_frozen_backbone_in_eval |
| embed_image | backbone.py; core gọi lại |
| embed_language_tokens | backbone.py; core gọi lại |
| forward_attn_layer | attention.AttentionEngine |
| forward_cross_attn_layer | attention.AttentionEngine |
| get_model_layers | action_expert.align_model_layers; core gọi lại |
| forward | smolvla.SmolVLMWithExpertModel |
| get_attention_interface | helper nhỏ trong smolvla core |
| eager_attention_forward | attention.AttentionEngine |

Không bỏ mất nhánh cross/self, freeze, reduced-depth mapping hay cache. Ngoài phần của file gốc, các projection/prefix/suffix trước đây chỉ nằm trong ví dụ cũng đã chuyển vào module chính.

## 8. Luồng gọi và thứ tự tự viết

```text
smolvla.py
 ├── backbone.py       tải VLM, freeze, encode ảnh/text
 ├── action_expert.py  tạo expert và map layers
 ├── embeddings.py     state/action/time, prefix/suffix
 ├── attention.py      self/cross/GQA/cache
 │    └── embeddings.apply_rope
 └── flow_matching.py  x_t/target và Euler
```

Nên đọc/viết theo thứ tự `embeddings primitives -> attention -> backbone -> action_expert -> prefix/suffix embeddings -> smolvla`. Tách primitive math trước, rồi ghép weight và dữ liệu. Không dùng `from smolvla import ...` trong bốn module còn lại.

## 9. Chạy trực tiếp code trong repo

Tại thư mục repo `/Users/tt/Documents/GitHub/SmolVLA-Lite-Lab`:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 \
'/Users/tt/Documents/ChatGPT/SmolVLA plan/.venv-smolvlm/bin/python' -B test/test_smolvlm_with_expert.py

PYTHONPATH=. HF_HUB_OFFLINE=1 \
'/Users/tt/Documents/ChatGPT/SmolVLA plan/.venv-smolvlm/bin/python' -B -m examples.smolvlm_expert_steps
```

Test mặc định tìm file LeRobot ở repo `lerobot` cạnh Lite-Lab; biến `LEROBOT_SMOLVLM_PATH` cho phép chỉ định vị trí khác. Test kiểm tra SHA256 của đúng source đã học; nếu đổi phiên bản source phải xem lại baseline đối chiếu. Test tham chiếu cần môi trường có thể import LeRobot; năm module triển khai không import LeRobot.

Ví dụ chỉ tạo tiny config/local token IDs và gọi class trong model modules. Không giấu model implementation trong `examples/`. `VelocityExample` trong ví dụ là alias của class `SmolVLA` thực tế.

Các test gồm output/loss/gradient/cache so với source nguyên trạng, GQA so SDPA, time embedding so utility LeRobot, masked training loss, all-padded loss, sampling có/không cache và kiểm tra parameter chỉ đăng ký một lần.

Config dự án vẫn có vấn đề `RTCConfig` chưa import/định nghĩa như đã chỉ ra ở tài liệu trước. Ví dụ offline truyền core trực tiếp nên không phụ thuộc lỗi đó. Khi train expert mới trên VLM pretrained phải đặt `load_vlm_weights=True`; lần chia module này không tự thay lựa chọn trong config của m.
