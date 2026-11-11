# Giải thích smolvla.py: từng hàm, luồng dữ liệu và shape tensor

Bài cuối trong chuỗi năm module model. Đọc theo file **`smolvla_scratch/models/smolvla.py` gồm 336 dòng** ở thời điểm 10/11/2026. Bản trong thư mục bài học và bản trong repo `SmolVLA-Lite-Lab` giống nhau tại thời điểm kiểm tra. Số dòng trong bài là của file này, không phải `modeling_smolvla.py` của LeRobot.

Mục tiêu là hiểu đủ để tự vẽ kiến trúc: input nào đi vào nhánh nào, tensor đổi chiều ở đâu, hai nhánh trao đổi thông tin thế nào, và train/inference khác nhau ở đâu.

## 1. File này có hai tầng

```text
SmolVLA                         policy model
├── embeddings                  state/action/time projections
├── core: SmolVLMWithExpertModel
│   ├── vlm                     vision encoder, connector, text transformer
│   ├── lm_expert               action transformer
│   └── attention               bộ thực thi attention, không sở hữu weight
└── action_out_proj             expert hidden → velocity
```

`SmolVLMWithExpertModel` nhận **embedding** và trả **hidden state**. `SmolVLA` dùng hidden state đó để dự đoán velocity, tính flow-matching loss hoặc tích phân thành action. Vì vậy `core.forward()` và `policy.forward()` trả hai thứ khác nhau.

Điểm quan trọng khi vẽ: VLM text transformer và expert chạy **từng layer song song, trao đổi K/V ở từng layer**. Không vẽ thành “chạy xong toàn bộ text transformer → mới đưa output cuối sang toàn bộ expert”. Khi inference, có thể tính và cache nhánh prefix trước vì mask ngăn prefix đọc action.

## 2. Các ký hiệu phải phân biệt

| Ký hiệu | Ý nghĩa | Trong cấu hình dự án |
|---|---|---|
| `B` | batch size | tùy batch |
| `I` | số token ảnh sau connector | 64 với một ảnh 512×512 theo đường encode trực tiếp này |
| `T` | số vị trí token ngôn ngữ, tính cả padding | thay đổi theo batch; tối đa 48 theo config |
| `P` | số vị trí prefix | `I + T + 1`; một token state |
| `S` | số action trong chunk, cũng là số suffix token | 50 |
| `A` | chiều action sau feature padding | 32 |
| `d_a` | số chiều action thật | 6 |
| `Dv` | hidden width của nhánh VLM text | 960 |
| `De` | hidden width của expert | `int(960 × 0.75) = 720` |
| `Hq` | số query heads | 15 |
| `Hkv` | số KV heads | 5 |
| `Dh` | chiều mỗi head | 64 |
| `N` | số text layers giữ lại của VLM | 16 |
| `Ne` | số expert layers thật | 16 vì config chọn `-1` |

Các giá trị gốc của backbone được đối chiếu với [config chính thức](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/blob/main/config.json): text width 960, query/KV heads 15/5, head dimension 64, text FFN 2560, text depth gốc 32; vision width 768, patch 16, image 512, scale factor 4. Các số 720, 16 layer giữ lại, chunk 50 và pad 32 là kết quả ghép config/code của dự án.

**Đừng gộp `6`, `32`, `720`, `960`, `64` thành một loại “dim”:** chúng lần lượt là chiều action thật, chiều action padded, chiều expert hidden, chiều VLM hidden, chiều mỗi attention head. `S=50` là trục thời gian tương lai của robot; flow time `t` là mức nhiễu trong solver.

### Default trong file và cấu hình dự án khác nhau

| Tham số core | Default constructor trong smolvla.py | Config dự án nếu truyền qua `from_project_config()` |
|---|---|---|
| `attention_mode` | `self_attn` | `cross_attn` |
| `num_vlm_layers` | `-1`: giữ depth gốc | 16 |
| `num_expert_layers` | `-1`: cùng depth VLM | `-1`: cùng depth VLM đã giữ |
| `expert_width_multiplier` | 0.5 | 0.75 |
| `self_attn_every_n_layers` | `-1` | 2 |
| `load_vlm_weights` | True | False trong config hiện tại |
| `freeze_vision_encoder` | False | True |

`smolvla.py` không tự import và áp dụng dataclass config dự án. Bài này dùng bộ số **960/720, 16 layer, cross xen self** khi ghi ví dụ dự án. Nếu gọi core constructor mặc định với backbone này, bộ số sẽ là **960/480, 32 layer, self-attention**.

## 3. Bản đồ tổng thể để vẽ

Trường hợp minh họa `T=48`, một ảnh 512×512: `P=64+48+1=113`. Padding ngôn ngữ vẫn chiếm vị trí tensor, nhưng bị mask. Nếu batch chỉ pad tới câu dài nhất dưới 48 token thì `P` nhỏ hơn 113.

```text
image [B,3,512,512]
  → vision patch/transformer [B,1024,768]
  → connector pixel shuffle [B,64,12288]
  → connector projection [B,64,960]
  → nhân sqrt(960) ──────────────────────────────────┐
                                                     │
tokens [B,T] → embedding lookup [B,T,960]              ├→ cat theo token
  → nhân sqrt(960) ──────────────────────────────────┤  prefix [B,P,960]
                                                     │
state [B,6] → zero-pad [B,32]                         │
  → Linear(32,960) → unsqueeze [B,1,960] ─────────────┘

x_t [B,50,32] → Linear(32,720) [B,50,720] ───────────┐
                                                     ├→ cat theo feature
t [B] → sin/cos [B,720] → expand [B,50,720] ──────────┘  [B,50,1440]
  → Linear(1440,720) → SiLU → Linear(720,720)
  → suffix [B,50,720]

[prefix [B,P,960], suffix [B,50,720]]
  → core: 16 cặp layer VLM/expert
  → [prefix_out [B,P,960], suffix_out [B,50,720]]
  → lấy suffix_out → float32 → Linear(720,32)
  → velocity [B,50,32]
  → train: MSE so với target, lấy 6 chiều thật → scalar
  → sample: Euler 10 lần, giữ [B,50,32] suốt solver
  → cắt 6 chiều thật ở cuối → normalized actions [B,50,6]
```

Shape ảnh ở trên được suy ra từ config và code `vision_model`/`connector`: `(512/16)²=1024` patches; pixel shuffle gộp `4²=16` patches nên còn 64 token và tăng channel thành `768×16=12288`, rồi projection về 960. Đây là baseline đưa **một ảnh trực tiếp** vào vision encoder; không áp dụng số 64 cho một pipeline khác có nhiều crop/camera.

Hai nhánh khác hidden width nên **không** `torch.cat([prefix,suffix], dim=1)` trực tiếp. Core giữ chúng trong một Python list; ở self-attention, chỉ ghép Q/K/V **sau projection** vì khi đó head geometry đã khớp.

## 4. Imports, dòng 19–30

| Import | Dùng cho việc gì trong file |
|---|---|
| `torch`, `nn`, `F` | thao tác tensor, đăng ký module, Linear và MSE |
| `DynamicCache` | chứa K/V theo từng global layer |
| `build_action_expert`, `align_model_layers` | tạo expert và đặt expert layers tương ứng VLM |
| `AttentionEngine`, `block_mask`, `crop_prefix_cache` | tính attention, tạo visibility mask, dọn cache sau mỗi bước Euler |
| các helper từ `backbone` | tải VLM, freeze, mode train/eval, encode ảnh/text |
| `PrefixSuffixEmbeddings` | sở hữu projection state/action/time |
| các helper flow matching | sample noise/time, tạo x_t/target, tích phân Euler |

## 5. positions(pad), dòng 33–35

```python
return torch.cumsum(pad, dim=1) - 1
```

Input `pad: bool[B,L]`, **True là token thật**, output `position_ids: int64[B,L]`.

```text
pad       = [1,1,0,0,1,1]
positions = [0,1,1,1,2,3]
```

Padding không làm tăng vị trí. Vị trí của padding có thể trùng token trước; đầu chuỗi pad có thể cho -1. Visibility mask mới là thứ loại các vị trí đó khỏi attention của query hợp lệ. Đây là vị trí token dùng cho RoPE, khác flow time `t` dùng trong suffix.

## 6. SmolVLMWithExpertModel.__init__, dòng 44–84

1. `super().__init__()` cho PyTorch đăng ký parameters/submodules.
2. `load_backbone(...)` trả VLM thật và processor. Có thể truyền sẵn `vlm`, `processor` để chạy offline. Nếu `num_vlm_layers>0`, chỉ giữ số text layer đầu được yêu cầu; không cắt vision depth theo tham số này.
3. `self.config=self.vlm.config`; `self.num_vlm_layers` lấy từ **số layer thật còn lại**, không dựa vào field config có thể vẫn ghi depth gốc.
4. `build_action_expert(...)` tạo expert mới từ bản copy text config. `De=int(Dv×multiplier)`, còn `Hq/Hkv/Dh` được giữ. Số expert layer mặc định bằng số VLM layer đang có.
5. Lưu depth, width, head counts và các tùy chọn attention/freeze.
6. `AttentionEngine(Hq,Hkv)` chỉ thực thi toán, không tạo một bộ Q/K/V weight khác. Weight vẫn thuộc layer VLM/expert.
7. `set_requires_grad()` thực thi freeze VLM.
8. Nếu `device!='auto'`, chuyển core sang device đó. Chuỗi `'auto'` ở đây nghĩa là không gọi `.to()` trong nhánh này; không phải một thuật toán tự chia model ra nhiều thiết bị.

**`load_vlm_weights=False` tạo VLM random khi không truyền sẵn VLM.** Nó không chỉ reset action expert. Expert vốn đã được tạo mới riêng. Khi muốn VLM pretrained và expert mới, dùng `load_vlm_weights=True`. Comment hiện tại trong config dự án có thể gây hiểu nhầm về điều này.

## 7. Các method core ngắn, dòng 86–110 và 233–238

| Method | Ý nghĩa và shape |
|---|---|
| `get_vlm_model()` | trả `self.vlm.model`, đối tượng chứa vision/connector/text model; không tạo tensor |
| `set_requires_grad()` | gọi helper freeze trên `self.vlm`; không đổi shape |
| `train(mode)` | gọi train/eval đệ quy rồi đưa phần backbone frozen về eval; trả `self` |
| `embed_image(image)` | `[B,3,Himg,Wimg] → [B,I,Dv]`, qua vision encoder và connector |
| `embed_language_tokens(tokens)` | int64 `[B,T] → [B,T,Dv]`, chỉ lookup embedding |
| `get_model_layers(models)` | lấy hai danh sách layer đã căn index; không phải forward qua transformer |
| `get_attention_interface()` | trả callable `eager_attention_forward` |
| property `eager_attention_forward` | cũng trả callable đó, giữ interface thuận tiện |

Khi `train_expert_only=True`, VLM frozen/eval, nhưng **expert, state/action/time projections và action_out_proj vẫn train** theo mặc định. `.eval()` điều khiển mode, còn `requires_grad=False` điều khiển cập nhật weight; đây là hai việc khác nhau. Không bọc toàn bộ core trong `no_grad()` khi train: gradient còn cần đi qua phép tính VLM đến state projection trainable.

Nếu `train_expert_only=False`, helper vẫn freeze `lm_head`, text final norm và một số tail layers theo quy tắc trong `backbone.py`; không hiểu tùy chọn này là mở mọi weight VLM.

## 8. Core.forward: giao diện và kiểm tra, dòng 112–155

Đầu vào chính `inputs_embeds=[prefix,suffix]`. Mỗi stream đang có phải là tensor rank 3, có ít nhất một token; hai stream cùng batch/device, width lần lượt đúng `Dv`, `De`.

| Cách gọi | `inputs_embeds` | `position_ids` | `attention_mask` | Cache |
|---|---|---|---|---|
| train/full forward | `[B,P,Dv]`, `[B,S,De]` | `[B,P+S]` | bool `[B,P+S,P+S]` | `use_cache=False` |
| prefix prefill | `[B,P,Dv]`, `None` | `[B,P]` | bool `[B,P,P]` | `use_cache=True`, cache ban đầu None |
| denoise với cache | `None`, `[B,S,De]` | `[B,S]` | bool `[B,S,P+S]` | `use_cache=True`, prefix cache đã đầy |

Output luôn là `([prefix_out,suffix_out],past_key_values)`; vị trí stream không chạy được giữ `None`. Với full pass, output shapes vẫn `[B,P,Dv]`, `[B,S,De]`. Core không chiếu ra 32 và không tự tính loss.

Code tính `total` là số token **đang đưa vào**, `key_len=total+cached_length`. Vì thế cached denoise có Q length S nhưng tổng key length P+S ở mask giao diện. Pure cross layers bên trong chỉ lấy phần prefix keys từ mask này.

Các guard ngăn dùng cache sai: không đưa cache khi `use_cache=False`; cache đã đầy chỉ nhận `[None,suffix]`; prefill phải là `[prefix,None]`; suffix-only cross không có prefix/cache sẽ bị từ chối. Prefix-only không cache cũng được hỗ trợ.

`fill_kv_cache=use_cache and past_key_values is None` tạo `DynamicCache()` đúng lúc prefill.

## 9. Core.forward: căn layer và chọn self/cross, dòng 148–189

`models=[VLM.text_model,lm_expert]`. Nếu N=4, Ne=2:

```text
global index    0      1      2      3
VLM layer       V0     V1     V2     V3
expert layer    E0     None   E1     None
```

`None` nghĩa là expert hidden state đi qua nguyên trạng ở vòng đó; không nhân đôi expert và không tạo thêm weight. `num_expert_layers` dương phải chia hết N.

Mỗi global layer chạy **self-attention** nếu một trong các điều kiện đúng:

- Đang prefill cache.
- Không có suffix.
- `attention_mode` không phải cross.
- `self_attn_every_n_layers>0` và global index chia hết tham số này.

Còn lại chạy cross-attention. Với cấu hình dự án, index 0,2,…,14 là self; 1,3,…,15 là cross: **8 self + 8 cross** ở expert. Prefix vẫn tự attention trong cả hai loại layer. Mỗi global slot chọn một trong hai kiểu attention cho expert rồi dùng một FFN; không phải block chuẩn có self-attention rồi thêm cross-attention liên tiếp trong cùng slot.

### 9.1. Shape ở self-attention layer

Đặt `L=P+S`, `Cq=Hq×Dh=960`, `Ckv=Hkv×Dh=320`.

| Tensor/phép chiếu | VLM prefix | Expert suffix |
|---|---|---|
| hidden sau input RMSNorm | `[B,P,960]` | `[B,S,720]` |
| Q projection | `960→960` | `720→960` |
| Q sau reshape | `[B,P,15,64]` | `[B,S,15,64]` |
| K/V projection, mỗi cái | `960→320` | `720→320` |
| K/V sau reshape, mỗi cái | `[B,P,5,64]` | `[B,S,5,64]` |

Ghép theo token thành `Q[B,L,15,64]`, `K/V[B,L,5,64]`. RoPE quay Q/K, giữ shape. GQA lặp mỗi KV head 3 lần để khớp 15 Q heads.

```text
Q sau transpose             [B,15,L,64]
K sau GQA + transpose       [B,15,L,64]
Q @ Kᵀ / sqrt(64)           [B,15,L,L]
mask[:,None,:,:]            [B,1,L,L]   broadcast theo head
softmax × V                 [B,15,L,64]
transpose + flatten heads   [B,L,960]
```

Attention helper trả **list một phần tử** chứa tensor chung `[B,P+S,960]`, chưa qua `o_proj`. Khi cached denoise, Q chỉ gồm suffix nên attention score là `[B,15,S,P+S]`, output `[B,S,960]`.

**Expert rộng 720 nhưng Q rộng 960** vì giữ 15 heads × 64/head. Không lấy `720/15=48` làm Dh của expert; code không chọn geometry đó.

### 9.2. Shape ở cross-attention layer

Prefix vẫn tự attention với Q `[B,P,15,64]`, K/V `[B,P,5,64]`, score `[B,15,P,P]`, output `[B,P,960]`.

Expert đọc K/V của **cùng global VLM layer**, được tính từ normalized prefix đầu layer, không phải final hidden output của toàn VLM:

```text
prefix K sau RoPE           [B,P,5,64] → flatten [B,P,320]
expert cross k_proj         320→320    → [B,P,5,64]
prefix V                    [B,P,5,64] → flatten [B,P,320]
expert cross v_proj         320→320    → [B,P,5,64]
expert Q                    [B,S,720] → 720→960 → [B,S,15,64]
cross score sau GQA         [B,15,S,P]
cross attention output      [B,S,960]
expert o_proj               960→720 → [B,S,720]
```

Vậy cross `k_proj/v_proj` nhận **320**, khác expert self K/V nhận **720**. Nó chiếu lại VLM K/V, không chiếu trực tiếp raw prefix hidden 960. Cross query được đặt vị trí bắt đầu lại từ 0; key lấy từ VLM đã qua RoPE, không quay key thêm lần nữa ở đoạn này.

Khi full forward, helper trả `[prefix_att[B,P,960],suffix_att[B,S,960]]`. Khi cached denoise chỉ trả `[suffix_att[B,S,960]]`. Cross chỉ đọc prefix; không append suffix K/V vào cache của layer đó.

## 10. Core.forward: o_proj, residual, FFN và final norm, dòng 190–232

Hai nhánh nhận output attention nhưng dùng **weight riêng** cho `o_proj`, RMSNorm và MLP.

```text
h_l [B,L_i,D_i]
   │
   ├────────────── residual 1 ────────────┐
   ↓                                      │
input RMSNorm → attention → [B,L_i,960]    │
   → o_proj(960,D_i) [B,L_i,D_i] → cộng ──┘
   → h_att [B,L_i,D_i]
   │
   ├────────────── residual 2 ────────────┐
   ↓                                      │
post_attention RMSNorm → MLP               │
   → [B,L_i,D_i] → cộng ──────────────────┘
   → h_(l+1) [B,L_i,D_i]
```

Ở prefix, `L_i=P,D_i=960`; suffix là `S,720`. Theo công thức:

```text
h_att = h_l + o_proj(Attention(input_RMSNorm(h_l)))
h_(l+1) = h_att + MLP(post_attention_RMSNorm(h_att))
```

`input_layernorm` nằm trong attention helper; `o_proj` trở đi nằm trong `smolvla.py`. Đừng vẽ hai input norms vì đọc hai file thấy hai chỗ khác nhau.

`start/end` chỉ dùng cắt đúng token từ output attention: self chung thì prefix lấy `:P`, suffix lấy `P:P+S`; cross đã có hai tensor riêng nên mỗi nhánh bắt đầu từ 0. Nhánh `layer is None` giữ hidden state và bỏ qua projection/FFN.

`.to(weight.dtype)` chỉ đổi dtype để khớp weight, không đổi shape. `.clone()` lưu giá trị residual thứ hai; không phải layer học được hoặc một nhánh transformer mới.

FFN kiểu Llama có hai projection song song và một projection xuống:

```text
gate = SiLU(gate_proj(h))    [B,L_i,F_i]
up   = up_proj(h)            [B,L_i,F_i]
mlp  = down_proj(gate * up)  [B,L_i,D_i]
```

VLM: `960→2560→960`. Expert: `720→2048→720`; 2048 được tính bằng `get_intermediate_size(720)` trong action_expert.py, làm tròn lên bội số 256. Đây không phải chiều head hoặc chiều action.

Sau toàn bộ N vòng, core chạy final RMSNorm riêng: prefix `[B,P,960]`, suffix `[B,S,720]`, rồi trả list output và cache. Không chạy VLM vocabulary `lm_head` để sinh instruction/token ngôn ngữ.

## 11. SmolVLA.__init__ và state_proj, dòng 248–264

Constructor nhận một core đã có, kiểm tra `0<action_dim<=padded_dim` và horizon dương. Nó tạo:

| Projection | Map feature dự án | Weight shape của PyTorch |
|---|---|---|
| `embeddings.state_proj` | `32→960` | `[960,32]` |
| `embeddings.action_in_proj` | `32→720` | `[720,32]` |
| `embeddings.action_time_mlp_in` | `1440→720` | `[720,1440]` |
| `embeddings.action_time_mlp_out` | `720→720` | `[720,720]` |
| `action_out_proj` | `720→32` | `[32,720]` |

Các Linear trên có bias mặc định, shape `[out_features]`. Với `nn.Linear(in,out)`, weight lưu `[out,in]`, còn sơ đồ kiến trúc nên ghi hướng `in→out`.

`state_proj` là property trả lại `self.embeddings.state_proj`, không tạo projection thứ hai. State và action cùng dùng padded width 32 trong wrapper này; code không nhận hai padded widths riêng.

`self.horizon=50` dùng kiểm tra shape trong `training_loss`. `sample` lấy số token thực tế từ noise, không tự tạo noise và không tự kiểm tra bằng horizon. Caller nên luôn đưa `[B,50,32]` theo cấu hình đang học.

## 12. prefix(...) và suffix(...), dòng 266–270

Hai hàm chuyển lời gọi sang embeddings.py, nhưng cần vẽ toàn bộ quá trình ở mục 3.

`prefix(image,tokens,language_pad,state)` nhận:

- image float `[B,3,Himg,Wimg]`, đã resize/normalize phù hợp trước đó.
- tokens int64 `[B,T]` và language_pad bool `[B,T]`, **True=token hợp lệ**.
- state float `[B,d_s]`, `d_s<=32`; baseline robot có `d_s=6`.

Nó trả tuple:

```text
prefix             float [B,P,Dv]
prefix_pad         bool  [B,P]
prefix_boundaries  bool  [B,P]
```

Ghép theo thứ tự **ảnh → language → state**. Image/text embedding được nhân sqrt(Dv); state không có phép scale đó. State được pad rồi project thành **một token** `[B,1,Dv]`, không phải 6 hay 32 token. Prefix boundaries bằng 0, riêng state cuối bằng 1.

`suffix(x_t,time)` nhận `x_t[B,S,32]`, `time[B]`, trả `[B,S,De]`. Mỗi sample dùng một time chung cho mọi action token. Time embedding `[B,De]` được expand thành `[B,S,De]`, rồi concatenate với action embedding theo feature thành `[B,S,2De]`. Nó **không thêm token time**, nên suffix vẫn S=50 token. Embedding helper cũng hỗ trợ time `[B,S]`; flow-matching recipe trong training_loss dùng `[B]`.

## 13. SmolVLA.forward và mask, dòng 272–281

Đây là hàm dự đoán **velocity ở một x_t và một time**, chưa tích phân thành action.

```text
suffix = self.suffix(x_t,time)                         [B,S,De]
suffix_pad = ones                                     bool[B,S]
pad = cat(prefix_pad,suffix_pad, dim=1)                bool[B,P+S]
boundaries = cat(prefix_boundaries,suffix_pad, dim=1)   bool[B,P+S]
mask = block_mask(pad,boundaries)                      bool[B,P+S,P+S]
position_ids = positions(pad)                         int64[B,P+S]
(_,out),_ = core([prefix,suffix],...,use_cache=False)  out[B,S,De]
action_out_proj(out.float())                          float32[B,S,32]
```

`(_,out),_` bỏ prefix output và cache, chỉ lấy suffix hidden. `.float()` ép hidden về float32 trước output projection; đường chạy mặc định dùng head float32. Nếu tự đổi dtype toàn policy, cần giữ dtype head phù hợp với phép ép này.

### Mask hiện tại cho ai đọc ai?

`block_mask` tính group bằng cumsum(boundaries), cho phép query đọc key nếu `group_key<=group_query` và cả hai token hợp lệ. **True nghĩa là được đọc.**

Ví dụ hai token ảnh/text, một state, ba action:

```text
token       c0 c1 state a0 a1 a2
boundary     0  0   1    1  1  1
group        0  0   1    2  3  4

                key →
query ↓     c0 c1 state a0 a1 a2
c0           1  1   0    0  0  0
c1           1  1   0    0  0  0
state        1  1   1    0  0  0
a0           1  1   1    1  0  0
a1           1  1   1    1  1  0
a2           1  1   1    1  1  1
```

Vậy ảnh/text cùng một block bidirectional; chúng không đọc state/action. State đọc ảnh/text và chính nó, không đọc action. Ở joint self layers, action j đọc toàn bộ prefix hợp lệ và action `0..j`. Các action được tính đồng thời bằng ma trận attention nhưng bị giới hạn quan hệ đọc theo causal mask.

Ở **cross layer**, expert chỉ lấy phần mask suffix→prefix; không có suffix→suffix attention trong layer đó. Thông tin giữa action tokens được trao đổi ở các self layers xen kẽ.

`suffix_pad` toàn True nghĩa là mọi suffix token được coi là có mặt trong attention. `action_is_pad` ở loss là một mask khác; nó không được truyền vào core ở đây.

## 14. training_loss(...), dòng 283–303

Đầu vào actions phải **đã normalize và pad** thành `[B,horizon,padded_dim]=[B,50,32]`. Hàm không tự pad action 6→32; việc tự pad trong prefix chỉ áp dụng state.

Nếu chưa cung cấp:

```text
noise ε ~ Normal(0,1)                    [B,50,32]
z ~ Beta(1.5,1.0)                       [B]
time t = 0.999 z + 0.001                [B]
t[:,None,None]                          [B,1,1]
```

Convention `NOISE_AT_ONE`:

```text
x_t = t*ε + (1-t)*actions               [B,50,32]
target = ε-actions                      [B,50,32]
prediction = self(...,x_t,t)             [B,50,32]
losses = (prediction-target)^2           [B,50,32]
losses = losses[...,:6]                  [B,50,6]
```

Model học **velocity**, không lấy `prediction` so trực tiếp với clean actions. Khi t=1, x là noise; t=0, x là clean action.

Nếu không có timestep padding: `losses.mean()` trả scalar, tensor shape `[]`. Nếu có `action_is_pad: bool[B,50]`, **True nghĩa là timestep padding**, ngược ý nghĩa `prefix_pad` True=valid:

```text
valid = ~action_is_pad                  [B,50]
valid[...,None]                         [B,50,1]
loss = sum(losses*valid[...,None]) / max(valid.sum()*6,1)
```

Như vậy 26 feature padding không tính loss; các timestep không có action thật cũng không tính loss. Nếu toàn bộ timestep padding và các giá trị đầu vào hữu hạn, loss bằng 0, vẫn có graph để backward. Mask loss không tự xóa các token đó khỏi attention.

32 chiều vẫn tồn tại ở input/output model; noise vẫn được sample cho cả 32. Không hiểu “bỏ feature padding khỏi loss” là model chỉ còn một tensor 6 chiều bên trong.

## 15. sample(...): Euler không cache, dòng 305–311

`@torch.no_grad()` tắt ghi graph trong hàm sample. Caller vẫn cần `policy.eval()` để đặt mode inference. Prefix được chuẩn bị ở bên ngoài; nên chuẩn bị prefix trong `with torch.no_grad()` khi inference.

Hàm nhận noise `[B,50,32]` từ caller. Với `use_cache=False`, mỗi Euler step gọi lại toàn bộ `self(prefix,...,x_t,time)`, tức chạy lại cả prefix text layers và suffix; image encoder/lookup chỉ chạy lúc dựng prefix trước vòng lặp.

Convention noise ở 1:

```text
x = noise                               [B,50,32]
dt = -1/10 = -0.1
t gọi model = 1.0,0.9,...,0.1           mỗi lần là [B]
v = model(prefix,x,t)                    [B,50,32]
x = x + dt*v                            [B,50,32]
sau 10 bước đạt mốc t=0
return x[...,:6]                        [B,50,6]
```

Không cần model call tại t=0 để hoàn thành bước Euler cuối. Số **10** là số lần đánh giá cùng một mạng velocity; không phải 10 mạng khác weight. Số **50** là số action tương lai trong mỗi lần đánh giá; không phải số bước Euler.

Output còn normalized; cần inverse-normalize theo action statistics trước khi dùng làm lệnh robot.

## 16. sample(...): prefix cache, dòng 312–336

### Bước A: prefill một lần

```text
inputs_embeds = [prefix,None]
mask          = [B,P,P]
positions     = [B,P]
use_cache     = True
```

Tất cả global layers dùng đường self-attention để tạo K/V prefix. Ở mỗi layer, DynamicCache lưu riêng:

```text
K_cache [B,Hkv,P,Dh] = [B,5,P,64]
V_cache [B,Hkv,P,Dh] = [B,5,P,64]
```

Cache giữ 5 KV heads, chưa repeat thành 15. Có N cặp K/V theo N global VLM layers, không phải một cặp dùng cho tất cả layer. Đây là intermediate của một observation/prefix cụ thể, không phải parameter học được; observation đổi thì cần prefix cache mới.

### Bước B: denoise(x_t,time) trong từng bước Euler

Tạo suffix mới `[B,S,720]`. Prefix không đổi, nhưng x_t/time đổi nên không dùng lại hidden/K/V suffix từ bước trước.

```text
prefix_pad[:,None,:]                    [B,1,P]
.expand(-1,S,-1)                        [B,S,P]
block_mask(suffix_pad,suffix_pad)        [B,S,S] lower triangular
cat hai phần theo key                   [B,S,P+S]

prefix_pad.sum(-1)[:,None]               [B,1]
positions(suffix_pad)                   [B,S]
suffix_positions = tổng hai phần        [B,S]
```

Offset dùng **số prefix token hợp lệ từng sample**, không phải P bao gồm padding. Nếu P=113 nhưng chỉ 105 token hợp lệ, suffix position bắt đầu ở 105; cache vẫn dài 113 slot và prefix_pad sẽ che 8 slot padding.

Gọi core `[None,suffix]` với cache. Ở self layers:

```text
Q_suffix                                [B,S,15,64]
cached prefix K/V + current suffix K/V   [B,P+S,5,64]
score                                   [B,15,S,P+S]
output sau o_proj                       [B,S,720]
```

Ở cross layers chỉ đọc prefix cache và chiếu lại K/V: score `[B,15,S,P]`. Sau core, output head vẫn trả velocity `[B,S,32]` để solver cập nhật x.

### Bước C: crop cache trong finally

Self layers đã append S token, cache tạm dài P+S. Cross layers chỉ đọc, vẫn dài P. `crop_prefix_cache(cache,P)` xóa phần suffix mới append ở **từng layer**, đưa mọi layer về đúng P.

Nếu không crop, bước tiếp theo sẽ đọc cả suffix của mức nhiễu cũ, mask/key length sai và thuật toán thay đổi. `finally` đảm bảo dọn cache kể cả khi forward gặp lỗi. P để crop là số vị trí prefix thật trong tensor, không phải số token hợp lệ.

## 17. Cách ghép và gọi đúng

Sau khi có `core` theo cấu hình muốn học:

```python
policy = SmolVLA(core, action_dim=6, padded_dim=32, horizon=50)

# TRAIN — image/state/actions đã được normalize từ trước.
policy.train()
prefix, pad, boundaries = policy.prefix(image, tokens, language_pad, state)
actions32 = F.pad(actions6, (0, 26))  # [B,50,6] → [B,50,32]
loss = policy.training_loss(prefix, pad, boundaries, actions32,
                            action_is_pad=action_is_pad)
loss.backward()
# optimizer.step() do training loop ở bên ngoài thực hiện.

# INFERENCE
policy.eval()
with torch.no_grad():
    prefix, pad, boundaries = policy.prefix(image, tokens, language_pad, state)
    noise = torch.randn(image.shape[0], 50, 32, device=image.device)
    action6_normalized = policy.sample(prefix, pad, boundaries, noise,
                                       num_steps=10, use_cache=True)
# action6_normalized: [B,50,6]; inverse-normalize trước khi dùng.
```

Prefix, noise và policy cần cùng device. Ví dụ đầy đủ tạo tiny core offline nằm ở `examples/smolvlm_expert_steps.py`. Baseline hiện tại nhận một camera và input đã xử lý; không có dataset trainer, optimizer loop, action queue, RTC hay giao tiếp robot trong file này.

## 18. Bảng shape để ghi cạnh sơ đồ

Ví dụ một ảnh 512×512, T=48, P=113, S=50:

| Vị trí | Shape |
|---|---|
| ảnh vào | `[B,3,512,512]` |
| sau vision encoder | `[B,1024,768]` |
| image tokens sau connector | `[B,64,960]` |
| language IDs / embedding | `[B,48]` / `[B,48,960]` |
| state thật / padded / token | `[B,6]` / `[B,32]` / `[B,1,960]` |
| prefix | `[B,113,960]` |
| prefix pad / boundaries | `[B,113]` / `[B,113]` |
| noisy actions / flow time | `[B,50,32]` / `[B]` |
| action embedding / broadcast time embedding | mỗi tensor `[B,50,720]` |
| ghép action-time | `[B,50,1440]` |
| suffix | `[B,50,720]` |
| full mask / positions | `[B,163,163]` / `[B,163]` |
| joint self Q / KV trước GQA | `[B,163,15,64]` / `[B,163,5,64]` |
| joint self score | `[B,15,163,163]` |
| cross expert score | `[B,15,50,113]` |
| expert FFN trung gian | `[B,50,2048]` |
| suffix output sau final norm | `[B,50,720]` |
| velocity output | `[B,50,32]` |
| feature loss trước reduce | `[B,50,6]` |
| scalar loss | `[]` |
| mỗi prefix K hoặc V trong cache | `[B,5,113,64]` |
| cached denoise mask / positions | `[B,50,163]` / `[B,50]` |
| cached self score | `[B,15,50,163]` |
| sampled action trước/sau cắt chiều | `[B,50,32]` / `[B,50,6]` |

## 19. Kiểm chứng shape bằng chạy thật

Đã chạy trên CPU với **tiny VLM/expert Transformers thật, random weights**, không tải pretrained 500M. Kết quả ghi tại `SMOLVLA_SHAPES_TINY.json` cùng thư mục.

```text
B=2, ảnh 16×16, patch=4, connector scale=2
Dv=32, De=24, Hq=4, Hkv=2, Dh=8, N=4, T=5
vision output trước connector  [2,16,16]
image tokens sau connector     [2,4,32]
prefix                        [2,10,32]
suffix                        [2,50,24]
full mask                     [2,60,60]
expert Q trước reshape        [2,50,32]    khác De=24
expert self K                 [2,50,16]
expert cross K input/output   [2,10,16]
expert FFN intermediate       [2,50,256]
velocity                      [2,50,32]
mỗi prefix K/V cache          [2,2,10,8]
sample 10 bước                [2,50,6]
loss                          scalar []
```

Action sau 10 bước với/không cache khớp trong tolerance đã kiểm tra; sai lệch tuyệt đối lớn nhất `3.5762786865234375e-07`. Loss hữu hạn, backward có gradient vào state projection và action output projection; VLM frozen không có parameter gradient.

Các shape 960/720/512 ở phần cấu hình dự án là **suy ra từ source/config**, chưa phải kết quả chạy checkpoint pretrained đầy đủ. Tiny run xác minh cách ghép tensor, mask, projection và solver, chưa đánh giá chất lượng điều khiển robot.
