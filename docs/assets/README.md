# Figures and Rollout Media

The project README includes visible figure/video placeholders with commented HTML image elements. Add the files below, then replace each placeholder with the corresponding image element. No missing image is loaded until its element is uncommented.

| Asset path, relative to the repository root | Content |
| --- | --- |
| `docs/assets/model_architecture.png` | Architecture diagram for this implementation |
| `docs/assets/metrics/train_epoch.png` | TensorBoard `loss/train_epoch` |
| `docs/assets/metrics/val_epoch.png` | TensorBoard `loss/val_epoch` |
| `docs/assets/metrics/train_step.png` | TensorBoard `loss/train_step` |
| `docs/assets/metrics/best_val.png` | TensorBoard `loss/best_val` |
| `docs/assets/metrics/learning_rate.png` | TensorBoard `learning_rate/step` |
| `docs/assets/metrics/grad_norm.png` | TensorBoard `grad_norm/before_clip` |
| `docs/assets/rollouts/nominal_thumbnail.jpg` | Thumbnail linking to a nominal-task video |
| `docs/assets/rollouts/variation_thumbnail.jpg` | Thumbnail linking to a variation or failure-analysis video |

Create `metrics/` and `rollouts/` when adding media. The supplied screenshots contain two views of the same learning-rate curve; use one of them. Export raw traces or disclose the smoothing setting. Use optimizer updates as the horizontal axis, and distinguish the running best-validation curve from per-epoch validation loss.

For videos, upload to GitHub attachments or a video host, replace `VIDEO_URL` with the actual URL, and add a thumbnail. A Markdown image pointing directly to an MP4 does not create an embedded GitHub video player. Keep failures available alongside successful demonstrations.
