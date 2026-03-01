<!-- .github/copilot-instructions.md - guidance for AI coding agents -->

# Quick orientation

This repo implements a selective sensor-fusion visual-inertial odometry (VIO) training pipeline in PyTorch.
Focus your edits around the small set of core modules: dataset loading, encoders, fusion, temporal module, training loop, and utilities.

Key entrypoints
- `train.py` — main training and validation loop, argument parsing, checkpoint saving, and loss plotting.
- `data_loader_euroc.py` — `EuRoC_Loader` that expects an Euroc-style folder (examples in `dataset/`). See `euroc_collate` for the batch shape.
- `models.py` — `VisionEncoder`, `ImuEncoder`, `FusionModule`, `TemporalModule`, `PoseRegressor` (core model pieces).
- `utils.py` — helpers (path/formatting, transforms between pose/rotation representations, checkpoint helpers).
- `logger.py` — terminal progress UI and `AverageMeter` used by existing code.

Data flow / architecture summary
- Input: image sequences + IMU segments. Images are loaded via OpenCV (`cv2`) in `EuRoC_Loader` and transformed with `custom_transforms` before batching.
- Per timestep: `VisionEncoder(tgt, ref)` consumes two images concatenated into a 6-channel tensor (target + reference). `ImuEncoder` consumes averaged IMU samples for the frame interval.
- `FusionModule` uses a small attention-like block (query/key/value + residual) to fuse vision and IMU features.
- `TemporalModule` is a transformer encoder (global average pooling after encoding) that produces a sequence representation.
- `PoseRegressor` outputs 7 values: 3 translation and a 4-component quaternion — quaternion is L2-normalized in `forward()`.

Important implementation details (do not change lightly)
- `VisionEncoder` uses a pretrained ResNet18; the initial conv is modified to accept 6 channels by copying the 3-channel weights into channels 0-2 and 3-5.
- `EuRoC_Loader.build_sequences()` creates samples with per-frame averaged IMU segments (shape: [1, 6]) — the training code expects that shape.
- `euroc_collate()` returns three lists (one per timestep):
  - images: list of length seq_len where each element is a tensor [batch, C, H, W]
  - imus: list of length seq_len where each element is a tensor [batch, 1, 6]
  - poses: list of length seq_len where each element is a tensor [batch, 3, 4]
- Checkpoint format: `checkpoints/run_<timestamp>/best_model.pth` contains a dict with keys: `'vision','imu','fusion','temporal','pose'` (each a state_dict).

Common developer tasks and examples
- Train locally (example):
  python train.py dataset/vicon_room2/V2_02_medium/mav0 --sequence-length 3 --epochs 100 --batch-size 4
  - `train.py` auto-selects CUDA if available (via torch.device).
- Inspect dataset loader quickly by running a small script that imports `EuRoC_Loader` and prints `len(loader)` or a single sample shape.

Patterns & conventions
- Minimal, explicit modules: each encoder/fusion/temporal block lives in `models.py` and returns plain tensors (no side effects). Prefer small, local changes rather than large refactors.
- Utility functions that convert between pose formats (euler, quat, 3x4 matrices) live in `utils.py`. Use those helpers to keep numeric conventions consistent.
- Logging/UI: the repository uses `logger.TermLogger` (blessings + progressbar). Avoid replacing terminal UI without preserving existing AverageMeter behavior.

Dependencies (discoverable from imports)
- torch, torchvision, numpy, scipy, pandas, opencv (cv2), matplotlib, blessings, progressbar, pillow, path.
  If you add new top-level dependencies, update repository documentation and mention why (rare for experimental models).

Edge cases & gotchas
- OpenCV loads images as BGR (`cv2.imread`) while normalization in `custom_transforms` assumes a particular channel order; preserve that pipeline when changing preprocessing.
- The first convolution weight copy in `VisionEncoder` is subtle — if swapping backbone or freezing layers, ensure consistency of weight shapes.
- IMU segments sometimes are empty and are replaced with zeros (see `EuRoC_Loader.load_data()`); changes here affect stability of training.

If you edit or extend training:
- Keep the checkpoint keys consistent (`vision`,`imu`,`fusion`,`temporal`,`pose`) so existing evaluation scripts remain compatible.
- When changing the model output (pose dimension/rotation mode), update `train.py`'s `compute_trans_pose` and loss computation accordingly.

Where to look first for related functionality
- Small utilities: `utils.py`
- Data: `data_loader_euroc.py` and `dataset/` sample folders
- Model anatomy: `models.py`
- Training loop and loss: `train.py`
- Terminal UX: `logger.py`

Questions for a human reviewer (when in doubt)
- Should new features preserve the current checkpoint key names and `best_model.pth` layout? (recommended: yes)
- Confirm expected image channel ordering (BGR vs RGB) for downstream evaluation code.

If anything here is incomplete or unclear, point me at the file or example you want expanded and I will update these instructions.
