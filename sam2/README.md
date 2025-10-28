# sam2

This folder contains utilities and demo scripts related to instance segmentation and tracking workflows (likely using interactive prompts or frame-by-frame inference). 

Requirements (typical)
- Python 3.8+.
- Common libraries used by these scripts may include: torch, torchvision, opencv-python, numpy, imageio, tqdm. If the repository uses the Segment Anything Model (SAM) or another external model, install the corresponding package or follow the model's installation instructions.
- To discover exact requirements, run: `python3 -c "import <module>"` for modules you need or open each script to check the imports.


## Files

- `instance_tracking.py`
  - Purpose: Track instances across frames in a video or sequence of images. Likely assembles per-frame instance segmentation outputs into temporally consistent tracks (assigns persistent IDs across frames).
  - Typical behavior: loads per-frame detections/segmentations, applies a tracking algorithm (IoU matching, Hungarian assignment, or a simple centroid tracker), and writes a tracking output (e.g., JSON, CSV, or per-frame instance masks with stable IDs).
  - Run: `python3 sam2/instance_tracking.py --pred-dir <per-frame-masks> --out <tracks.json> [--min-iou 0.3]` (check script for exact flags).

- `instance-seg-point-prompt.py`
  - Purpose: Perform instance segmentation using point prompts (clicks) per image — commonly used with interactive models like SAM (Segment Anything Model). Given a point prompt and an image, it produces an instance mask for the selected object.
  - Typical behavior: loads an image, accepts one or more point prompts (x,y, positive/negative), queries the segmentation model, and saves or visualizes the resulting mask.
  - Run: `python3 sam2/instance-seg-point-prompt.py --image input.png --points "x1,y1:pos x2,y2:neg" --out mask.png` (open the script for the exact CLI format).

- `video-inference.py`
  - Purpose: Run segmentation or instance-model inference over a video (frame-by-frame) and optionally save per-frame outputs, overlays, or a result video. May include batching, tiling, or GPU support depending on implementation.
  - Typical behavior: opens a video file or image sequence, forwards each frame through the model, and writes outputs to an output directory or a new video file.
  - Run: `python3 sam2/video-inference.py --video input.mp4 --out-dir results --model <model-path-or-name>`.