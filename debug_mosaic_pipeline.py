#!/usr/bin/env python3
"""Debug helper: run lada's mosaic detection and restoration on the first N frames
of a video and dump per-frame images for inspection.

Outputs (all under --output):
  detections/frame_XXXX.jpg   original frame with detected boxes + masks drawn
  restored/frame_XXXX.jpg     frame after pipeline-style crop -> restore -> blend
  side_by_side/frame_XXXX.jpg original | restored for quick comparison
  crops_in/, crops_out/       what the restoration model actually sees / outputs
  summary.txt                 per-frame detection counts and clip layout

Usage:
  python debug_mosaic_pipeline.py input.mp4 output_dir [-n 48] [--fp16] [--device cuda]
"""

import argparse
import itertools
import os
import sys

import cv2
import numpy as np
import torch

from lada.cli.utils import ModelFiles
from lada.restorationpipeline import load_models
from lada.restorationpipeline.mosaic_detector import Clip, Scene
from lada.utils import image_utils, mask_utils, video_utils
from lada.utils.ultralytics_utils import convert_yolo_box, convert_yolo_mask_tensor


def build_scenes(results_per_frame, file_path, video_meta):
    """Replicates MosaicDetector scene handling: scenes are appended/merged per frame
    and completed as soon as a later frame no longer extends them."""
    scenes = []
    completed = []
    for frame_num, results in enumerate(results_per_frame):
        for i in range(len(results.boxes)):
            mask = convert_yolo_mask_tensor(results.masks[i], results.orig_shape).to(device=results.orig_img.device)
            box = convert_yolo_box(results.boxes[i], results.orig_shape)
            current_scene = None
            for scene in scenes:
                if scene.belongs(box):
                    if scene.frame_end == frame_num:
                        current_scene = scene
                        scene.merge_mask_box(mask, box)
                    else:
                        current_scene = scene
                        scene.add_frame(frame_num, results.orig_img, mask, box)
                    break
            if current_scene is None:
                current_scene = Scene(file_path, video_meta)
                scenes.append(current_scene)
                current_scene.add_frame(frame_num, results.orig_img, mask, box)
        done = [s for s in scenes if s.frame_end < frame_num]
        for s in done:
            scenes.remove(s)
            completed.append(s)
    completed.extend(scenes)
    return completed


def blend_into_frame(frame, clip_img, blend_mask, orig_clip_box):
    """Replicates FrameRestorer._blend_cpu (frames come from VideoReader as CPU tensors)."""
    t, l, b, r = orig_clip_box
    frame_roi = frame[t:b + 1, l:r + 1, :].numpy()
    clip_img = clip_img.cpu().numpy()
    mask = blend_mask.cpu().numpy()
    temp = np.empty_like(frame_roi, dtype=np.float32)
    np.subtract(clip_img, frame_roi, out=temp, dtype=np.float32)
    np.multiply(temp, mask[..., None], out=temp)
    np.add(temp, frame_roi, out=temp)
    frame_roi[:] = temp.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='input video file')
    parser.add_argument('output', help='output directory for debug images')
    parser.add_argument('-n', '--num-frames', type=int, default=48, help='number of frames to process from the start (default: 48)')
    parser.add_argument('--fp16', action='store_true', help='run models in fp16 (restoration and detection unless overridden)')
    parser.add_argument('--fp16-detection', action=argparse.BooleanOptionalAction, default=None, help='override fp16 for detection only (default: same as --fp16)')
    parser.add_argument('--device', default='cuda', help='device for detection/restoration (default: cuda)')
    parser.add_argument('--mosaic-detection-model', default='v4-fast')
    parser.add_argument('--mosaic-restoration-model', default='basicvsrpp-v1.2')
    parser.add_argument('--max-clip-length', type=int, default=180)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)
    for sub in ('detections', 'restored', 'side_by_side', 'crops_in', 'crops_out'):
        os.makedirs(os.path.join(args.output, sub), exist_ok=True)

    if os.path.isfile(args.mosaic_detection_model):
        detection_model_path = args.mosaic_detection_model
    else:
        detection_modelfile = ModelFiles.get_detection_model_by_name(args.mosaic_detection_model)
        detection_model_path = detection_modelfile.path if detection_modelfile else None
    if detection_model_path is None:
        sys.exit(f'No detection model found for "{args.mosaic_detection_model}" '
                 f'(not a name in LADA_MODEL_WEIGHTS_DIR and not a file path)')
    restoration_modelfile = ModelFiles.get_restoration_model_by_name(args.mosaic_restoration_model)
    if restoration_modelfile is None:
        sys.exit('Could not find requested restoration model file in LADA_MODEL_WEIGHTS_DIR')

    fp16_detection = args.fp16_detection if args.fp16_detection is not None else args.fp16
    detection_model, restoration_model, pad_mode = load_models(
        device, args.mosaic_restoration_model, restoration_modelfile.path, None,
        detection_model_path, args.fp16, fp16_detection, False)
    print(f'device={device} fp16_restoration={args.fp16} fp16_detection={fp16_detection} pad_mode={pad_mode} '
          f'restoration_dtype={restoration_model.dtype} detection_dtype={detection_model.dtype}')

    with video_utils.VideoReader(args.input) as reader:
        frames = [f for f, _ in itertools.islice(reader.frames(), args.num_frames)]
    print(f'read {len(frames)} frames')

    # --- detection (batched like the pipeline, batch_size=4) ---
    results_per_frame = []
    for i in range(0, len(frames), 4):
        batch = frames[i:i + 4]
        frames_batch = detection_model.preprocess(batch)
        results_per_frame.extend(detection_model.inference_and_postprocess(frames_batch, batch))

    summary = []
    total_boxes = 0
    for frame_num, results in enumerate(results_per_frame):
        n = len(results.boxes)
        total_boxes += n
        img = frames[frame_num].clone()
        for i in range(n):
            box = convert_yolo_box(results.boxes[i], results.orig_shape)
            mask = convert_yolo_mask_tensor(results.masks[i], results.orig_shape).squeeze(-1)
            t, l, b, r = box
            cv2.rectangle(img.numpy(), (l, t), (r, b), (255, 0, 255), 2)
            overlay = img.numpy()
            m = mask.cpu().numpy() > 0
            overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, 'detections', f'frame_{frame_num:04d}.jpg'), img.numpy())
        summary.append(f'frame {frame_num:04d}: {n} detection(s)')
    print(f'total detections: {total_boxes} across {sum(1 for r in results_per_frame if len(r.boxes))} frames')
    if total_boxes == 0:
        print('No mosaics detected - nothing to restore. Check detections/ to see what the model sees.')
        return

    # --- build clips exactly like MosaicDetector does ---
    video_meta = video_utils.get_video_meta_data(args.input)
    scenes = build_scenes(results_per_frame, args.input, video_meta)
    clips = [Clip(scene, 256, pad_mode, idx) for idx, scene in enumerate(scenes)]
    for clip in clips:
        summary.append(f'clip {clip.id}: frames {clip.frame_start}-{clip.frame_end} ({len(clip)} frames)')

    # --- restore crops ---
    for clip in clips:
        # save what the model gets as input (unpadded crops)
        for i in range(len(clip)):
            frame_idx = clip.frame_start + i
            crop_in = image_utils.unpad_image(clip.frames[i], clip.pad_after_resizes[i]).numpy()
            cv2.imwrite(os.path.join(args.output, 'crops_in', f'clip{clip.id:02d}_frame_{frame_idx:04d}.jpg'), crop_in)

        clip.frames = restoration_model.restore(clip.frames)

        # save what the model produced
        for i in range(len(clip)):
            frame_idx = clip.frame_start + i
            crop_out = image_utils.unpad_image(clip.frames[i], clip.pad_after_resizes[i]).cpu().numpy()
            cv2.imwrite(os.path.join(args.output, 'crops_out', f'clip{clip.id:02d}_frame_{frame_idx:04d}.jpg'), crop_out)

    # --- blend restored crops back into full frames (replicates FrameRestorer) ---
    restored_frames = [f.clone() for f in frames]
    for clip in clips:
        clip_iter = clip.frame_start
        for _ in range(len(clip)):
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2], interpolation=cv2.INTER_NEAREST)
            blend_mask = mask_utils.create_blend_mask(clip_mask.to(device=device).float()).to(
                device=clip_img.device, dtype=torch.float32)
            blend_into_frame(restored_frames[clip_iter], clip_img, blend_mask, orig_clip_box)
            clip_iter += 1

    for frame_num in range(len(frames)):
        orig = frames[frame_num].numpy()
        restored = restored_frames[frame_num].numpy()
        cv2.imwrite(os.path.join(args.output, 'restored', f'frame_{frame_num:04d}.jpg'), restored)
        cv2.imwrite(os.path.join(args.output, 'side_by_side', f'frame_{frame_num:04d}.jpg'),
                    np.hstack([orig, restored]))

    with open(os.path.join(args.output, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary) + '\n')
    print(f'done - results in {args.output}')


if __name__ == '__main__':
    main()
