# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
import os

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import loguru
import torch
import argparse
import einops
import imageio
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw, ImageFont


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    else:
        raise ValueError(f'{value} is not a valid boolean value')


from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from generate_custom_trajectory import generate_camera_trajectory_local
from generate import save_video, pose_to_input, parse_pose_string, pose_string_to_json

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def generate_infinite_video(args):
    """Generate an infinite-length video by sliding window stitching."""
    initialize_infer_state(args)

    task = 'i2v' if args.image_path else 't2v'
    enable_sr = args.sr

    # Build transformer version
    transformer_version = f'{args.resolution}_{task}'
    assert transformer_version == "480p_i2v"

    # Set dtype
    if args.dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Initialize pipeline
    print(f"Initializing video generation pipeline...")
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )

    # Parse pose commands for infinite motion
    print(f"Parsing motion commands: {args.pose}")
    motions = parse_pose_string(args.pose)
    print(f"Generated {len(motions)} motion steps")

    extra_kwargs = {}
    if task == 'i2v':
        extra_kwargs['reference_image'] = args.image_path

    enable_rewrite = args.rewrite
    if not args.rewrite:
        print("Warning: Prompt rewriting is disabled.")

    print(f"Starting infinite video generation...")
    print(f"  Window size: {args.video_length} frames")
    print(f"  Overlap: {args.overlap} frames")
    print(f"  Total segments: {args.num_segments}")

    all_frames = []  # To store all generated frames from segments
    current_motion_idx = 0  # Track current position in motion sequence

    # Generate each video segment
    for segment_idx in range(args.num_segments):
        print(f"\nGenerating segment {segment_idx + 1}/{args.num_segments}...")

        # Calculate start and end motion indices for current segment
        # Each segment needs (video_length - 1) // 4 + 1 latents
        num_latents = (args.video_length - 1) // 4 + 1
        segment_motions = []

        # Get motions for this segment, looping infinitely if we reach end
        for _ in range(num_latents - 1):  # We need num_latents -1 motions for num_latents latents
            if current_motion_idx >= len(motions):
                current_motion_idx = 0  # Loop back to start of motion sequence
            segment_motions.append(motions[current_motion_idx])
            current_motion_idx += 1

        # Generate camera trajectory for this segment
        poses = generate_camera_trajectory_local(segment_motions)

        # Convert to pose JSON
        intrinsic = [[969.6969696969696, 0.0, 960.0],
                     [0.0, 969.6969696969696, 540.0],
                     [0.0, 0.0, 1.0]]
        pose_json = {}
        for i, p in enumerate(poses):
            pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

        # Convert pose to input tensors
        viewmats, Ks, action = pose_to_input(pose_json, num_latents)

        # Generate video segment
        print(f"Generating {args.video_length} frames for segment {segment_idx + 1}...")
        out = pipe(
            enable_sr=enable_sr,
            prompt=args.prompt,
            aspect_ratio=args.aspect_ratio,
            num_inference_steps=args.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=args.video_length,
            negative_prompt=args.negative_prompt,
            seed=args.seed + segment_idx,  # Use different seed for each segment
            output_type="pt",
            prompt_rewrite=enable_rewrite,
            return_pre_sr_video=args.save_pre_sr_video,
            viewmats=viewmats.unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            action=action.unsqueeze(0),
            few_step=args.few_step,
            chunk_latent_frames=4 if args.model_type == "ar" else 16,
            model_type=args.model_type,
            user_height=args.height,
            user_width=args.width,
            **extra_kwargs,
        )

        # Extract frames from generated video
        if enable_sr and hasattr(out, 'sr_videos'):
            segment_video = out.sr_videos
        else:
            segment_video = out.videos

        # Convert to frame list
        if segment_video.ndim == 5:
            assert segment_video.shape[0] == 1
            segment_video = segment_video[0]

        # Add to all_frames, skipping overlap to avoid duplication
        frames = einops.rearrange(segment_video, 'c f h w -> f c h w')

        if segment_idx == 0:
            # Keep all frames for first segment
            all_frames.extend(frames[:])
        else:
            # Skip overlapping frames from subsequent segments
            all_frames.extend(frames[args.overlap:])

        print(f"Segment {segment_idx + 1} completed. Total frames so far: {len(all_frames)}")

    # Combine all segments into final video
    print(f"\nCombining all {args.num_segments} segments...")
    final_video = torch.stack(all_frames, dim=1)

    # Save final infinite video
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    save_video_path = os.path.join(output_path, "infinite_video.mp4")
    save_video(final_video, save_video_path)
    print(f"Saved infinite video to: {save_video_path}")

    if enable_sr and args.save_pre_sr_video:
        print("Note: Save_pre_sr_video option only saves individual segments, not the final combined video.")

    print("\nInfinite video generation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate infinite-length video using HunyuanWorld-1.5 with sliding window')

    parser.add_argument("--pose", type=str, required=True,
                        help="Pose string (e.g., 'w-3, right-0.5, d-4') for continuous motion")
    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, required=True, choices=['480p', '720p'],
        help='Video resolution (480p or 720p)'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--action_ckpt', type=str, required=True,
        help='Path to pretrained action model'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=50,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--video_length', type=int, default=127,
        help='Number of frames per segment (default: 127)'
    )
    parser.add_argument(
        '--overlap', type=int, default=16,
        help='Number of overlapping frames between segments (default: 16)'
    )
    parser.add_argument(
        '--num_segments', type=int, default=5,
        help='Number of video segments to generate (default: 5)'
    )
    parser.add_argument(
        '--sr', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable super resolution (default: true)'
    )
    parser.add_argument(
        '--save_pre_sr_video', type=str_to_bool, nargs='?', const=True, default=False,
        help='Save original video before super resolution (default: false)'
    )
    parser.add_argument(
        '--rewrite', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable prompt rewriting (default: true)'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true)'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None)'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16)'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    parser.add_argument(
        '--output_path', type=str, default='./outputs_infinite',
        help='Output directory for generated video (default: ./outputs_infinite)'
    )
    parser.add_argument(
        '--few_step', type=str_to_bool, nargs='?', const=False, default=False,
        help='Enable few step mode (default: false)'
    )
    parser.add_argument(
        '--model_type', type=str, required=True, choices=['bi', 'ar'],
        help='Inference bidirectional or autoregressive model'
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='Height for generation (recommended to set as 480)'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='Width for generation (recommended to set as 832)'
    )

    args = parser.parse_args()

    # Validate arguments
    assert args.image_path is not None, "Must provide reference image for i2v mode"
    assert args.overlap >= 0, "Overlap must be non-negative"
    assert args.overlap < args.video_length, "Overlap can't be larger than video_length"
    assert args.num_segments >= 1, "Must generate at least one segment"

    generate_infinite_video(args)


if __name__ == '__main__':
    main()