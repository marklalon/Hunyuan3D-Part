# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from partgen.partformer_pipeline import PartFormerPipeline

import pytorch_lightning as pl
from partgen.utils.misc import get_config_from_file
import argparse
import numpy as np
import torch


def parse_torch_dtype(dtype: str):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def main(ckpt_path, config, mesh_path, save_dir, args, ignore_keys=()):
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    # pipeline = PartFormerPipeline.from_single_file(
    #     ckpt_path=ckpt_path,
    #     config=config,
    #     verbose=True,
    #     ignore_keys=ignore_keys,
    # )
    pipeline = PartFormerPipeline.from_pretrained(
        model_path="tencent/Hunyuan3D-Part",
        verbose=True,
    )
    pipeline.to(device=args.device, dtype=parse_torch_dtype(args.dtype))
    cfg = args.guidance_scale
    # for mesh paths
    uid = Path(mesh_path).stem
    additional_params = {
        "output_type": "trimesh",
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "dual_guidance_scale": args.dual_guidance_scale,
        "dual_guidance": args.dual_guidance,
        "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks,
        "cond_chunk_size": args.cond_chunk_size,
        "point_num": args.point_num,
        "prompt_num": args.prompt_num,
        "bbox_threshold": args.bbox_threshold,
        "bbox_post_process": args.bbox_post_process,
        "bbox_show_info": args.bbox_show_info,
        "bbox_clean_mesh_flag": args.bbox_clean_mesh_flag,
    }
    if args.mc_level is not None:
        additional_params["mc_level"] = args.mc_level

    obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
        mesh_path=mesh_path,
        **additional_params,
    )
    obj_mesh.export(save_dir / f"train_cfg_{cfg:04f}_dit_boxgpt_{uid}.glb")
    out_bbox.export(save_dir / f"train_cfg_{cfg:04f}_dit_boxgpt_{uid}_bbox.glb")
    mesh_gt_bbox.export(save_dir / f"train_cfg_{cfg:04f}_input_boxgpt_{uid}.glb")
    explode_object.export(save_dir / f"train_cfg_{cfg:04f}_explode_boxgpt_{uid}.glb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
    )
    # input_dir
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="./data/test.glb",
    )
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=-1.0)
    parser.add_argument("--dual_guidance_scale", type=float, default=10.5)
    parser.add_argument(
        "--dual_guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--octree_resolution", type=int, default=256)
    parser.add_argument("--num_chunks", type=int, default=20000)
    parser.add_argument("--mc_level", type=float, default=None)
    parser.add_argument("--cond_chunk_size", type=int, default=2)
    parser.add_argument("--point_num", type=int, default=50000)
    parser.add_argument("--prompt_num", type=int, default=200)
    parser.add_argument("--bbox_threshold", type=float, default=0.95)
    parser.add_argument(
        "--bbox_post_process",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bbox_show_info",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bbox_clean_mesh_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    config = get_config_from_file(
        args.config
        if args.config
        else str(Path(__file__).parent / "partgen/config" / "infer.yaml")
    )
    assert hasattr(config, "ckpt") or hasattr(
        config, "ckpt_path"
    ), "ckpt or ckpt_path must be specified in config"
    ckpt_path = Path(args.ckpt if args.ckpt else config.ckpt_path)
    ignore_keys = config.get("ignore_keys", [])
    save_dir = (
        Path(args.save_dir)
        / ckpt_path.parent.parent.relative_to(
            ckpt_path.parent.parent.parent.parent.parent
        )
        # / ckpt_path.stem
    )
    print(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    main(
        ckpt_path=ckpt_path,
        config=config,
        mesh_path=args.mesh_path,
        save_dir=save_dir,
        args=args,
        ignore_keys=ignore_keys,
    )
