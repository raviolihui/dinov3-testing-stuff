# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from pathlib import Path

from typing import Union

import torch
import torch.nn as nn

from dinov3.layers.fp8_linear import convert_linears_to_fp8

from . import vision_transformer as vits

logger = logging.getLogger("dinov3")


def init_fp8(model: nn.Module, args) -> nn.Module:
    if not args.fp8_enabled:
        logger.info("fp8 matmuls: OFF (disabled in config)")
        return model
    logger.info("fp8 matmuls: ON")
    # Multi-kernel makes Inductor auto-tune between a regular "streaming"-based
    # reduction kernel and a "persistent" reduction kernel. Since fp8 has some
    # multi-pass steps (e.g., first get amax, then scale), persistent kernels
    # should perform better.
    torch._inductor.config.triton.multi_kernel = 1
    return convert_linears_to_fp8(model, filter=args.fp8_filter)


def build_model(args, only_teacher=False, img_size=224, device=None):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            in_chans=args.in_chans,
            pos_embed_rope_base=args.pos_embed_rope_base,
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
            qkv_bias=args.qkv_bias,
            layerscale_init=args.layerscale,
            norm_layer=args.norm_layer,
            ffn_layer=args.ffn_layer,
            ffn_bias=args.ffn_bias,
            proj_bias=args.proj_bias,
            n_storage_tokens=args.n_storage_tokens,
            mask_k_bias=args.mask_k_bias,
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
            device=device,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        teacher = init_fp8(teacher, args)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim
        # If a pretrained checkpoint is provided, attempt to load it and adapt
        # 3-channel patch-embed weights to the requested `in_chans` by copying
        # the RGB weights into the first three channels and filling the extra
        # channels with the mean of the RGB filters.
        try:
            pretrained_path = getattr(args, "pretrained_weights", None)
        except Exception:
            pretrained_path = None
        if pretrained_path:
            try:
                p = Path(pretrained_path)
                if p.exists() and p.is_file():
                    logger.info(f"Loading pretrained weights from {pretrained_path} and adapting patch_embed for in_chans={args.in_chans}")
                    ckpt = torch.load(pretrained_path, map_location="cpu")
                    # some checkpoints store under a 'teacher' key
                    if isinstance(ckpt, dict) and "teacher" in ckpt and isinstance(ckpt["teacher"], dict):
                        ckpt = ckpt["teacher"]
                    # normalize keys (remove common prefixes)
                    norm_ckpt = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt.items()}

                    def adapt_and_load(model):
                        sd = model.state_dict()
                        # find any ckpt key that ends with the target state key
                        for key in list(sd.keys()):
                            if key.endswith("patch_embed.proj.weight"):
                                # search for a matching key in norm_ckpt
                                matched = None
                                for ck in norm_ckpt.keys():
                                    if ck.endswith("patch_embed.proj.weight"):
                                        matched = ck
                                        break
                                if matched is None:
                                    continue
                                old_w = norm_ckpt[matched]
                                if old_w.ndim == 4 and old_w.shape[1] == 3 and args.in_chans > 3:
                                    out, in_ch, kh, kw = old_w.shape
                                    new_w = torch.zeros((out, args.in_chans, kh, kw), dtype=old_w.dtype)
                                    # copy RGB
                                    new_w[:, :3, :, :] = old_w
                                    # fill remaining channels with mean of RGB
                                    mean_rgb = old_w.mean(dim=1, keepdim=True)  # shape [out,1,kh,kw]
                                    repeat_count = args.in_chans - 3
                                    new_w[:, 3:, :, :] = mean_rgb.repeat(1, repeat_count, 1, 1)
                                    # place into model state dict and load
                                    sd[key] = new_w
                                else:
                                    # if channels already match or not 3, try direct copy if shapes align
                                    try:
                                        if sd[key].shape == old_w.shape:
                                            sd[key] = old_w
                                    except Exception:
                                        pass
                        # load updated state (non-strict to avoid missing keys)
                        model.load_state_dict(sd, strict=False)

                    # adapt for both student and teacher if possible
                    try:
                        adapt_and_load(student)
                    except Exception:
                        logger.exception("Failed to adapt/load pretrained weights into student")
                    try:
                        adapt_and_load(teacher)
                    except Exception:
                        logger.exception("Failed to adapt/load pretrained weights into teacher")
                else:
                    logger.info(f"Pretrained path {pretrained_path} does not exist or is not a file; skipping pretrained load")
            except Exception:
                logger.exception("Error while loading/adapting pretrained weights; continuing with random init")
    else:
        raise NotImplementedError(f"Unrecognized architecture {args.arch}")
    student = init_fp8(student, args)
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher: bool = False):
    outputs = build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size
        if isinstance(cfg.crops.global_crops_size, int)
        else max(cfg.crops.global_crops_size),
        device="meta",
    )
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim


def build_model_for_eval(
    config,
    pretrained_weights: Union[str, Path] | None,
    shard_unsharded_model: bool = False,  # If the model is not sharded, shard it. No effect if already sharded on disk
):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is None or pretrained_weights == "":
        logger.info("No pretrained weights")
        model.init_weights()
    elif Path(pretrained_weights).is_dir():
        logger.info("PyTorch DCP checkpoint")
        from dinov3.checkpointer import load_checkpoint
        from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize

        moduledict = nn.ModuleDict({"backbone": model})
        # Wrap with FSDP
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
        # Move to CUDA
        model.to_empty(device="cuda")
        # Load checkpoint
        load_checkpoint(pretrained_weights, model=moduledict, strict_loading=True)
        shard_unsharded_model = False
    else:
        logger.info("PyTorch consolidated checkpoint")
        from dinov3.checkpointer import init_model_from_checkpoint_for_evals

        # consolidated checkpoint codepath
        model.to_empty(device="cuda")
        init_model_from_checkpoint_for_evals(model, pretrained_weights, "teacher")
    if shard_unsharded_model:
        logger.info("Sharding model")
        moduledict = nn.ModuleDict({"backbone": model})
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
    model.eval()
    return model
