#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Dump layer-by-layer reference taps from the official PyTorch GKD model.
# The C++ parity toolchain (scripts/parity_reference.py + gkd-cli --dump-taps)
# compares these float32 dumps against the ggml engine, tap by tap.
#
# Usage:
#   python3 scripts/dump_taps.py --image <img.jpg> \
#       [--support-image <img.jpg> --support-kps x1 y1 ...] \
#       [--kps-texts 'nose' 'left eye' ...] \
#       [--out-dir /tmp/gkd_taps]
# ------------------------------------------------------------------------------
import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from test_real_world.predefined_keypoints import get_prompt_info  # noqa: E402
from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, data_preprocess  # noqa: E402
from test_real_world.gkd_inference_lib import transforms as mytransforms  # noqa: E402
from test_real_world.gkd_inference_lib.gkd_model import visual_prompt_extraction  # noqa: E402
import torchvision.transforms as tv_transforms  # noqa: E402
from PIL import Image  # noqa: E402
import cv2  # noqa: E402


def dump(name, t, out_dir):
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    t = np.ascontiguousarray(t, dtype=np.float32)
    with open(os.path.join(out_dir, name + ".bin"), "wb") as f:
        f.write(np.asarray([len(t.shape)], dtype=np.int64).tobytes())  # ndims
        f.write(np.asarray(t.shape, dtype=np.int64).tobytes())
        f.write(t.tobytes())
    print(f"  tap {name}: shape={list(t.shape)} mean={t.mean():.6f} absmax={np.abs(t).max():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.path.join(ROOT, "test_real_world/configs/gkd.yaml"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "cpp_ggml/models/pytorch/gkd_fullset.best"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", type=int, nargs="*", default=[])
    ap.add_argument("--support-image", default="")
    ap.add_argument("--support-kps", type=int, nargs="*", default=[])
    ap.add_argument("--kps-texts", nargs="*", default=[])
    ap.add_argument("--out-dir", default="/tmp/gkd_taps")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    # ---------------- official preprocessing (identical to demo()) ----------
    infer = GKDInference(cfg_file=args.cfg, checkpoint_path=args.checkpoint)
    infer.gkd_model.eval()          # disable dropout (vanilla_detect does the same)
    torch.set_grad_enabled(False)
    square = infer.square_image_length
    preprocess = mytransforms.Compose([
        mytransforms.RandomCrop(crop_gt_bbox=True),
        mytransforms.Resize(longer_length=square),
        mytransforms.CenterPad(target_size=square),
        mytransforms.CoordinateNormalize(normalize_bbox=True),
    ])
    image_transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = cv2.imread(args.image)[:, :, [2, 1, 0]]
    image = Image.fromarray(image)
    w, h = image.size
    bbox = np.array(args.bbox, np.float64).reshape(-1, 4) if args.bbox else np.array([0, 0, w - 1, h - 1], np.float64).reshape(1, 4)
    queries, q_scale_trans, _ = data_preprocess(preprocess, image_transform, image, bbox, keypoints=[None] * bbox.shape[0])

    dump("q_img", queries, args.out_dir)
    dump("q_scale_trans", q_scale_trans, args.out_dir)

    supports, support_kps, support_kp_mask = None, None, None
    if args.support_image:
        sim = cv2.imread(args.support_image)[:, :, [2, 1, 0]]
        sim = Image.fromarray(sim)
        sw, sh = sim.size
        s_kps = np.array(args.support_kps, np.float64).reshape(-1, 2)
        s_kps_v = np.ones((s_kps.shape[0], 3), dtype=np.float64)
        s_kps_v[:, :2] = s_kps
        supports, _, s_kps_list = data_preprocess(preprocess, image_transform, sim, np.array([0, 0, sw - 1, sh - 1], np.float64).reshape(1, 4), keypoints=[s_kps_v.reshape(-1).tolist()])
        skp_full = torch.tensor(s_kps_list).reshape(1, -1, 3)
        support_kps, support_kp_mask = skp_full[:, :, :2], skp_full[:, :, 2].long()
        dump("s_img", supports, args.out_dir)
        dump("s_kps", support_kps, args.out_dir)
        dump("s_kp_mask", support_kp_mask, args.out_dir)

    kps_texts = list(args.kps_texts)
    kps_texts_mask = infer.supplement_mask_for_kps_texts(kps_texts)
    print(f"==> N_t={len(kps_texts)} N_v={0 if support_kp_mask is None else support_kp_mask.shape[1]}")

    # ---------------- model forward, tap by tap -----------------------------
    m = infer.gkd_model
    B2 = queries.shape[0]
    queries5 = queries.unsqueeze(0)  # S x B2 x C x H x W
    if supports is not None:
        supports5 = supports.unsqueeze(0)
        support_kps5 = support_kps.unsqueeze(0)
        support_kp_mask5 = support_kp_mask.unsqueeze(0)
        B1 = supports.shape[0]
        in_ims = torch.cat([supports5, queries5], dim=1).reshape(B1 + B2, 3, square, square)
    else:
        supports5, support_kps5, support_kp_mask5 = None, None, None
        B1 = 0
        in_ims = queries5.reshape(B2, 3, square, square)
    in_ims = in_ims.cuda()
    dump("in_ims", in_ims.cpu(), args.out_dir)

    if len(kps_texts) > 0:
        kps_texts = [kps_texts]
        kps_texts_mask = kps_texts_mask.unsqueeze(0)

    pe = m.dinov3_visual_encoder.patch_embed(in_ims)  # B x H x W x D (flatten_embedding=False)
    dump("vis_patch_embed", pe[0].flatten(0, 1), args.out_dir)  # [(24*24), D] token-major

    # 1) text tokenize + encode
    if len(kps_texts) > 0:
        in_texts_tokens = m.dinov3_text_tokenizer.tokenize(kps_texts[0]).cuda()
        dump("tok_ids", in_texts_tokens.cpu(), args.out_dir)
        out_texts_features = m.dinov3_text_encoder(in_texts_tokens, cls_token_only=False)
        dump("txt_tower_out", out_texts_features.cpu(), args.out_dir)  # T x 77 x 2048
        # anet internals (LND seq-first) for stage-wise parity
        anet_m = m.text_anet
        hooks = []
        def _mk_hook(name):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                dump(name, o.detach().cpu(), args.out_dir)
            return h
        def _mk_pre(name):
            def h(mod, inp):
                dump(name, inp[0].detach().cpu(), args.out_dir)
            return h
        blk0 = anet_m.net.resblocks[0]
        hooks.append(anet_m.proj_in.register_forward_hook(_mk_hook("anet_s1_proj_in")))
        hooks.append(blk0.ln_1.register_forward_hook(_mk_hook("anet_s2_ln1")))
        hooks.append(blk0.attn.register_forward_hook(_mk_hook("anet_s3_attn")))
        hooks.append(blk0.mlp.register_forward_hook(_mk_hook("anet_s4_mlp")))
        hooks.append(blk0.register_forward_hook(_mk_hook("anet_s5_block")))
        hooks.append(anet_m.ln.register_forward_hook(_mk_hook("anet_s6_ln")))
        hooks.append(anet_m.last_norm.register_forward_pre_hook(_mk_pre("anet_s7_proj")))
        anet_out = m.text_anet(out_texts_features)
        for h in hooks:
            h.remove()
        dump("anet_out", anet_out.cpu(), args.out_dir)
        cls = m.dinov3_text_encoder.get_cls_tokens(anet_out, in_texts_tokens)
        dump("txt_cls_full", cls.cpu(), args.out_dir)
        cls = cls[:, cls.shape[1] // 2:]
        dump("txt_cls_half", cls.cpu(), args.out_dir)
        out_texts_features_CLS = m.t2i_projector(cls)
        dump("txt_proto", out_texts_features_CLS.cpu(), args.out_dir)

    # 2) visual encoder
    vis = m.dinov3_visual_encoder
    feats = {}
    def pre_hook(mod, args):
        feats["vis_tokens_in"] = args[0].detach()
    def mk_hook(name):
        def hook(mod, inp, out):
            feats[name] = out.detach()
        return hook
    vis.blocks[0].register_forward_pre_hook(pre_hook)
    vis.blocks[0].register_forward_hook(mk_hook("vis_blk0"))
    vis.blocks[1].register_forward_hook(mk_hook("vis_blk1"))
    vis.blocks[23].register_forward_hook(mk_hook("vis_blk23"))
    tokens = m.dinov3_visual_encoder.get_intermediate_layers(in_ims, n=1, return_class_token=True, return_extra_tokens=True)
    for name in ["vis_tokens_in", "vis_blk0", "vis_blk1", "vis_blk23"]:
        t = feats[name][0]  # first batch item [581, D]
        dump(name, t, args.out_dir)
    vis_tokens = tokens[0][0]
    dump("vis_tokens", vis_tokens.cpu(), args.out_dir)  # (B1+B2) x 576 x D
    C = vis_tokens.shape[-1]
    fw = int(np.sqrt(vis_tokens.shape[1]))
    vis_map = vis_tokens.permute(0, 2, 1).reshape(-1, C, fw, fw).contiguous()
    dump("vis_map", vis_map.cpu(), args.out_dir)  # (B1+B2) x D x fw x fw

    # 3) visual prompt extraction
    prompt_set = {"text": [], "image": []}
    prompt_mask = {"text": [], "image": []}
    N_v_real = support_kp_mask5.shape[2] if B1 > 0 else 0
    if B1 > 0:
        support_features = vis_map[0:B1]
        repres = visual_prompt_extraction(m.cfg, m.visual_prompt_extraction_type, support_features, support_kps5[0].cuda(), support_kp_mask5[0].cuda(), square)
        dump("vis_repres", repres.cpu(), args.out_dir)  # B1 x C x N_v
        from network.models_gridms2 import average_representations2
        avg = average_representations2(repres, support_kp_mask5[0].cuda())
        dump("vis_proto", avg.transpose(1, 0).cpu(), args.out_dir)  # N_v x C
        prompt_mask["image"] = (support_kp_mask5[0].sum(dim=0) > 0).long()
    N_t_raw = len(kps_texts[0]) if (len(kps_texts) > 0 and isinstance(kps_texts[0], (list, tuple))) else len(kps_texts)
    if N_t_raw > 0:
        prompt_mask["text"] = torch.ones(N_t_raw, dtype=torch.long)

    # Official PAD_KPS.SAME_AT_TEST=True: prompts are padded to max_kps=80 before
    # the model forward (padded text rows become zero-protos with mask 0).
    N_pad = 80
    N_t_pad = max(N_t_raw, N_pad) if N_t_raw > 0 else 0
    prompt_list, mask_list = [], []
    if N_t_raw > 0:
        prompt_list.append(out_texts_features_CLS.cpu())
        prompt_list.append(torch.zeros(N_t_pad - N_t_raw, out_texts_features_CLS.shape[1]))
        mask_list.append(torch.cat([torch.ones(N_t_raw, dtype=torch.long),
                                    torch.zeros(N_t_pad - N_t_raw, dtype=torch.long)]))
    if B1 > 0:
        prompt_list.append(avg.transpose(1, 0).cpu())
        prompt_list.append(torch.zeros(N_pad - N_v_real, avg.shape[0]))
        mask_list.append(torch.cat([prompt_mask["image"].cpu(),
                                    torch.zeros(N_pad - N_v_real, dtype=torch.long)]))
    prompt_combined = torch.cat(prompt_list, dim=0)
    prompt_mask_combined = torch.cat(mask_list, dim=0)
    dump("prompts", prompt_combined.cpu(), args.out_dir)      # N x C (pre-pad, pre-mask-token)
    dump("prompt_mask", prompt_mask_combined.cpu(), args.out_dir)

    # pad to max_kps exactly like vanilla_detect
    from test_real_world.gkd_inference_lib.gkd_inference import pad_kps
    s_kp5 = support_kps5 if B1 > 0 else None
    msk5 = support_kp_mask5 if B1 > 0 else None
    s_kp5, msk5, kps_texts_p, kps_texts_mask_p = pad_kps(80, s_kp5, msk5, kps_texts if len(kps_texts) > 0 else [], kps_texts_mask if len(kps_texts) > 0 else None)
    N_t_pad = kps_texts_mask_p.shape[1] if kps_texts_mask_p is not None else 0
    N_v_pad = msk5.shape[2] if msk5 is not None else 0

    # build padded prompt tensor (N_t + N_v rows): text protos then visual protos
    prompt_pad = prompt_combined.unsqueeze(0).expand(B2, *prompt_combined.shape)
    prompt_mask_pad = prompt_mask_combined.unsqueeze(0).expand(B2, prompt_mask_combined.shape[0])
    context = vis_map[B1:].reshape(B2, C, -1).permute(0, 2, 1).cuda()  # B2 x (h*w) x C
    dump("context", context.cpu(), args.out_dir)

    # kg transformer internals for stage-wise parity (batch-first B x N x C)
    kgm = m.kg_transformer
    kghooks = []
    def _mk(name):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            dump(name, o.detach().cpu(), args.out_dir)
        return h
    kgb0 = kgm.blocks[0]
    kghooks.append(kgb0.norm1.register_forward_hook(_mk("kg_s1_sa_in")))
    kghooks.append(kgb0.self_attention.register_forward_hook(_mk("kg_s2_sa_out")))
    kghooks.append(kgb0.norm_ca1.register_forward_hook(_mk("kg_s3_ca_q")))
    kghooks.append(kgb0.norm_ca2.register_forward_hook(_mk("kg_s4_ca_kv")))
    kghooks.append(kgb0.cross_attention.register_forward_hook(_mk("kg_s5_ca_out")))
    kghooks.append(kgb0.feed_forward.register_forward_hook(_mk("kg_s6_ffn")))
    kernels = m.kg_transformer(prompt_pad.cuda(), context, prompt_mask_pad.cuda())
    for h in kghooks:
        h.remove()
    dump("kernels", kernels.cpu(), args.out_dir)  # B2 x N x C

    heatmaps = m.detection_head(vis_map[B1:].cuda(), kernels)
    dump("heatmaps_all", heatmaps.cpu(), args.out_dir)  # B2 x N x 96 x 96

    heatmaps_set = {"text": [], "image": []}
    cnt = 0
    if N_t_pad > 0:
        heatmaps_set["text"] = heatmaps[:, cnt:cnt + N_t_pad]
        cnt += N_t_pad
    if B1 > 0:
        heatmaps_set["image"] = heatmaps[:, cnt:cnt + N_v_pad]
        cnt += N_v_pad
    fused, fused_mask_sum, _, _ = m.openkd_heatmap_fuse(
        heatmaps_set,
        support_kp_mask=msk5[0].cuda() if B1 > 0 else None,
        kps_texts_mask=kps_texts_mask_p[0].cuda() if N_t_pad > 0 else None,
    )
    # slice away padded prompts like remove_pad_kps does
    fused = fused[:, :N_t_raw + N_v_real]
    fused_mask_sum = fused_mask_sum[:, :N_t_raw + N_v_real]
    dump("heatmaps_fused", fused.cpu(), args.out_dir)
    dump("fused_mask", fused_mask_sum.cpu(), args.out_dir)

    # decode (identical to vanilla_detect)
    fused = fused.cpu()
    fused_mask_sum = fused_mask_sum.cpu()
    B2_, _, Hh, Ww = fused.shape
    score, grid = torch.max(fused.reshape(B2_, -1, Hh * Ww), 2)
    gxy = torch.FloatTensor(B2_, fused.shape[1], 2)
    gxy[:, :, 0] = grid % Ww
    gxy[:, :, 1] = grid // Hh
    pred = ((gxy + 0.5) / Hh - 0.5) * 2
    valid = (fused_mask_sum > 0).long()
    pred = pred * valid.view(1, -1, 1)
    score = score * valid
    dump("pred_norm", pred.cpu(), args.out_dir)
    dump("pred_score", score.cpu(), args.out_dir)

    pred_o = mytransforms.recover_kps(pred.cpu(), square, q_scale_trans)
    dump("pred_orig", pred_o.cpu(), args.out_dir)
    print(f"==> taps dumped to {args.out_dir}")


if __name__ == "__main__":
    main()
