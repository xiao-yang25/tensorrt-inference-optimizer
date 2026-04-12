#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import yaml


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BEVPool ranks/interval binaries for two-stage runtime."
    )
    parser.add_argument("--model-cfg", required=True, help="Path to model yaml (e.g. bevdet_lt_depth.yaml)")
    parser.add_argument("--cam-yaml", required=True, help="Path to sample camera yaml (sample0000.yaml)")
    parser.add_argument("--out-dir", required=True, help="Output directory for *.bin files")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = yaml.safe_load(Path(args.model_cfg).read_text())
    cam_yaml = yaml.safe_load(Path(args.cam_yaml).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = model_cfg["data_config"]["cams"]
    n_img = int(model_cfg["data_config"]["Ncams"])
    if n_img != len(cams):
        raise ValueError(f"Ncams mismatch: Ncams={n_img}, cams list={len(cams)}")

    input_h, input_w = map(int, model_cfg["data_config"]["input_size"])
    src_h, src_w = map(int, model_cfg["data_config"]["src_size"])
    crop_h, crop_w = map(int, model_cfg["data_config"]["crop"])
    down_sample = int(model_cfg["model"]["down_sample"])
    depth_start, depth_end, depth_step = map(float, model_cfg["grid_config"]["depth"])
    x_start, x_end, x_step = map(float, model_cfg["grid_config"]["x"])
    y_start, y_end, y_step = map(float, model_cfg["grid_config"]["y"])
    z_start, z_end, z_step = map(float, model_cfg["grid_config"]["z"])

    feat_h = input_h // down_sample
    feat_w = input_w // down_sample
    depth_num = int((depth_end - depth_start) / depth_step)
    xgrid_num = int((x_end - x_start) / x_step)
    ygrid_num = int((y_end - y_start) / y_step)
    zgrid_num = int((z_end - z_start) / z_step)

    resize_ratio = float(input_w) / float(src_w)
    post_rot_inv = np.linalg.inv(
        np.array(
            [[resize_ratio, 0.0, 0.0], [0.0, resize_ratio, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    )
    post_trans = np.array([-crop_w, -crop_h, 0.0], dtype=np.float32)
    bev_origin = np.array([x_start, y_start, z_start], dtype=np.float32)
    bev_step = np.array([x_step, y_step, z_step], dtype=np.float32)

    num_points = n_img * depth_num * feat_h * feat_w
    ranks_depth = np.arange(num_points, dtype=np.int32)
    ranks_feat = np.zeros(num_points, dtype=np.int32)
    voxel_xyz = np.zeros((num_points, 3), dtype=np.int32)

    cam_rot = []
    cam_trans = []
    cam_intrin_inv = []
    for cam in cams:
        cam_data = cam_yaml["cams"][cam]
        cam_rot.append(quat_wxyz_to_rotmat(cam_data["sensor2ego_rotation"]))
        cam_trans.append(np.array(cam_data["sensor2ego_translation"], dtype=np.float32))
        cam_intrin_inv.append(np.linalg.inv(np.array(cam_data["cam_intrinsic"], dtype=np.float32)))

    offset = 0
    for cam_i in range(n_img):
        for d_i in range(depth_num):
            depth = d_i * depth_step + depth_start
            for h_i in range(feat_h):
                py = float(h_i) * float(input_h - 1) / float(feat_h - 1)
                for w_i in range(feat_w):
                    px = float(w_i) * float(input_w - 1) / float(feat_w - 1)
                    p = np.array([px, py, depth], dtype=np.float32)
                    p = post_rot_inv @ (p - post_trans)
                    p[0] *= p[2]
                    p[1] *= p[2]
                    p = cam_rot[cam_i] @ (cam_intrin_inv[cam_i] @ p) + cam_trans[cam_i]
                    p = (p - bev_origin) / bev_step
                    voxel_xyz[offset] = p.astype(np.int32)
                    ranks_feat[offset] = cam_i * feat_h * feat_w + h_i * feat_w + w_i
                    offset += 1

    valid_mask = (
        (voxel_xyz[:, 0] >= 0)
        & (voxel_xyz[:, 0] < xgrid_num)
        & (voxel_xyz[:, 1] >= 0)
        & (voxel_xyz[:, 1] < ygrid_num)
        & (voxel_xyz[:, 2] >= 0)
        & (voxel_xyz[:, 2] < zgrid_num)
    )
    kept_idx = np.nonzero(valid_mask)[0].astype(np.int32)

    kept_xyz = voxel_xyz[kept_idx]
    ranks_depth_kept = ranks_depth[kept_idx]
    ranks_feat_kept = ranks_feat[kept_idx]
    ranks_bev = (
        kept_xyz[:, 2] * xgrid_num * ygrid_num + kept_xyz[:, 1] * xgrid_num + kept_xyz[:, 0]
    ).astype(np.int32)

    order = np.argsort(ranks_bev, kind="quicksort")
    ranks_bev = ranks_bev[order]
    ranks_depth_kept = ranks_depth_kept[order]
    ranks_feat_kept = ranks_feat_kept[order]

    split_points = np.nonzero(np.diff(ranks_bev) != 0)[0] + 1
    interval_starts = np.concatenate(([0], split_points)).astype(np.int32)
    interval_ends = np.concatenate((split_points, [ranks_bev.shape[0]])).astype(np.int32)
    interval_lengths = (interval_ends - interval_starts).astype(np.int32)

    (out_dir / "ranks_depth.bin").write_bytes(ranks_depth_kept.tobytes())
    (out_dir / "ranks_feat.bin").write_bytes(ranks_feat_kept.tobytes())
    (out_dir / "ranks_bev.bin").write_bytes(ranks_bev.tobytes())
    (out_dir / "interval_starts.bin").write_bytes(interval_starts.tobytes())
    (out_dir / "interval_lengths.bin").write_bytes(interval_lengths.tobytes())

    print(f"out_dir={out_dir}")
    print(f"valid_feat_num={ranks_bev.shape[0]}")
    print(f"unique_bev_num={interval_lengths.shape[0]}")
    print("files:")
    for name in [
        "ranks_depth.bin",
        "ranks_feat.bin",
        "ranks_bev.bin",
        "interval_starts.bin",
        "interval_lengths.bin",
    ]:
        p = out_dir / name
        print(f"  {name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
