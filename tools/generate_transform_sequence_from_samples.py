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
        description="Generate per-frame BEV geometric transform sequence from sample yaml files."
    )
    parser.add_argument("--model-cfg", required=True, help="Path to bevdet model yaml")
    parser.add_argument("--samples-dir", required=True, help="Directory containing sample*.yaml")
    parser.add_argument("--adj-num", type=int, default=8, help="Number of adjacent frames")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-frame *.bin transforms")
    return parser.parse_args()


def e2g_matrix(rot_wxyz, trans_xyz):
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = quat_wxyz_to_rotmat(rot_wxyz)
    t[:3, 3] = np.asarray(trans_xyz, dtype=np.float32)
    return t


def curr_to_adj_grid(curr_e2g, adj_e2g, x_start, x_step, y_start, y_step):
    curr_to_adj = np.linalg.inv(adj_e2g) @ curr_e2g
    curr2adj_2d = np.eye(3, dtype=np.float32)
    curr2adj_2d[:2, :2] = curr_to_adj[:2, :2]
    curr2adj_2d[0, 2] = curr_to_adj[0, 3]
    curr2adj_2d[1, 2] = curr_to_adj[1, 3]

    grid_to_ego = np.eye(3, dtype=np.float32)
    grid_to_ego[0, 0] = x_step
    grid_to_ego[1, 1] = y_step
    grid_to_ego[0, 2] = x_start
    grid_to_ego[1, 2] = y_start
    ego_to_grid = np.linalg.inv(grid_to_ego)
    return ego_to_grid @ curr2adj_2d @ grid_to_ego


def main():
    args = parse_args()
    model_cfg = yaml.safe_load(Path(args.model_cfg).read_text())
    cams = model_cfg["data_config"]["cams"]
    ref_cam = cams[0]
    x_start, _, x_step = map(float, model_cfg["grid_config"]["x"])
    y_start, _, y_step = map(float, model_cfg["grid_config"]["y"])

    sample_paths = sorted(Path(args.samples_dir).glob("sample*.yaml"))
    if not sample_paths:
        raise RuntimeError(f"No sample*.yaml found in {args.samples_dir}")

    e2g_seq = []
    for p in sample_paths:
        sample = yaml.safe_load(p.read_text())
        cam_data = sample["cams"][ref_cam]
        e2g_seq.append(e2g_matrix(cam_data["ego2global_rotation"], cam_data["ego2global_translation"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, curr in enumerate(e2g_seq):
        mats = np.zeros((args.adj_num, 3, 3), dtype=np.float32)
        for j in range(args.adj_num):
            adj_idx = i - (j + 1)
            if adj_idx < 0:
                adj_idx = 0
            adj = e2g_seq[adj_idx]
            mats[j] = curr_to_adj_grid(curr, adj, x_start, x_step, y_start, y_step)
        out_file = out_dir / f"{i:06d}.bin"
        out_file.write_bytes(mats.reshape(-1).tobytes())

    print(f"Generated {len(sample_paths)} transform files in {out_dir}")


if __name__ == "__main__":
    main()
