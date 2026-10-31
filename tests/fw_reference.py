"""Reference copies of FlashWorld camera utilities, used only to test c44.cameras against the original.

Source: https://github.com/imlixinyang/FlashWorld (Apache-2.0), utils.py at commit 424db44;
quaternion conversions originally from PyTorch3D (BSD). Copied verbatim except for the removed
autocast decorator (CPU tests) and imports.
"""

import torch


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix):
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def matrix_to_square(mat):
    l = len(mat.shape)
    if l == 3:
        return torch.cat([mat, torch.tensor([0, 0, 0, 1]).repeat(mat.shape[0], 1, 1).to(mat.device)], dim=1)
    elif l == 4:
        return torch.cat([mat, torch.tensor([0, 0, 0, 1]).repeat(mat.shape[0], mat.shape[1], 1, 1).to(mat.device)], dim=2)


def normalize_cameras(cameras, return_meta=False, ref_w2c=None, T_norm=None, n_frame=None):
    B, N = cameras.shape[:2]
    c2ws = torch.zeros(B, N, 3, 4, device=cameras.device)
    c2ws[..., :3, :3] = quaternion_to_matrix(cameras[..., 0:4])
    c2ws[..., :, 3] = cameras[..., 4:7]
    _c2ws = c2ws
    ref_w2c = torch.inverse(matrix_to_square(_c2ws[:, :1])) if ref_w2c is None else ref_w2c
    _c2ws = (ref_w2c.repeat(1, N, 1, 1) @ matrix_to_square(_c2ws))[..., :3, :]
    if n_frame is not None:
        T_norm = _c2ws[..., :n_frame, :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None] if T_norm is None else T_norm
    else:
        T_norm = _c2ws[..., :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None] if T_norm is None else T_norm
    _c2ws[..., :3, 3] = _c2ws[..., :3, 3] / (T_norm + 1e-2)
    R = matrix_to_quaternion(_c2ws[..., :3, :3])
    T = _c2ws[..., :3, 3]
    cameras = torch.cat([R.float(), T.float(), cameras[..., 7:]], dim=-1)
    if return_meta:
        return cameras, ref_w2c, T_norm
    return cameras
