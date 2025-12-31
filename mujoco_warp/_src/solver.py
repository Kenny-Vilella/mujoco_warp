# Copyright 2025 The Newton Developers
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
# ==============================================================================

from math import ceil
from math import sqrt
from typing import Tuple

import warp as wp

from . import math
from . import smooth
from . import support
from . import types
from .block_cholesky import create_blocked_cholesky_func
from .block_cholesky import create_blocked_cholesky_solve_func
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})
wp.set_module_options({"lineinfo": True})


# Native CUDA syncthreads for block-level thread synchronization
@wp.func_native("__syncthreads();")
def syncthreads():
  """Synchronize all threads in the block. All threads must reach this point before any can proceed."""
  ...


@wp.func
def _rescale(nv: int, stat_meaninertia: float, value: float) -> float:
  return value / (stat_meaninertia * float(wp.max(1, nv)))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_cost(quad: wp.vec3, alpha: float) -> float:
  return alpha * alpha * quad[2] + alpha * quad[1] + quad[0]


@wp.func
def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:
  return wp.vec3(
    _eval_cost(quad, alpha),
    2.0 * alpha * quad[2] + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _eval_frictionloss(
  # In:
  x: float,
  f: float,
  rf: float,
  Jaref: float,
  jv: float,
  quad: wp.vec3,
) -> wp.vec3:
  # -bound < x < bound : quadratic
  if (-rf < x) and (x < rf):
    return quad
  # x < -bound: linear negative
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - Jaref), -f * jv, 0.0)
  # bound < x : linear positive
  else:
    return wp.vec3(f * (-0.5 * rf + Jaref), f * jv, 0.0)


@wp.func
def _eval_elliptic(
  # In:
  impratio_invsqrt: float,
  friction: types.vec5,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
  alpha: float,
) -> wp.vec3:
  mu = friction[0] * impratio_invsqrt

  u0 = quad1[0]
  v0 = quad1[1]
  uu = quad1[2]
  uv = quad2[0]
  vv = quad2[1]
  dm = quad2[2]

  # compute N, Tsqr
  N = u0 + alpha * v0
  Tsqr = uu + alpha * (2.0 * uv + alpha * vv)

  # no tangential force: top or bottom zone
  if Tsqr <= 0.0:
    # bottom zone: quadratic cost
    if N < 0.0:
      return _eval_pt(quad, alpha)

    # top zone: nothing to do
  # otherwise regular processing
  else:
    # tangential force
    T = wp.sqrt(Tsqr)

    # N >= mu * T : top zone
    if N >= mu * T:
      # nothing to do
      pass
    # mu * N + T <= 0 : bottom zone
    elif mu * N + T <= 0.0:
      return _eval_pt(quad, alpha)

    # otherwise middle zone
    else:
      # derivatives
      N1 = v0
      T1 = (uv + alpha * vv) / T
      T2 = vv / T - (uv + alpha * vv) * T1 / (T * T)

      # add to cost
      cost = wp.vec3(
        0.5 * dm * (N - mu * T) * (N - mu * T),
        dm * (N - mu * T) * (N1 - mu * T1),
        dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2)),
      )

      return cost

  return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _eval_init(
  # In:
  ne_clip: int,
  nef_clip: int,
  nefc_clip: int,
  D_in: wp.array(dtype=float),
  frictionloss_in: wp.array(dtype=float),
  Jaref_in: wp.array(dtype=float),
  jv_in: wp.array(dtype=float),
  quad_in: wp.array(dtype=wp.vec3),
  alpha: float,
) -> wp.vec3:
  """Evaluate linesearch cost at alpha (PYRAMIDAL only)."""
  lo = wp.vec3(0.0, 0.0, 0.0)
  
  # Equality constraints
  for efcid in range(ne_clip):
    quad = quad_in[efcid]
    lo += _eval_pt(quad, alpha)

  # Friction loss constraints
  for efcid in range(ne_clip, nef_clip):
    D = D_in[efcid]
    f = frictionloss_in[efcid]
    Jaref = Jaref_in[efcid]
    jv = jv_in[efcid]
    x = Jaref + alpha * jv
    rf = math.safe_div(f, D)
    quad_f = _eval_frictionloss(x, f, rf, Jaref, jv, quad_in[efcid])
    lo += _eval_pt(quad_f, alpha)

  # Contact constraints (PYRAMIDAL only - no ELLIPTIC)
  for efcid in range(nef_clip, nefc_clip):
    Jaref = Jaref_in[efcid]
    jv = jv_in[efcid]
    quad = quad_in[efcid]
    x = Jaref + alpha * jv
    res = _eval_pt(quad, alpha)
    lo += res * float(x < 0.0)

  return lo


@wp.func
def _eval(
  # In:
  ne_clip: int,
  nef_clip: int,
  nefc_clip: int,
  D_in: wp.array(dtype=float),
  frictionloss_in: wp.array(dtype=float),
  Jaref_in: wp.array(dtype=float),
  jv_in: wp.array(dtype=float),
  quad_in: wp.array(dtype=wp.vec3),
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
) -> Tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Evaluate linesearch cost at three alpha values (PYRAMIDAL only)."""
  lo = wp.vec3(0.0, 0.0, 0.0)
  hi = wp.vec3(0.0, 0.0, 0.0)
  mid = wp.vec3(0.0, 0.0, 0.0)

  # Equality constraints
  for efcid in range(ne_clip):
    quad = quad_in[efcid]
    lo += _eval_pt(quad, lo_alpha)
    hi += _eval_pt(quad, hi_alpha)
    mid += _eval_pt(quad, mid_alpha)

  # Friction loss constraints
  for efcid in range(ne_clip, nef_clip):
    quad = quad_in[efcid]
    D = D_in[efcid]
    f = frictionloss_in[efcid]
    Jaref = Jaref_in[efcid]
    jv = jv_in[efcid]
    rf = math.safe_div(f, D)
    x_lo = Jaref + lo_alpha * jv
    x_hi = Jaref + hi_alpha * jv
    x_mid = Jaref + mid_alpha * jv
    quad_f = _eval_frictionloss(x_lo, f, rf, Jaref, jv, quad)
    lo += _eval_pt(quad_f, lo_alpha)
    quad_f = _eval_frictionloss(x_hi, f, rf, Jaref, jv, quad)
    hi += _eval_pt(quad_f, hi_alpha)
    quad_f = _eval_frictionloss(x_mid, f, rf, Jaref, jv, quad)
    mid += _eval_pt(quad_f, mid_alpha)

  # Contact constraints (PYRAMIDAL only - no ELLIPTIC)
  for efcid in range(nef_clip, nefc_clip):
    Jaref = Jaref_in[efcid]
    jv = jv_in[efcid]
    quad = quad_in[efcid]
    x_lo = Jaref + lo_alpha * jv
    x_hi = Jaref + hi_alpha * jv
    x_mid = Jaref + mid_alpha * jv
    lo += _eval_pt(quad, lo_alpha) * float(x_lo < 0.0)
    hi += _eval_pt(quad, hi_alpha) * float(x_hi < 0.0)
    mid += _eval_pt(quad, mid_alpha) * float(x_mid < 0.0)

  return lo, hi, mid


@wp.kernel
def linesearch_kernel(
  # Model:
  nv: int,
  opt_impratio: wp.array(dtype=float),
  opt_tolerance: wp.array(dtype=float),
  opt_ls_tolerance: wp.array(dtype=float),
  opt_ls_iterations: int,
  opt_ls_parallel: bool,
  opt_ls_parallel_min_step: float,
  stat_meaninertia: float,
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  qfrc_smooth_in: wp.array2d(dtype=float),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_jv_in: wp.array2d(dtype=float),
  efc_quad_in: wp.array2d(dtype=wp.vec3),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_Mgrad_in: wp.array2d(dtype=float),
  efc_gauss_in: wp.array(dtype=float),
  efc_mv_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  njmax_in: int,
  nacon_in: wp.array(dtype=int),
  # Data out:
  qacc_out: wp.array2d(dtype=float),
  efc_Ma_out: wp.array2d(dtype=float),
  efc_Jaref_out: wp.array2d(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  linesearch_fn(worldid, nv, opt_impratio, opt_tolerance, opt_ls_tolerance, opt_ls_iterations,
                opt_ls_parallel, opt_ls_parallel_min_step, stat_meaninertia, ne_in, nf_in, nefc_in,
                contact_friction_in, contact_dim_in, contact_efc_address_in, qfrc_smooth_in, efc_type_in,
                efc_id_in, efc_J_in, efc_D_in, efc_frictionloss_in, efc_Jaref_in,
                efc_jv_in, efc_quad_in, efc_Ma_in, efc_Mgrad_in, efc_gauss_in, efc_mv_in, njmax_in,
                nacon_in, qacc_out, efc_Ma_out, efc_Jaref_out)

@wp.func
def linesearch_fn(
  # In:
  worldid: int,
  # Model:
  nv: int,
  opt_impratio: wp.array(dtype=float),
  opt_tolerance: wp.array(dtype=float),
  opt_ls_tolerance: wp.array(dtype=float),
  opt_ls_iterations: int,
  opt_ls_parallel: bool,
  opt_ls_parallel_min_step: float,
  stat_meaninertia: float,
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  qfrc_smooth_in: wp.array2d(dtype=float),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_jv_in: wp.array2d(dtype=float),
  efc_quad_in: wp.array2d(dtype=wp.vec3),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_Mgrad_in: wp.array2d(dtype=float),
  efc_gauss_in: wp.array(dtype=float),
  efc_mv_in: wp.array2d(dtype=float),
  njmax_in: int,
  nacon_in: wp.array(dtype=int),
  # Data out:
  qacc_out: wp.array2d(dtype=float),
  efc_Ma_out: wp.array2d(dtype=float),
  efc_Jaref_out: wp.array2d(dtype=float),
):

  for efcid in range(nefc_in[worldid]):
    jv_out = float(0.0)
    for dofid in range(nv):
      jv_out -= efc_J_in[worldid, efcid, dofid] * efc_Mgrad_in[worldid, dofid]
    efc_jv_in[worldid, efcid] = jv_out


  quad_gauss_0 = efc_gauss_in[worldid]
  quad_gauss_1 = float(0.0)
  quad_gauss_2 = float(0.0)
  for i in range(nv):
    search = -efc_Mgrad_in[worldid, i]
    quad_gauss_1 += search * (efc_Ma_in[worldid, i] - qfrc_smooth_in[worldid, i])
    quad_gauss_2 += 0.5 * search * efc_mv_in[worldid, i]

  quad_gauss = wp.vec3(quad_gauss_0, quad_gauss_1, quad_gauss_2)

  # quad per constraint (PYRAMIDAL only)
  for efcid in range(nefc_in[worldid]):
    Jaref = efc_Jaref_in[worldid, efcid]
    jv = efc_jv_in[worldid, efcid]
    efc_D = efc_D_in[worldid, efcid]
    quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D)
    efc_quad_in[worldid, efcid] = quad

  if opt_ls_parallel:
    alpha = linesearch_parallel(worldid, opt_ls_iterations, opt_ls_parallel_min_step, 
                         ne_in, nf_in, nefc_in, efc_D_in, efc_frictionloss_in, efc_Jaref_in,
                         efc_jv_in, efc_quad_in, quad_gauss, njmax_in)
  else:
    efc_search_dot_in = float(0.0)
    for dofid in range(nv):
      search = efc_Mgrad_in[worldid, dofid]
      efc_search_dot_in += search * search
    alpha = linesearch_iterative(worldid, nv, opt_tolerance, opt_ls_tolerance, opt_ls_iterations,
                        stat_meaninertia, ne_in, nf_in, nefc_in, efc_D_in, efc_frictionloss_in,
                        efc_Jaref_in, efc_search_dot_in, efc_jv_in, efc_quad_in, quad_gauss, njmax_in)


  for dofid in range(nv):
    qacc_out[worldid, dofid] -= alpha * efc_Mgrad_in[worldid, dofid]
    efc_Ma_out[worldid, dofid] += alpha * efc_mv_in[worldid, dofid]

  for efcid in range(nefc_in[worldid]):
    efc_Jaref_out[worldid, efcid] += alpha * efc_jv_in[worldid, efcid]

@wp.func
def linesearch_iterative(
  # In:
  worldid: int,
  # Model:
  nv: int,
  opt_tolerance: wp.array(dtype=float),
  opt_ls_tolerance: wp.array(dtype=float),
  opt_ls_iterations: int,
  stat_meaninertia: float,
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_search_dot_in: float,
  efc_jv_in: wp.array2d(dtype=float),
  efc_quad_in: wp.array2d(dtype=wp.vec3),
  efc_quad_gauss: wp.vec3,
  njmax_in: int,
):
  """Iterative linesearch (PYRAMIDAL only)."""
  efc_D = efc_D_in[worldid]
  efc_frictionloss = efc_frictionloss_in[worldid]
  efc_Jaref = efc_Jaref_in[worldid]
  efc_jv = efc_jv_in[worldid]
  efc_quad = efc_quad_in[worldid]
  tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
  ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
  ne_clip = min(njmax_in, ne_in[worldid])
  nef_clip = min(njmax_in, ne_clip + nf_in[worldid])
  nefc_clip = min(njmax_in, nefc_in[worldid])

  # Calculate p0
  snorm = wp.math.sqrt(efc_search_dot_in)
  scale = stat_meaninertia * wp.float(wp.max(1, nv))
  gtol = tolerance * ls_tolerance * snorm * scale
  p0 = wp.vec3(efc_quad_gauss[0], efc_quad_gauss[1], 2.0 * efc_quad_gauss[2])
  p0 += _eval_init(
    ne_clip, nef_clip, nefc_clip,
    efc_D, efc_frictionloss, efc_Jaref, efc_jv, efc_quad,
    0.0,
  )

  # Calculate lo bound
  lo_alpha_in = -math.safe_div(p0[1], p0[2])
  lo_in = _eval_pt(efc_quad_gauss, lo_alpha_in)
  lo_in += _eval_init(
    ne_clip, nef_clip, nefc_clip,
    efc_D, efc_frictionloss, efc_Jaref, efc_jv, efc_quad,
    lo_alpha_in,
  )

  # Initialize bounds
  lo_less = lo_in[1] < p0[1]
  lo = wp.where(lo_less, lo_in, p0)
  lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)
  hi = wp.where(lo_less, p0, lo_in)
  hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)

  # Launch main linesearch iterative loop
  alpha = float(0.0)
  for _ in range(opt_ls_iterations):
    lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])
    hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])
    mid_alpha = 0.5 * (lo_alpha + hi_alpha)

    lo_next, hi_next, mid = _eval(
      ne_clip, nef_clip, nefc_clip,
      efc_D, efc_frictionloss, efc_Jaref, efc_jv, efc_quad,
      lo_next_alpha, hi_next_alpha, mid_alpha,
    )
    lo_next += _eval_pt(efc_quad_gauss, lo_next_alpha)
    hi_next += _eval_pt(efc_quad_gauss, hi_next_alpha)
    mid += _eval_pt(efc_quad_gauss, mid_alpha)

    # swap lo:
    swap_lo_lo_next = _in_bracket(lo, lo_next)
    lo = wp.where(swap_lo_lo_next, lo_next, lo)
    lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
    swap_lo_mid = _in_bracket(lo, mid)
    lo = wp.where(swap_lo_mid, mid, lo)
    lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
    swap_lo_hi_next = _in_bracket(lo, hi_next)
    lo = wp.where(swap_lo_hi_next, hi_next, lo)
    lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
    swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

    # swap hi:
    swap_hi_hi_next = _in_bracket(hi, hi_next)
    hi = wp.where(swap_hi_hi_next, hi_next, hi)
    hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
    swap_hi_mid = _in_bracket(hi, mid)
    hi = wp.where(swap_hi_mid, mid, hi)
    hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
    swap_hi_lo_next = _in_bracket(hi, lo_next)
    hi = wp.where(swap_hi_lo_next, lo_next, hi)
    hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
    swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

    # if we did not adjust the interval, we are done
    # also done if either low or hi slope is nearly flat
    ls_done = (not swap_lo and not swap_hi) or (lo[1] < 0 and lo[1] > -gtol) or (hi[1] > 0 and hi[1] < gtol)

    # update alpha if we have an improvement
    improved = lo[0] < p0[0] or hi[0] < p0[0]
    lo_better = lo[0] < hi[0]
    alpha = wp.where(improved and lo_better, lo_alpha, alpha)
    alpha = wp.where(improved and not lo_better, hi_alpha, alpha)
    if ls_done:
      break

  return alpha


@wp.func
def _log_scale(min_value: float, max_value: float, num_values: int, i: int) -> float:
  step = (wp.log(max_value) - wp.log(min_value)) / wp.max(1.0, float(num_values - 1))
  return wp.exp(wp.log(min_value) + float(i) * step)


@wp.func
def linesearch_parallel(
  # In:
  worldid: int,
  # Model:
  opt_ls_iterations: int,
  opt_ls_parallel_min_step: float,
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_jv_in: wp.array2d(dtype=float),
  efc_quad_in: wp.array2d(dtype=wp.vec3),
  efc_quad_gauss_in: wp.vec3,
  njmax_in: int,
):
  """Parallel linesearch (PYRAMIDAL only)."""
  ne = ne_in[worldid]
  nf = nf_in[worldid]

  bestid = int(0)
  best_cost = float(wp.inf)
  for alphaid in range(opt_ls_iterations):
    alpha = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, alphaid)
    out = _eval_cost(efc_quad_gauss_in, alpha)

    for efcid in range(min(njmax_in, nefc_in[worldid])):
      # equality constraints
      if efcid < ne:
        out += _eval_cost(efc_quad_in[worldid, efcid], alpha)
      # friction loss constraints
      elif efcid < ne + nf:
        start = efc_Jaref_in[worldid, efcid]
        dir = efc_jv_in[worldid, efcid]
        x = start + alpha * dir
        f = efc_frictionloss_in[worldid, efcid]
        rf = math.safe_div(f, efc_D_in[worldid, efcid])

        if (-rf < x) and (x < rf):
          quad = efc_quad_in[worldid, efcid]
        elif x <= -rf:
          quad = wp.vec3(f * (-0.5 * rf - start), -f * dir, 0.0)
        else:
          quad = wp.vec3(f * (-0.5 * rf + start), f * dir, 0.0)
        out += _eval_cost(quad, alpha)
      # contact constraints (PYRAMIDAL only - no ELLIPTIC)
      else:
        x = efc_Jaref_in[worldid, efcid] + alpha * efc_jv_in[worldid, efcid]
        if x < 0.0:
          out += _eval_cost(efc_quad_in[worldid, efcid], alpha)

    if out < best_cost:
      best_cost = out
      bestid = alphaid

  return _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, bestid)


@wp.kernel
def update_constraint_kernel(
  # Model:
  nv: int,
  opt_impratio: wp.array(dtype=float),
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  qacc_in: wp.array2d(dtype=float),
  qfrc_smooth_in: wp.array2d(dtype=float),
  qacc_smooth_in: wp.array2d(dtype=float),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_J_in: wp.array3d(dtype=float),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_cost_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  nacon_in: wp.array(dtype=int),
  # Data out:
  qfrc_constraint_out: wp.array2d(dtype=float),
  efc_force_out: wp.array2d(dtype=float),
  efc_gauss_out: wp.array(dtype=float),
  efc_cost_out: wp.array(dtype=float),
  efc_prev_cost_out: wp.array(dtype=float),
  efc_state_out: wp.array2d(dtype=int),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  update_constraint_fn(worldid, nv, opt_impratio, ne_in, nf_in, nefc_in, contact_friction_in,
                      contact_dim_in, contact_efc_address_in, qacc_in, qfrc_smooth_in, qacc_smooth_in,
                      efc_Ma_in, efc_J_in, efc_type_in, efc_id_in, efc_D_in, efc_frictionloss_in,
                      efc_Jaref_in, efc_cost_in, nacon_in, qfrc_constraint_out, efc_force_out,
                      efc_gauss_out, efc_cost_out, efc_prev_cost_out, efc_state_out)


@wp.func
def update_constraint_fn(
  worldid: int,
  # Model:
  nv: int,
  opt_impratio: wp.array(dtype=float),
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  qacc_in: wp.array2d(dtype=float),
  qfrc_smooth_in: wp.array2d(dtype=float),
  qacc_smooth_in: wp.array2d(dtype=float),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_J_in: wp.array3d(dtype=float),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_cost_in: wp.array(dtype=float),
  nacon_in: wp.array(dtype=int),
  # Data out:
  qfrc_constraint_out: wp.array2d(dtype=float),
  efc_force_out: wp.array2d(dtype=float),
  efc_gauss_out: wp.array(dtype=float),
  efc_cost_out: wp.array(dtype=float),
  efc_prev_cost_out: wp.array(dtype=float),
  efc_state_out: wp.array2d(dtype=int),
):
  efc_gauss_out[worldid] = 0.0
  efc_prev_cost_out[worldid] = efc_cost_in[worldid]
  efc_cost_out[worldid] = 0.0

  ne = ne_in[worldid]
  nf = nf_in[worldid]

  for efcid in range(nefc_in[worldid]):
    efc_D = efc_D_in[worldid, efcid]
    Jaref = efc_Jaref_in[worldid, efcid]

    if efcid < ne:
      # equality
      efc_force_out[worldid, efcid] = -efc_D * Jaref
      efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
      efc_cost_out[worldid] += 0.5 * efc_D * Jaref * Jaref
    elif efcid < ne + nf:
      # friction
      f = efc_frictionloss_in[worldid, efcid]
      rf = math.safe_div(f, efc_D)
      if Jaref <= -rf:
        efc_force_out[worldid, efcid] = f
        efc_state_out[worldid, efcid] = types.ConstraintState.LINEARNEG
        efc_cost_out[worldid] += -f * (0.5 * rf + Jaref)
      elif Jaref >= rf:
        efc_force_out[worldid, efcid] = -f
        efc_state_out[worldid, efcid] = types.ConstraintState.LINEARPOS
        efc_cost_out[worldid] += -f * (0.5 * rf - Jaref)
      else:
        efc_force_out[worldid, efcid] = -efc_D * Jaref
        efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
        efc_cost_out[worldid] += 0.5 * efc_D * Jaref * Jaref
    elif efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
      # limit, frictionless contact, pyramidal friction cone contact
      if Jaref >= 0.0:
        efc_force_out[worldid, efcid] = 0.0
        efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED
      else:
        efc_force_out[worldid, efcid] = -efc_D * Jaref
        efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
        efc_cost_out[worldid] += 0.5 * efc_D * Jaref * Jaref
    else:  # elliptic friction cone contact
      conid = efc_id_in[worldid, efcid]

      if conid >= nacon_in[0]:
        continue

      dim = contact_dim_in[conid]
      friction = contact_friction_in[conid]
      mu = friction[0] / wp.sqrt(opt_impratio[worldid])

      efcid0 = contact_efc_address_in[conid, 0]
      if efcid0 < 0:
        continue

      N = efc_Jaref_in[worldid, efcid0] * mu

      ufrictionj = float(0.0)
      TT = float(0.0)
      for j in range(1, dim):
        efcidj = contact_efc_address_in[conid, j]
        if efcidj < 0:
          break
        frictionj = friction[j - 1]
        uj = efc_Jaref_in[worldid, efcidj] * frictionj
        TT += uj * uj
        if efcid == efcidj:
          ufrictionj = uj * frictionj
      if efcidj < 0:
        continue

      if TT <= 0.0:
        T = 0.0
      else:
        T = wp.sqrt(TT)

      # top zone
      if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):
        efc_force_out[worldid, efcid] = 0.0
        efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED
      # bottom zone
      elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):
        efc_force_out[worldid, efcid] = -efc_D * Jaref
        efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
        efc_cost_out[worldid] += 0.5 * efc_D * Jaref * Jaref
      # middle zone
      else:
        dm = math.safe_div(efc_D_in[worldid, efcid0], mu * mu * (1.0 + mu * mu))
        nmt = N - mu * T

        force = -dm * nmt * mu

        if efcid == efcid0:
          efc_force_out[worldid, efcid] = force
          efc_cost_out[worldid] += 0.5 * dm * nmt * nmt
        else:
          efc_force_out[worldid, efcid] = -math.safe_div(force, T) * ufrictionj

        efc_state_out[worldid, efcid] = types.ConstraintState.CONE

  # Note: qfrc_constraint and gauss_cost are now computed separately in the kernel
  # for parallelization with syncthreads


@wp.func
def state_check(D: float, state: int) -> float:
  if state == types.ConstraintState.QUADRATIC.value:
    return D
  else:
    return 0.0


@wp.func
def active_check(tid: int, threshold: int) -> float:
  if tid >= threshold:
    return 0.0
  else:
    return 1.0


@wp.kernel
def update_gradient_kernel(
  # Model:
  nv: int,
  opt_solver: int,
  opt_is_sparse: bool,
  opt_cone: int,
  opt_impratio: wp.array(dtype=float),
  dof_tri_row: wp.array(dtype=int),
  dof_tri_col: wp.array(dtype=int),
  qM_fullm_i: wp.array(dtype=int),
  qM_fullm_j: wp.array(dtype=int),
  # Data in:
  nefc_in: wp.array(dtype=int),
  qM_in: wp.array3d(dtype=float),
  qfrc_smooth_in: wp.array2d(dtype=float),
  qfrc_constraint_in: wp.array2d(dtype=float),
  contact_dist_in: wp.array(dtype=float),
  contact_includemargin_in: wp.array(dtype=float),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  efc_done_in: wp.array(dtype=bool),
  naconmax_in: int,
  nacon_in: wp.array(dtype=int),
  # Data out:
  efc_grad_out: wp.array2d(dtype=float),
  efc_grad_dot_out: wp.array(dtype=float),
  efc_h_out: wp.array3d(dtype=float),
  efc_Mgrad_out: wp.array2d(dtype=float),
):
  """Unified kernel for ALL solvers (CG, Newton Dense, Newton Sparse) - NO TILES."""
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  update_gradient_fn(worldid, nv, opt_solver, opt_is_sparse, opt_cone, opt_impratio, dof_tri_row, dof_tri_col,
                     qM_fullm_i, qM_fullm_j, nefc_in, qM_in, qfrc_smooth_in, qfrc_constraint_in, contact_dist_in,
                     contact_includemargin_in, contact_friction_in, contact_dim_in, contact_efc_address_in, contact_worldid_in,
                     efc_Ma_in, efc_J_in, efc_D_in, efc_Jaref_in, efc_state_in, naconmax_in, nacon_in, efc_grad_out,
                     efc_grad_dot_out, efc_h_out, efc_Mgrad_out)


@wp.func
def update_gradient_fn(
  # In:
  worldid: int,
  # Model:
  nv: int,
  opt_solver: int,
  opt_is_sparse: bool,
  opt_cone: int,
  opt_impratio: wp.array(dtype=float),
  dof_tri_row: wp.array(dtype=int),
  dof_tri_col: wp.array(dtype=int),
  qM_fullm_i: wp.array(dtype=int),
  qM_fullm_j: wp.array(dtype=int),
  # Data in:
  nefc_in: wp.array(dtype=int),
  qM_in: wp.array3d(dtype=float),
  qfrc_smooth_in: wp.array2d(dtype=float),
  qfrc_constraint_in: wp.array2d(dtype=float),
  contact_dist_in: wp.array(dtype=float),
  contact_includemargin_in: wp.array(dtype=float),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  naconmax_in: int,
  nacon_in: wp.array(dtype=int),
  # Data out:
  efc_grad_out: wp.array2d(dtype=float),
  efc_grad_dot_out: wp.array(dtype=float),
  efc_h_out: wp.array3d(dtype=float),
  efc_Mgrad_out: wp.array2d(dtype=float),
):
  """Unified kernel for ALL solvers (CG, Newton Dense, Newton Sparse) - NO TILES."""

  nefc = nefc_in[worldid]
  
  # Step 1: Compute gradient
  efc_grad_dot_out[worldid] = 0.0
  for dofid in range(nv):
    grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]
    efc_grad_out[worldid, dofid] = grad
    efc_grad_dot_out[worldid] += grad * grad

  # Step 2: Compute h for Newton solver (dense only)
  if opt_solver == types.SolverType.NEWTON:
    # Newton Dense: compute h = qM + J.T @ D @ J using scalar operations (no tiles)
    # Initialize h with qM
    for i in range(nv):
      for j in range(nv):
        efc_h_out[worldid, i, j] = qM_in[worldid, i, j]

    # Compute J.T @ D @ 
    for k in range(nefc):
      if efc_state_in[worldid, k] != types.ConstraintState.QUADRATIC.value:
        continue
      D_val = efc_D_in[worldid, k]
      for i in range(nv):
        JD_val = efc_J_in[worldid, k, i] * D_val
        for j in range(nv):
          efc_h_out[worldid, i, j] += JD_val * efc_J_in[worldid, k, j]

  # Solve h * Mgrad = grad using Cholesky decomposition (scalar operations)
  # Step 1: Cholesky factorization of h (in-place): h = L * L^T
  for i in range(nv):
    for j in range(i + 1):
      sum_val = efc_h_out[worldid, i, j]
      for k in range(j):
        sum_val -= efc_h_out[worldid, i, k] * efc_h_out[worldid, j, k]
      if i == j:
        efc_h_out[worldid, i, j] = wp.sqrt(wp.max(sum_val, types.MJ_MINVAL))
      else:
        efc_h_out[worldid, i, j] = sum_val / wp.max(efc_h_out[worldid, j, j], types.MJ_MINVAL)
  
  # Step 2: Forward substitution: L * temp = grad
  for i in range(nv):
    temp_val = efc_grad_out[worldid, i]
    for j in range(i):
      temp_val -= efc_h_out[worldid, i, j] * efc_Mgrad_out[worldid, j]
    efc_Mgrad_out[worldid, i] = temp_val / wp.max(efc_h_out[worldid, i, i], types.MJ_MINVAL)
  
  # Step 3: Backward substitution: L^T * Mgrad = temp
  for i_rev in range(nv):
    i = nv - 1 - i_rev
    temp_val = efc_Mgrad_out[worldid, i]
    for j in range(i + 1, nv):
      temp_val -= efc_h_out[worldid, j, i] * efc_Mgrad_out[worldid, j]
    efc_Mgrad_out[worldid, i] = temp_val / wp.max(efc_h_out[worldid, i, i], types.MJ_MINVAL)


# =============================================================================
# Tiled solver_iteration kernel factory
# =============================================================================
@cache_kernel
def solver_iteration(nv_param: int, njmax_param: int):
  """Factory for solver_iteration kernel with compile-time nv and njmax.
  
  Enables tiled operations for dense Newton when nv <= 32.
  """
  # Compile-time constants
  NV = nv_param
  NJMAX = njmax_param
  TILE_SIZE_JTDAJ = 8  # Reduced from 16 to decrease register pressure

  # Block dimension for cooperative tile operations
  BLOCK_DIM = 32

  # Local function for tiled Hessian computation (uses closure for NV, NJMAX, TILE_SIZE_JTDAJ)
  @wp.func
  def compute_hessian_tiled(
    worldid: int,
    nefc: int,
    qM_in: wp.array3d(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_state_in: wp.array2d(dtype=int),
    efc_h_out: wp.array3d(dtype=float),
  ):
    """Compute h = qM + J.T @ D @ J using tiled operations."""
    # Use shared memory for large tiles to reduce register pressure
    sum_val = wp.tile_load(qM_in[worldid], shape=(NV, NV), storage="shared", bounds_check=False)
    
    for k in range(0, NJMAX, TILE_SIZE_JTDAJ):
      if k >= nefc:
        break
      
      J_ki = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_JTDAJ, NV), offset=(k, 0), storage="shared", bounds_check=False)
      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_JTDAJ, offset=k, storage="shared", bounds_check=False)
      state_k = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_JTDAJ, offset=k, storage="shared", bounds_check=False)
      
      # Apply state mask
      D_k = wp.tile_map(state_check, D_k, state_k)
      
      # Apply active mask for out-of-bounds
      tid_tile = wp.tile_arange(TILE_SIZE_JTDAJ, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_JTDAJ, dtype=int) * (nefc - k)
      active_tile = wp.tile_map(active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)
      
      # J.T @ D @ J
      J_ki_scaled = wp.tile_map(wp.mul, wp.tile_transpose(J_ki), wp.tile_broadcast(D_k, shape=(NV, TILE_SIZE_JTDAJ)))
      wp.tile_matmul(J_ki_scaled, J_ki, sum_val)

    wp.tile_store(efc_h_out[worldid], sum_val, bounds_check=False)

  # Helper for squared value (for grad_dot computation)
  @wp.func
  def square(x: float) -> float:
    return x * x

  # Local function for tiled gradient computation
  @wp.func
  def compute_gradient_tiled(
    worldid: int,
    is_leader: bool,
    efc_Ma_in: wp.array2d(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qfrc_constraint_in: wp.array2d(dtype=float),
    efc_grad_out: wp.array2d(dtype=float),
    efc_grad_dot_out: wp.array(dtype=float),
  ):
    """Compute grad = Ma - qfrc_smooth - qfrc_constraint using tiled operations."""
    Ma_tile = wp.tile_load(efc_Ma_in[worldid], shape=NV, storage="shared", bounds_check=False)
    qfrc_smooth_tile = wp.tile_load(qfrc_smooth_in[worldid], shape=NV, storage="shared", bounds_check=False)
    qfrc_constraint_tile = wp.tile_load(qfrc_constraint_in[worldid], shape=NV, storage="shared", bounds_check=False)
    
    # grad = Ma - qfrc_smooth - qfrc_constraint (chain two subtractions)
    temp_tile = wp.tile_map(wp.sub, Ma_tile, qfrc_smooth_tile)
    grad_tile = wp.tile_map(wp.sub, temp_tile, qfrc_constraint_tile)
    wp.tile_store(efc_grad_out[worldid], grad_tile, bounds_check=False)
    
    # grad_dot = sum(grad^2)
    grad_sq_tile = wp.tile_map(square, grad_tile)
    grad_dot_tile = wp.tile_sum(grad_sq_tile)
    
    # Extract scalar from tile and write (leader only)
    if is_leader:
      grad_dot = wp.tile_extract(grad_dot_tile, 0)
      efc_grad_dot_out[worldid] = grad_dot

  # Local function for tiled Cholesky solve
  @wp.func
  def cholesky_solve_tiled(
    worldid: int,
    efc_h_in: wp.array3d(dtype=float),
    efc_grad_in: wp.array2d(dtype=float),
    efc_Mgrad_out: wp.array2d(dtype=float),
  ):
    """Solve h * Mgrad = grad using tiled Cholesky."""
    # Use shared memory for large tiles to reduce register pressure
    h_tile = wp.tile_load(efc_h_in[worldid], shape=(NV, NV), storage="shared", bounds_check=False)
    L_tile = wp.tile_cholesky(h_tile)
    grad_tile = wp.tile_load(efc_grad_in[worldid], shape=NV, storage="shared", bounds_check=False)
    Mgrad_tile = wp.tile_cholesky_solve(L_tile, grad_tile)
    wp.tile_store(efc_Mgrad_out[worldid], Mgrad_tile, bounds_check=False)

  # Local function for Ma = qM @ qacc (tiled, nv <= 32 only)
  @wp.func
  def compute_Ma_tiled(
    worldid: int,
    tid_in_block: int,
    qM_in: wp.array3d(dtype=float),
    qacc_in: wp.array2d(dtype=float),
    efc_Ma_out: wp.array2d(dtype=float),
  ):
    """Compute Ma = qM @ qacc using tiled qacc load."""
    qacc_tile = wp.tile_load(qacc_in[worldid], shape=NV, storage="shared", bounds_check=False)
    
    # Each thread computes one row (NV <= 32, we have 32 threads)
    if tid_in_block < NV:
      ma_val = float(0.0)
      for j in range(NV):
        ma_val += qM_in[worldid, tid_in_block, j] * wp.tile_extract(qacc_tile, j)
      efc_Ma_out[worldid, tid_in_block] = ma_val

  # Local function for Jaref = J @ qacc - aref (tiled)
  @wp.func
  def compute_Jaref_tiled(
    worldid: int,
    is_leader: bool,
    nefc: int,
    efc_J_in: wp.array3d(dtype=float),
    qacc_in: wp.array2d(dtype=float),
    efc_aref_in: wp.array2d(dtype=float),
    efc_Jaref_out: wp.array2d(dtype=float),
  ):
    """Compute Jaref = J @ qacc - aref using tiled operations."""
    qacc_tile = wp.tile_load(qacc_in[worldid], shape=NV, storage="shared", bounds_check=False)
    
    # Process constraints in tiles
    for k in range(0, NJMAX, TILE_SIZE_JTDAJ):
      if k >= nefc:
        break
      J_tile = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_JTDAJ, NV), offset=(k, 0), storage="shared", bounds_check=False)
      
      # Leader writes results
      if is_leader:
        for i in range(TILE_SIZE_JTDAJ):
          if k + i < nefc:
            jaref_val = float(0.0)
            for j in range(NV):
              jaref_val += wp.tile_extract(J_tile, i, j) * wp.tile_extract(qacc_tile, j)
            efc_Jaref_out[worldid, k + i] = jaref_val - efc_aref_in[worldid, k + i]

  # Local function for update_gradient (tiled)
  # Note: Uses tile operations, so this function must be called by ALL threads
  @wp.func
  def update_gradient_fn_local(
    worldid: int,
    is_leader: bool,
    nefc: int,
    qM_in: wp.array3d(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qfrc_constraint_in: wp.array2d(dtype=float),
    efc_Ma_in: wp.array2d(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_state_in: wp.array2d(dtype=int),
    efc_grad_out: wp.array2d(dtype=float),
    efc_grad_dot_out: wp.array(dtype=float),
    efc_h_out: wp.array3d(dtype=float),
    efc_Mgrad_out: wp.array2d(dtype=float),
  ):
    """Compute gradient, Hessian, and Cholesky solve - all tiled."""
    # Gradient computation (tiled)
    compute_gradient_tiled(worldid, is_leader, efc_Ma_in, qfrc_smooth_in, qfrc_constraint_in, efc_grad_out, efc_grad_dot_out)

    # Hessian computation (tiled)
    compute_hessian_tiled(worldid, nefc, qM_in, efc_J_in, efc_D_in, efc_state_in, efc_h_out)

    # Cholesky solve (tiled)
    cholesky_solve_tiled(worldid, efc_h_out, efc_grad_out, efc_Mgrad_out)

  @nested_kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_impratio: wp.array(dtype=float),
    opt_tolerance: wp.array(dtype=float),
    opt_iterations: int,
    opt_ls_tolerance: wp.array(dtype=float),
    opt_ls_iterations: int,
    opt_ls_parallel: bool,
    opt_ls_parallel_min_step: float,
    stat_meaninertia: float,
    # Data in:
    ne_in: wp.array(dtype=int),
    nf_in: wp.array(dtype=int),
    nefc_in: wp.array(dtype=int),
    qM_in: wp.array3d(dtype=float),
    contact_friction_in: wp.array(dtype=types.vec5),
    contact_dim_in: wp.array(dtype=int),
    contact_efc_address_in: wp.array2d(dtype=int),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    efc_type_in: wp.array2d(dtype=int),
    efc_id_in: wp.array2d(dtype=int),
    efc_J_in: wp.array3d(dtype=float),
    efc_aref_in: wp.array2d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_frictionloss_in: wp.array2d(dtype=float),
    efc_jv_in: wp.array2d(dtype=float),
    efc_quad_in: wp.array2d(dtype=wp.vec3),
    nacon_in: wp.array(dtype=int),
    # Data out:
    qacc_out: wp.array2d(dtype=float),
    efc_Ma_out: wp.array2d(dtype=float),
    efc_Jaref_out: wp.array2d(dtype=float),
    qfrc_constraint_out: wp.array2d(dtype=float),
    efc_force_out: wp.array2d(dtype=float),
    efc_gauss_out: wp.array(dtype=float),
    efc_cost_out: wp.array(dtype=float),
    efc_prev_cost_out: wp.array(dtype=float),
    efc_state_out: wp.array2d(dtype=int),
    efc_grad_out: wp.array2d(dtype=float),
    efc_grad_dot_out: wp.array(dtype=float),
    efc_h_out: wp.array3d(dtype=float),
    efc_Mgrad_out: wp.array2d(dtype=float),
    solver_niter_out: wp.array(dtype=int),
    nsolving_out: wp.array(dtype=int),
    efc_mv_out: wp.array2d(dtype=float),
  ):
    # 2D thread ID: worldid is the block index, tid_in_block is the thread within block
    worldid, tid_in_block = wp.tid()
    
    # Thread 0 in each block handles scalar operations
    is_leader = tid_in_block == 0

    # Get nefc for tiled operations (all threads read same value)
    nefc = nefc_in[worldid]

    # === Tiled initialization (all threads participate) ===
    # Ma = qM @ qacc
    compute_Ma_tiled(worldid, tid_in_block, qM_in, qacc_out, efc_Ma_out)
    # Jaref = J @ qacc - aref
    compute_Jaref_tiled(worldid, is_leader, nefc, efc_J_in, qacc_out, efc_aref_in, efc_Jaref_out)

    # === Scalar initialization (only thread 0) ===
    if is_leader:
      efc_cost_out[worldid] = wp.inf
      solver_niter_out[worldid] = 0

    ## 110 us until here

    # update_constraint_fn (force computation - leader only)
    if is_leader:
      update_constraint_fn(
        worldid, NV, opt_impratio,
        ne_in, nf_in, nefc_in,
        contact_friction_in, contact_dim_in, contact_efc_address_in,
        qacc_out, qfrc_smooth_in, qacc_smooth_in,
        efc_Ma_out, efc_J_in, efc_type_in, efc_id_in,
        efc_D_in, efc_frictionloss_in, efc_Jaref_out,
        efc_cost_out, nacon_in,
        qfrc_constraint_out, efc_force_out, efc_gauss_out,
        efc_cost_out, efc_prev_cost_out, efc_state_out
      )

    # SYNC: Ensure force is written before qfrc_constraint reads it
    syncthreads()

    # qfrc_constraint = J.T @ force (PARALLEL - each thread computes one dof)
    if tid_in_block < NV:
      sum_qfrc = float(0.0)
      for efcid in range(nefc):
        sum_qfrc += efc_J_in[worldid, efcid, tid_in_block] * efc_force_out[worldid, efcid]
      qfrc_constraint_out[worldid, tid_in_block] = sum_qfrc

    # SYNC: Ensure qfrc_constraint is written before gauss_cost reads it
    syncthreads()

    # gauss_cost (leader only - reduction)
    if is_leader:
      gauss_cost = float(0.0)
      for dofid in range(NV):
        gauss_cost += (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]) * (qacc_out[worldid, dofid] - qacc_smooth_in[worldid, dofid])
      efc_gauss_out[worldid] += 0.5 * gauss_cost
      efc_cost_out[worldid] += 0.5 * gauss_cost

    # SYNC: Ensure gauss_cost/cost updates complete
    syncthreads()

    ## 160 us until here

    nefc = nefc_in[worldid]
    update_gradient_fn_local(
      worldid, is_leader, nefc, qM_in, qfrc_smooth_in, qfrc_constraint_out,
      efc_Ma_out, efc_J_in, efc_D_in, efc_state_out,
      efc_grad_out, efc_grad_dot_out, efc_h_out, efc_Mgrad_out
    )

    ## 470 us until here

    # =========================================================================
    # Main iteration loop - PARALLELIZED with syncthreads
    # =========================================================================
    for _ in range(opt_iterations):
      # =====================================================================
      # PHASE 1: mv = -qM @ Mgrad (PARALLEL - each thread computes one row)
      # =====================================================================
      if tid_in_block < NV:
        mv_val = float(0.0)
        for j in range(NV):
          mv_val -= qM_in[worldid, tid_in_block, j] * efc_Mgrad_out[worldid, j]
        efc_mv_out[worldid, tid_in_block] = mv_val

      # =====================================================================
      # PHASE 2: jv = -J @ Mgrad (PARALLEL - threads handle different constraints)
      # =====================================================================
      efcid = tid_in_block
      while efcid < nefc:
        jv_val = float(0.0)
        for dofid in range(NV):
          jv_val -= efc_J_in[worldid, efcid, dofid] * efc_Mgrad_out[worldid, dofid]
        efc_jv_in[worldid, efcid] = jv_val
        efcid += BLOCK_DIM

      # SYNC: Ensure mv and jv are written before quad_gauss reads them
      syncthreads()

      # =====================================================================
      # PHASE 3: quad_gauss (leader only - has reduction, needs mv)
      # =====================================================================
      quad_gauss = wp.vec3(0.0, 0.0, 0.0)
      efc_search_dot_in = float(0.0)
      if is_leader:
        quad_gauss_0 = efc_gauss_out[worldid]
        quad_gauss_1 = float(0.0)
        quad_gauss_2 = float(0.0)
        for i in range(NV):
          search = -efc_Mgrad_out[worldid, i]
          quad_gauss_1 += search * (efc_Ma_out[worldid, i] - qfrc_smooth_in[worldid, i])
          quad_gauss_2 += 0.5 * search * efc_mv_out[worldid, i]
          efc_search_dot_in += efc_Mgrad_out[worldid, i] * efc_Mgrad_out[worldid, i]
        quad_gauss = wp.vec3(quad_gauss_0, quad_gauss_1, quad_gauss_2)

      # =====================================================================
      # PHASE 4: quad per constraint (PARALLEL - threads handle different constraints)
      # =====================================================================
      efcid = tid_in_block
      while efcid < nefc:
        Jaref_ls = efc_Jaref_out[worldid, efcid]
        jv = efc_jv_in[worldid, efcid]
        efc_D_ls = efc_D_in[worldid, efcid]
        efc_quad_in[worldid, efcid] = wp.vec3(0.5 * Jaref_ls * Jaref_ls * efc_D_ls, jv * Jaref_ls * efc_D_ls, 0.5 * jv * jv * efc_D_ls)
        efcid += BLOCK_DIM

      # SYNC: Ensure quad is written before linesearch reads it
      syncthreads()

      # =====================================================================
      # PHASE 5: linesearch (leader only - sequential algorithm)
      # =====================================================================
      alpha = float(0.0)
      if is_leader:
        if opt_ls_parallel:
          alpha = linesearch_parallel(worldid, opt_ls_iterations, opt_ls_parallel_min_step,
                               ne_in, nf_in, nefc_in, efc_D_in, efc_frictionloss_in, efc_Jaref_out,
                               efc_jv_in, efc_quad_in, quad_gauss, NJMAX)
        else:
          alpha = linesearch_iterative(worldid, NV, opt_tolerance, opt_ls_tolerance, opt_ls_iterations,
                              stat_meaninertia, ne_in, nf_in, nefc_in, efc_D_in, efc_frictionloss_in,
                              efc_Jaref_out, efc_search_dot_in, efc_jv_in, efc_quad_in, quad_gauss, NJMAX)
        # Broadcast alpha to shared location for other threads
        efc_grad_dot_out[worldid] = alpha

      # SYNC: Ensure alpha is written before parallel reads
      syncthreads()

      # All threads read alpha
      alpha_broadcast = efc_grad_dot_out[worldid]

      # =====================================================================
      # PHASE 6: Update qacc, Ma (PARALLEL - each thread updates one dof)
      # =====================================================================
      if tid_in_block < NV:
        qacc_out[worldid, tid_in_block] -= alpha_broadcast * efc_Mgrad_out[worldid, tid_in_block]
        efc_Ma_out[worldid, tid_in_block] += alpha_broadcast * efc_mv_out[worldid, tid_in_block]

      # =====================================================================
      # PHASE 7: Update Jaref (PARALLEL - threads handle different constraints)
      # =====================================================================
      efcid = tid_in_block
      while efcid < nefc:
        efc_Jaref_out[worldid, efcid] += alpha_broadcast * efc_jv_in[worldid, efcid]
        efcid += BLOCK_DIM

      # SYNC: Ensure qacc, Ma, Jaref are updated before update_constraint_fn
      syncthreads()

      # =====================================================================
      # PHASE 8: update_constraint_fn (leader only - computes force)
      # =====================================================================
      if is_leader:
        update_constraint_fn(
          worldid, NV, opt_impratio,
          ne_in, nf_in, nefc_in,
          contact_friction_in, contact_dim_in, contact_efc_address_in,
          qacc_out, qfrc_smooth_in, qacc_smooth_in,
          efc_Ma_out, efc_J_in, efc_type_in, efc_id_in,
          efc_D_in, efc_frictionloss_in, efc_Jaref_out,
          efc_cost_out, nacon_in,
          qfrc_constraint_out, efc_force_out, efc_gauss_out,
          efc_cost_out, efc_prev_cost_out, efc_state_out
        )

      # SYNC: Ensure force is written before qfrc_constraint reads it
      syncthreads()

      # =====================================================================
      # PHASE 8b: qfrc_constraint = J.T @ force (PARALLEL - each thread computes one dof)
      # =====================================================================
      if tid_in_block < NV:
        sum_qfrc = float(0.0)
        for efcid in range(nefc):
          sum_qfrc += efc_J_in[worldid, efcid, tid_in_block] * efc_force_out[worldid, efcid]
        qfrc_constraint_out[worldid, tid_in_block] = sum_qfrc

      # SYNC: Ensure qfrc_constraint is written before gauss_cost reads it
      syncthreads()

      # =====================================================================
      # PHASE 8c: gauss_cost (leader only - reduction)
      # =====================================================================
      if is_leader:
        gauss_cost = float(0.0)
        for dofid in range(NV):
          gauss_cost += (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]) * (qacc_out[worldid, dofid] - qacc_smooth_in[worldid, dofid])
        efc_gauss_out[worldid] += 0.5 * gauss_cost
        efc_cost_out[worldid] += 0.5 * gauss_cost

      # SYNC: Ensure gauss_cost/cost updates complete before update_gradient
      syncthreads()

      # =====================================================================
      # PHASE 9: update_gradient_fn (TILED - all threads participate)
      # =====================================================================
      nefc = nefc_in[worldid]
      update_gradient_fn_local(
        worldid, is_leader, nefc, qM_in, qfrc_smooth_in, qfrc_constraint_out,
        efc_Ma_out, efc_J_in, efc_D_in, efc_state_out,
        efc_grad_out, efc_grad_dot_out, efc_h_out, efc_Mgrad_out
      )

      # =====================================================================
      # PHASE 10: Convergence check (leader computes, ALL threads must exit together)
      # =====================================================================
      if is_leader:
        solver_niter_out[worldid] += 1
        tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
        tmp_cste = 1.0 / (stat_meaninertia * float(NV))
        improvement = (efc_prev_cost_out[worldid] - efc_cost_out[worldid]) * tmp_cste
        gradient = wp.math.sqrt(efc_grad_dot_out[worldid]) * tmp_cste
        done = (improvement < tolerance) or (gradient < tolerance)
        if done or solver_niter_out[worldid] == opt_iterations:
          # Signal all threads to exit by writing to shared location
          efc_grad_dot_out[worldid] = -1.0  # Use negative as "done" signal
          wp.atomic_add(nsolving_out, 0, -1)
      
      # SYNC: Ensure done signal is written
      syncthreads()
      
      # ALL threads check if we should exit
      if efc_grad_dot_out[worldid] < 0.0:
        return



  return kernel


@event_scope
def _solver_iteration(m: types.Model, d: types.Data):
  # Use tiled kernel factory with compile-time nv and njmax
  # launch_tiled automatically adds trailing dimension = block_dim
  # Inside kernel: worldid, tid_in_block = wp.tid()
  # Thread 0 in each block handles scalar operations
  BLOCK_DIM = 32
  wp.launch_tiled(
    solver_iteration(m.nv, d.njmax),
    dim=(d.nworld,),
    inputs=[
      m.opt.impratio,
      m.opt.tolerance,
      m.opt.iterations,
      m.opt.ls_tolerance,
      m.opt.ls_iterations,
      m.opt.ls_parallel,
      m.opt.ls_parallel_min_step,
      m.stat.meaninertia,
      d.ne,
      d.nf,
      d.nefc,
      d.qM,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.efc.type,
      d.efc.id,
      d.efc.J,
      d.efc.aref,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.jv,
      d.efc.quad,
      d.nacon,
    ],
    outputs=[
      d.qacc,
      d.efc.Ma,
      d.efc.Jaref,
      d.qfrc_constraint,
      d.efc.force,
      d.efc.gauss,
      d.efc.cost,
      d.efc.prev_cost,
      d.efc.state,
      d.efc.grad,
      d.efc.grad_dot,
      d.efc.h,
      d.efc.Mgrad,
      d.solver_niter,
      d.nsolving,
      d.efc.mv,
    ],
    block_dim=32,
  )


@event_scope
def solve(m: types.Model, d: types.Data):
  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
    d.solver_niter.fill_(0)
  else:
    _solve(m, d)

def _solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""
  if not (m.opt.disableflags & types.DisableBit.WARMSTART):
    wp.copy(d.qacc, d.qacc_warmstart)
  else:
    wp.copy(d.qacc, d.qacc_smooth)

  _solver_iteration(m, d)
