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

import warp as wp

from . import derivative
from . import forward
from . import sensor
from . import smooth
from . import solver
from . import support
from .support import mul_m
from .types import Data
from .types import DisableBit
from .types import EnableBit
from .types import IntegratorType
from .types import Model

wp.set_module_options({"enable_backward": False})



def _update_constraint(m: Model, d: Data):
  """Update constraint arrays after each solve iteration."""
  wp.launch(
    solver.update_constraint_kernel,
    dim=(d.nworld),
    inputs=[
      m.nv,
      m.opt.impratio,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.qacc,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.efc.Ma,
      d.efc.J,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.cost,
      d.efc.done,
      d.nacon,
    ],
    outputs=[
      d.qfrc_constraint,
      d.efc.force,
      d.efc.gauss,
      d.efc.cost,
      d.efc.prev_cost,
      d.efc.state,
    ],
  )


@wp.kernel
def solve_init_jaref(
  # Model:
  nv: int,
  # Data in:
  nefc_in: wp.array(dtype=int),
  qacc_in: wp.array2d(dtype=float),
  efc_J_in: wp.array3d(dtype=float),
  efc_aref_in: wp.array2d(dtype=float),
  # Data out:
  efc_Jaref_out: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  jaref = float(0.0)
  for i in range(nv):
    jaref += efc_J_in[worldid, efcid, i] * qacc_in[worldid, i]

  efc_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]


@wp.kernel
def solve_init_efc(
  # Data out:
  solver_niter_out: wp.array(dtype=int),
  efc_search_dot_out: wp.array(dtype=float),
  efc_cost_out: wp.array(dtype=float),
  efc_done_out: wp.array(dtype=bool),
):
  worldid = wp.tid()
  efc_cost_out[worldid] = wp.inf
  solver_niter_out[worldid] = 0
  efc_done_out[worldid] = False
  efc_search_dot_out[worldid] = 0.0


def create_context(m: Model, d: Data):
  # initialize some efc arrays
  wp.launch(
    solve_init_efc,
    dim=(d.nworld),
    outputs=[d.solver_niter, d.efc.search_dot, d.efc.cost, d.efc.done],
  )

  # jaref = d.efc_J @ d.qacc - d.efc_aref
  wp.launch(
    solve_init_jaref,
    dim=(d.nworld, d.njmax),
    inputs=[m.nv, d.nefc, d.qacc, d.efc.J, d.efc.aref],
    outputs=[d.efc.Jaref],
  )

  # Ma = qM @ qacc
  support.mul_m(m, d, d.efc.Ma, d.qacc, skip=d.efc.done)

  _update_constraint(m, d)


@wp.kernel
def _qfrc_eulerdamp(
  # Model:
  opt_timestep: wp.array(dtype=float),
  dof_damping: wp.array2d(dtype=float),
  # Data in:
  qacc_in: wp.array2d(dtype=float),
  # Out:
  qfrc_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qfrc_out[worldid, dofid] += timestep * dof_damping[worldid % dof_damping.shape[0], dofid] * qacc_in[worldid, dofid]


@wp.kernel
def _qfrc_inverse(
  # Data in:
  qfrc_bias_in: wp.array2d(dtype=float),
  qfrc_passive_in: wp.array2d(dtype=float),
  qfrc_constraint_in: wp.array2d(dtype=float),
  # In:
  Ma: wp.array2d(dtype=float),
  # Data out:
  qfrc_inverse_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  qfrc_inverse = qfrc_bias_in[worldid, dofid]
  qfrc_inverse += Ma[worldid, dofid]
  qfrc_inverse -= qfrc_passive_in[worldid, dofid]
  qfrc_inverse -= qfrc_constraint_in[worldid, dofid]

  qfrc_inverse_out[worldid, dofid] = qfrc_inverse


def discrete_acc(m: Model, d: Data, qacc: wp.array2d(dtype=float)):
  """Convert discrete-time qacc to continuous-time qacc.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    qacc: Acceleration.
  """
  qfrc = wp.empty((d.nworld, m.nv), dtype=float)

  if m.opt.integrator == IntegratorType.RK4:
    raise NotImplementedError("discrete inverse dynamics is not supported by RK4 integrator")
  elif m.opt.integrator == IntegratorType.EULER:
    if m.opt.disableflags & DisableBit.EULERDAMP:
      wp.copy(qacc, d.qacc)
      return

    # TODO(team): qacc = d.qacc if (m.dof_damping == 0.0).all()

    # set qfrc = (d.qM + m.opt.timestep * diag(m.dof_damping)) * d.qacc

    # d.qM @ d.qacc
    support.mul_m(m, d, qfrc, d.qacc)

    # qfrc += m.opt.timestep * m.dof_damping * d.qacc
    wp.launch(
      _qfrc_eulerdamp,
      dim=(d.nworld, m.nv),
      inputs=[m.opt.timestep, m.dof_damping, d.qacc],
      outputs=[qfrc],
    )
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    if m.opt.is_sparse:
      qDeriv = wp.empty((d.nworld, 1, m.nM), dtype=float)
    else:
      qDeriv = wp.empty((d.nworld, m.nv, m.nv), dtype=float)
    derivative.deriv_smooth_vel(m, d, qDeriv)
    mul_m(m, d, qfrc, d.qacc, M=qDeriv)
    smooth.factor_solve_i(m, d, d.qM, d.qLD, d.qLDiagInv, qacc, qfrc)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")

  # solve for qacc: qfrc = d.qM @ d.qacc
  smooth.solve_m(m, d, qacc, qfrc)


def inv_constraint(m: Model, d: Data):
  """Inverse constraint solver."""
  # no constraints
  if d.njmax == 0:
    d.qfrc_constraint.zero_()
    return

  # update
  create_context(m, d)


def inverse(m: Model, d: Data):
  """Inverse dynamics."""
  forward.fwd_position(m, d)
  sensor.sensor_pos(m, d)
  forward.fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  invdiscrete = m.opt.enableflags & EnableBit.INVDISCRETE
  if invdiscrete:
    # save discrete-time qacc and compute continuous-time qacc
    qacc_discrete = wp.clone(d.qacc)
    discrete_acc(m, d, d.qacc)

  inv_constraint(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)
  sensor.sensor_acc(m, d)

  support.mul_m(m, d, d.qfrc_inverse, d.qacc)

  wp.launch(
    _qfrc_inverse,
    dim=(d.nworld, m.nv),
    inputs=[
      d.qfrc_bias,
      d.qfrc_passive,
      d.qfrc_constraint,
      d.qfrc_inverse,
    ],
    outputs=[d.qfrc_inverse],
  )

  if invdiscrete:
    # restore discrete-time qacc
    wp.copy(d.qacc, qacc_discrete)
