# Copyright 2021-2025
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

"""
Unrestricted coupled cluster doubles (UCCD).

Amplitude equations derived from Wick's theorem (hast-ucc / uccd_hast.py).
T1 amplitudes are identically zero; only T2 is optimized.
"""

from functools import reduce, partial
import numpy
import jax
from jax.lax import while_loop, custom_root

from pyscf.cc import uccsd as pyscf_uccsd

from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad import config
from pyscfad import ao2mo
from pyscfad.scipy.sparse.linalg import gmres_const_atol

# ---------------------------------------------------------------------------
# ERI attributes that must be marked as dynamic (JAX-traced) leaves.
# These are all tensors accessed in energy() and update_amps().
# ---------------------------------------------------------------------------
ERI_Tracers = (
    'focka', 'fockb', 'mo_energy',
    # alpha-alpha blocks
    'ovov', 'oovv', 'oooo', 'vvvv',
    # beta-beta blocks
    'OVOV', 'OOVV', 'OOOO', 'VVVV',
    # alpha-beta (ab) blocks
    'ovOV', 'ovVO', 'ooOO', 'ooVV', 'vvVV',
    # beta-alpha (ba) blocks
    'OOvv',
)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _uccd_implicit_amp(mycc, amp, eris, diis, max_cycle, tol, tolnormt, verbose):
    oracle = lambda fn, amp0: _iter(amp0, mycc, eris,
                                    diis=diis, max_cycle=max_cycle, tol=tol,
                                    tolnormt=tolnormt, verbose=verbose)

    def root_fn(amp):
        t1, t2 = mycc.vector_to_amplitudes(amp)
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        return mycc.amplitudes_to_vector(t1new, t2new) - amp

    solver = partial(gmres_const_atol,
                     tol=1e-6, atol=1e-6, maxiter=30,
                     solve_method="batched", restart=20)

    def project(vec):
        t1, t2 = mycc.vector_to_amplitudes(vec)
        t2aa, t2ab, t2bb = t2
        t2aa_anti = (t2aa - t2aa.transpose(0, 1, 3, 2))
        t2aa_anti = (t2aa_anti - t2aa_anti.transpose(1, 0, 2, 3)) / 4.0
        t2bb_anti = (t2bb - t2bb.transpose(0, 1, 3, 2))
        t2bb_anti = (t2bb_anti - t2bb_anti.transpose(1, 0, 2, 3)) / 4.0
        return mycc.amplitudes_to_vector(t1, (t2aa_anti, t2ab, t2bb_anti))

    def tangent_solve(g, amp_bar):
        def g_proj(x):
            x_anti = project(x)
            return g(x_anti) + (x - x_anti)
        return solver(g_proj, amp_bar)[0]

    amp_cnvg, conv = custom_root(root_fn, amp, oracle, tangent_solve, has_aux=True)
    return amp_cnvg, conv


def energy(cc, t1=None, t2=None, eris=None):
    """UCCD correlation energy (T1 terms are zero for CCD)."""
    if t1 is None:
        t1 = cc.t1
    if t2 is None:
        t2 = cc.t2
    if eris is None:
        eris = cc.ao2mo()

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    e  =  0.25 * np.einsum('iajb,ijab->', eris.ovov, t2aa)
    e += -0.25 * np.einsum('ibja,ijab->', eris.ovov, t2aa)
    e +=  1.0  * np.einsum('iajb,ijab->', eris.ovOV, t2ab)
    e +=  0.25 * np.einsum('iajb,ijab->', eris.OVOV, t2bb)
    e += -0.25 * np.einsum('ibja,ijab->', eris.OVOV, t2bb)
    return e.real


# ---------------------------------------------------------------------------
# Amplitude update  (T2 residuals, T1 kept at zero)
# ---------------------------------------------------------------------------

@jax.jit
def update_amps(cc, t1, t2, eris):
    """UCCD amplitude update from Wick-theorem equations.

    The T1 amplitudes are passed through unchanged (identically zero).
    Only T2 residuals are computed and accumulated.
    """
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift

    u2aa = np.zeros_like(t2aa)
    u2ab = np.zeros_like(t2ab)
    u2bb = np.zeros_like(t2bb)

    # -----------------------------------------------------------------------
    # Linear-in-T2 residuals  (Wick-theorem equations, icnt=2)
    # -----------------------------------------------------------------------

    # --- bare two-electron integrals (zeroth order in T2) ---
    u2aa +=  0.25 * np.einsum('iajb->ijab', eris.ovov)
    u2aa += -0.25 * np.einsum('ibja->ijab', eris.ovov)
    u2ab +=  1.0  * np.einsum('iajb->ijab', eris.ovOV)
    u2bb +=  0.25 * np.einsum('iajb->ijab', eris.OVOV)
    u2bb += -0.25 * np.einsum('ibja->ijab', eris.OVOV)

    # --- Fock-matrix one-body contributions ---
    u2aa += -0.5  * np.einsum('ac,ijbc->ijab', eris.focka[nocca:, nocca:], t2aa)
    u2aa +=  0.5  * np.einsum('ki,jkab->ijab', eris.focka[:nocca, :nocca], t2aa)
    u2ab +=  1.0  * np.einsum('ab,ijbc->ijac', eris.focka[nocca:, nocca:], t2ab)
    u2ab += -1.0  * np.einsum('ji,jkab->ikab', eris.focka[:nocca, :nocca], t2ab)
    u2ab +=  1.0  * np.einsum('ac,ijbc->ijba', eris.fockb[noccb:, noccb:], t2ab)
    u2ab += -1.0  * np.einsum('ki,jkab->jiab', eris.fockb[:noccb, :noccb], t2ab)
    u2bb += -0.5  * np.einsum('ac,ijbc->ijab', eris.fockb[noccb:, noccb:], t2bb)
    u2bb +=  0.5  * np.einsum('ki,jkab->ijab', eris.fockb[:noccb, :noccb], t2bb)

    # --- vvvv / VVVV / vvVV contractions ---
    u2aa +=  0.125 * np.einsum('acbd,ijcd->ijab', eris.vvvv, t2aa)
    u2aa += -0.125 * np.einsum('adbc,ijcd->ijab', eris.vvvv, t2aa)
    u2ab +=  1.0   * np.einsum('acbd,ijcd->ijab', eris.vvVV, t2ab)
    u2bb +=  0.125 * np.einsum('acbd,ijcd->ijab', eris.VVVV, t2bb)
    u2bb += -0.125 * np.einsum('adbc,ijcd->ijab', eris.VVVV, t2bb)

    # --- ovov / OVOV / ovVO contractions ---
    u2aa +=  1.0 * np.einsum('iakc,jkbc->ijab', eris.ovov, t2aa)
    u2aa += -1.0 * np.einsum('kiac,jkbc->ijab', eris.oovv, t2aa)
    u2ab +=  1.0 * np.einsum('iajb,jkbc->ikac', eris.ovov, t2ab)
    u2ab += -1.0 * np.einsum('jiab,jkbc->ikac', eris.oovv, t2ab)
    u2aa +=  1.0 * np.einsum('iack,jkbc->ijab', eris.ovVO, t2ab)
    u2ab +=  1.0 * np.einsum('iack,jkbc->ijab', eris.ovVO, t2bb)
    u2ab +=  1.0 * np.einsum('kcai,jkbc->jiba', eris.ovVO, t2aa)
    u2ab +=  1.0 * np.einsum('iakc,jkbc->jiba', eris.OVOV, t2ab)
    u2ab += -1.0 * np.einsum('kiac,jkbc->jiba', eris.OOVV, t2ab)
    u2bb +=  1.0 * np.einsum('jbai,jkbc->ikac', eris.ovVO, t2ab)
    u2bb +=  1.0 * np.einsum('iakc,jkbc->ijab', eris.OVOV, t2bb)
    u2bb += -1.0 * np.einsum('kiac,jkbc->ijab', eris.OOVV, t2bb)

    # --- oooo / OOOO / ooOO contractions ---
    u2aa +=  0.125 * np.einsum('kilj,klab->ijab', eris.oooo, t2aa)
    u2aa += -0.125 * np.einsum('kjli,klab->ijab', eris.oooo, t2aa)
    u2ab += -1.0   * np.einsum('jiac,jkbc->ikba', eris.ooVV, t2ab)
    u2ab +=  1.0   * np.einsum('kilj,klab->ijab', eris.ooOO, t2ab)
    u2ab += -1.0   * np.einsum('kiab,jkbc->jiac', eris.OOvv, t2ab)
    u2bb +=  0.125 * np.einsum('kilj,klab->ijab', eris.OOOO, t2bb)
    u2bb += -0.125 * np.einsum('kjli,klab->ijab', eris.OOOO, t2bb)

    # -----------------------------------------------------------------------
    # Quadratic-in-T2 residuals  (T2 x T2 terms from Wick's theorem)
    # -----------------------------------------------------------------------

    # --- aa x aa ---
    u2aa += -0.25   * np.einsum('kbld,ijab,klcd->ijac', eris.ovov, t2aa, t2aa)
    u2aa +=  0.25   * np.einsum('kdlb,ijab,klcd->ijac', eris.ovov, t2aa, t2aa)
    u2aa +=  0.0625 * np.einsum('kalb,ijab,klcd->ijcd', eris.ovov, t2aa, t2aa)
    u2aa += -0.0625 * np.einsum('kbla,ijab,klcd->ijcd', eris.ovov, t2aa, t2aa)
    u2aa += -0.25   * np.einsum('jcld,ijab,klcd->ikab', eris.ovov, t2aa, t2aa)
    u2aa +=  0.25   * np.einsum('jdlc,ijab,klcd->ikab', eris.ovov, t2aa, t2aa)
    u2aa +=  0.5    * np.einsum('jbld,ijab,klcd->ikac', eris.ovov, t2aa, t2aa)
    u2aa += -0.5    * np.einsum('jdlb,ijab,klcd->ikac', eris.ovov, t2aa, t2aa)

    # --- ab x ab ---
    u2ab += -1.0 * np.einsum('kclb,ijab,klcd->ijad', eris.ovOV, t2ab, t2ab)
    u2ab += -1.0 * np.einsum('kcjd,ijab,klcd->ilab', eris.ovOV, t2ab, t2ab)
    u2ab +=  1.0 * np.einsum('kcjb,ijab,klcd->ilad', eris.ovOV, t2ab, t2ab)
    u2ab += -1.0 * np.einsum('kald,ijab,klcd->ijcb', eris.ovOV, t2ab, t2ab)
    u2ab +=  1.0 * np.einsum('kalb,ijab,klcd->ijcd', eris.ovOV, t2ab, t2ab)
    u2ab +=  1.0 * np.einsum('kajd,ijab,klcd->ilcb', eris.ovOV, t2ab, t2ab)
    u2ab += -1.0 * np.einsum('kajb,ijab,klcd->ilcd', eris.ovOV, t2ab, t2ab)

    # --- aa x ab cross terms ---
    u2aa +=  1.0 * np.einsum('ldjb,ijab,klcd->ikac', eris.ovOV, t2ab, t2aa)
    u2aa +=  0.5 * np.einsum('jbld,ijab,klcd->ikac', eris.OVOV, t2ab, t2ab)
    u2aa += -0.5 * np.einsum('jdlb,ijab,klcd->ikac', eris.OVOV, t2ab, t2ab)
    u2aa +=  0.5 * np.einsum('lajb,ijab,klcd->ikcd', eris.ovOV, t2ab, t2aa)
    u2aa +=  0.5 * np.einsum('idjb,ijab,klcd->klac', eris.ovOV, t2ab, t2aa)
    u2ab += -0.5 * np.einsum('kald,ijab,klcd->ijcb', eris.ovov, t2ab, t2aa)
    u2ab +=  0.5 * np.einsum('kdla,ijab,klcd->ijcb', eris.ovov, t2ab, t2aa)
    u2ab += -0.5 * np.einsum('icld,ijab,klcd->kjab', eris.ovov, t2ab, t2aa)
    u2ab +=  0.5 * np.einsum('idlc,ijab,klcd->kjab', eris.ovov, t2ab, t2aa)
    u2ab +=  1.0 * np.einsum('iald,ijab,klcd->kjcb', eris.ovov, t2ab, t2aa)
    u2ab += -1.0 * np.einsum('idla,ijab,klcd->kjcb', eris.ovov, t2ab, t2aa)
    u2bb +=  0.5 * np.einsum('iakc,ijab,klcd->jlbd', eris.ovov, t2ab, t2ab)
    u2bb += -0.5 * np.einsum('icka,ijab,klcd->jlbd', eris.ovov, t2ab, t2ab)

    # --- bb x ab cross terms ---
    u2bb += -0.5 * np.einsum('kclb,ijab,klcd->ijad', eris.ovOV, t2bb, t2ab)
    u2bb += -0.5 * np.einsum('kcjd,ijab,klcd->ilab', eris.ovOV, t2bb, t2ab)
    u2bb +=  1.0 * np.einsum('kcjb,ijab,klcd->ilad', eris.ovOV, t2bb, t2ab)
    u2ab +=  1.0 * np.einsum('ldjb,ijab,klcd->kica', eris.ovOV, t2bb, t2aa)
    u2ab +=  1.0 * np.einsum('jbld,ijab,klcd->kica', eris.OVOV, t2bb, t2ab)
    u2ab += -1.0 * np.einsum('jdlb,ijab,klcd->kica', eris.OVOV, t2bb, t2ab)
    u2ab +=  0.5 * np.einsum('jalb,ijab,klcd->kicd', eris.OVOV, t2bb, t2ab)
    u2ab += -0.5 * np.einsum('jbla,ijab,klcd->kicd', eris.OVOV, t2bb, t2ab)
    u2ab +=  0.5 * np.einsum('ibjd,ijab,klcd->klca', eris.OVOV, t2bb, t2ab)
    u2ab += -0.5 * np.einsum('idjb,ijab,klcd->klca', eris.OVOV, t2bb, t2ab)

    # --- bb x bb ---
    u2bb += -0.25   * np.einsum('kbld,ijab,klcd->ijac', eris.OVOV, t2bb, t2bb)
    u2bb +=  0.25   * np.einsum('kdlb,ijab,klcd->ijac', eris.OVOV, t2bb, t2bb)
    u2bb +=  0.0625 * np.einsum('kalb,ijab,klcd->ijcd', eris.OVOV, t2bb, t2bb)
    u2bb += -0.0625 * np.einsum('kbla,ijab,klcd->ijcd', eris.OVOV, t2bb, t2bb)
    u2bb += -0.25   * np.einsum('jcld,ijab,klcd->ikab', eris.OVOV, t2bb, t2bb)
    u2bb +=  0.25   * np.einsum('jdlc,ijab,klcd->ikab', eris.OVOV, t2bb, t2bb)
    u2bb +=  0.5    * np.einsum('jbld,ijab,klcd->ikac', eris.OVOV, t2bb, t2bb)
    u2bb += -0.5    * np.einsum('jdlb,ijab,klcd->ikac', eris.OVOV, t2bb, t2bb)

    # -----------------------------------------------------------------------
    # Antisymmetrize T2aa and T2bb
    # -----------------------------------------------------------------------
    u2aa = u2aa - u2aa.transpose(0, 1, 3, 2)
    u2aa = u2aa - u2aa.transpose(1, 0, 2, 3)
    u2bb = u2bb - u2bb.transpose(0, 1, 3, 2)
    u2bb = u2bb - u2bb.transpose(1, 0, 2, 3)

    # -----------------------------------------------------------------------
    # Divide by orbital-energy denominators  (update step, not residual)
    # -----------------------------------------------------------------------
    eia_a = mo_ea_o[:, None] - mo_ea_v[None, :]
    eia_b = mo_eb_o[:, None] - mo_eb_v[None, :]
    eijab_aa = eia_a[:, None, :, None] + eia_a[None, :, None, :]
    eijab_ab = eia_a[:, None, :, None] + eia_b[None, :, None, :]
    eijab_bb = eia_b[:, None, :, None] + eia_b[None, :, None, :]

    u2aa /= eijab_aa
    u2ab /= eijab_ab
    u2bb /= eijab_bb

    return (t1a, t1b), (t2aa + u2aa, t2ab + u2ab, t2bb + u2bb)


# ---------------------------------------------------------------------------
# Amplitude vector packing / unpacking  (JAX-compatible, t1 always zero)
# ---------------------------------------------------------------------------

def amplitudes_to_vector(t1, t2, out=None):
    """Pack (t1, t2) into a flat vector.  t1 is zero for UCCD and omitted."""
    t2aa, t2ab, t2bb = t2
    return np.concatenate([t2aa.ravel(), t2ab.ravel(), t2bb.ravel()])


def vector_to_amplitudes(vec, nmo, nocc):
    """Unpack flat vector back into (t1, t2).  t1 is returned as zeros."""
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    n_aa = nocca * nocca * nvira * nvira
    n_ab = nocca * noccb * nvira * nvirb
    t2aa = vec[:n_aa].reshape(nocca, nocca, nvira, nvira)
    t2ab = vec[n_aa:n_aa + n_ab].reshape(nocca, noccb, nvira, nvirb)
    t2bb = vec[n_aa + n_ab:].reshape(noccb, noccb, nvirb, nvirb)
    t1a = np.zeros((nocca, nvira), dtype=vec.dtype)
    t1b = np.zeros((noccb, nvirb), dtype=vec.dtype)
    return (t1a, t1b), (t2aa, t2ab, t2bb)


# ---------------------------------------------------------------------------
# Kernel (fixed-point iteration + implicit differentiation)
# ---------------------------------------------------------------------------

def _iter(amp, mycc, eris, diis,
          max_cycle=50, tol=1e-8, tolnormt=1e-6, verbose=None):
    """DIIS-accelerated fixed-point iteration to convergence."""
    log = logger.new_logger(mycc, verbose)

    t1, t2 = mycc.vector_to_amplitudes(amp)
    eold = 0.
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(UCCD) = %.15g', eccsd)

    diis_space = mycc.diis_space
    diis_start_cycle = mycc.diis_start_cycle
    diis_start_energy_diff = mycc.diis_start_energy_diff
    if mycc.diis and mycc.diis is not True:
        raise NotImplementedError
    use_diis = bool(mycc.diis)

    amp_hist = np.zeros((diis_space, amp.size))
    err_hist = np.zeros((diis_space, amp.size))

    def cond_fun(value):
        istep, conv, eold, eccsd, normt, amp, amp_hist, err_hist = value
        return (istep < max_cycle) & (np.logical_not(conv))

    def body_fun(value):
        istep, conv, eold, eccsd, normt, amp, amp_hist, err_hist = value
        t1, t2 = mycc.vector_to_amplitudes(amp)
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        vec_new = mycc.amplitudes_to_vector(t1new, t2new)
        vec_old = mycc.amplitudes_to_vector(t1, t2)
        normt = np.linalg.norm(vec_new - vec_old)
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            vec_new = (1. - alpha) * vec_old + alpha * vec_new

        err = vec_new - vec_old
        idx = istep % diis_space
        amp_hist = amp_hist.at[idx].set(vec_new)
        err_hist = err_hist.at[idx].set(err)

        nd = np.minimum(istep + 1, diis_space)
        B = np.zeros((diis_space + 1, diis_space + 1))
        B = B.at[1:, 1:].set(np.dot(err_hist, err_hist.T))
        mask = np.arange(diis_space) < nd
        B = B.at[0, 1:].set(np.where(mask, 1.0, 0.0))
        B = B.at[1:, 0].set(np.where(mask, 1.0, 0.0))
        B = B.at[1:, 1:].add(np.diag(np.where(mask, 0.0, 1.0)))

        g = np.zeros(diis_space + 1)
        g = g.at[0].set(1.0)

        w, v = np.linalg.eigh(B)
        idx_w = abs(w) > 1e-14
        c = np.dot(v, np.where(idx_w, 1.0/w, 0.0) * np.dot(v.T.conj(), g))
        vec_diis = np.dot(c[1:], amp_hist)

        do_diis = use_diis & (istep >= diis_start_cycle) & \
                  (abs(eccsd - eold) < diis_start_energy_diff) & (nd >= 2)
        amp = np.where(do_diis, vec_diis, vec_new)

        t1, t2 = mycc.vector_to_amplitudes(amp)
        eold = eccsd
        eccsd = mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(UCCD) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep + 1, eccsd, eccsd - eold, normt)
        conv = ((abs(eccsd - eold) < tol) & (normt < tolnormt)).astype(float)
        return istep + 1, conv, eold, eccsd, normt, amp, amp_hist, err_hist

    init_val = (0, 0., eold, eccsd, 1.0, amp, amp_hist, err_hist)
    istep, conv, eold, eccsd, normt, amp, amp_hist, err_hist = while_loop(cond_fun, body_fun, init_val)

    del log
    return amp, conv


def kernel(mycc, eris=None, t1=None, t2=None,
           max_cycle=50, tol=1e-8, tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        _, t1, t2 = mycc.init_amps(eris)
    elif t2 is None:
        _, _, t2 = mycc.init_amps(eris)

    vec = mycc.amplitudes_to_vector(t1, t2)

    if config.ccsd_implicit_diff:
        adiis = mycc.diis
        vec, conv = _uccd_implicit_amp(mycc, vec, eris, adiis, max_cycle, tol, tolnormt, log)
    else:
        if isinstance(mycc.diis, lib.diis.DIIS):
            adiis = mycc.diis
        elif mycc.diis:
            adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
            adiis.space = mycc.diis_space
        else:
            adiis = None
        vec, conv = _iter(vec, mycc, eris,
                          diis=adiis, max_cycle=max_cycle, tol=tol,
                          tolnormt=tolnormt, verbose=log)

    t1, t2 = mycc.vector_to_amplitudes(vec)
    eccsd = mycc.energy(t1, t2, eris)
    log.timer('UCCD')
    del adiis, log
    return conv.astype(bool), eccsd, t1, t2


# ---------------------------------------------------------------------------
# UCCD class
# ---------------------------------------------------------------------------

class UCCD(pytree.PytreeNode, pyscf_uccsd.UCCSD):
    """Differentiable unrestricted CCD (doubles only).

    Inherits from pyscf UCCSD for frozen-orbital handling, DIIS, etc.
    T1 amplitudes are identically zero throughout.
    """
    _dynamic_attr = {'_scf'}

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocca, noccb = self.nocc

        mo_ea_o = eris.mo_energy[0][:nocca]
        mo_ea_v = eris.mo_energy[0][nocca:]
        mo_eb_o = eris.mo_energy[1][:noccb]
        mo_eb_v = eris.mo_energy[1][noccb:]

        eia_a = mo_ea_o[:, None] - mo_ea_v[None, :]
        eia_b = mo_eb_o[:, None] - mo_eb_v[None, :]
        eijab_aa = eia_a[:, None, :, None] + eia_a[None, :, None, :]
        eijab_ab = eia_a[:, None, :, None] + eia_b[None, :, None, :]
        eijab_bb = eia_b[:, None, :, None] + eia_b[None, :, None, :]

        t2aa = eris.ovov.transpose(0, 2, 1, 3) / eijab_aa
        t2aa -= t2aa.transpose(0, 1, 3, 2)
        t2ab = eris.ovOV.transpose(0, 2, 1, 3) / eijab_ab
        t2bb = eris.OVOV.transpose(0, 2, 1, 3) / eijab_bb
        t2bb -= t2bb.transpose(0, 1, 3, 2)

        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        t1a = np.zeros((nocca, nvira))
        t1b = np.zeros((noccb, nvirb))

        emp2 = (0.25 * np.einsum('ijab,iajb->', t2aa, eris.ovov)
                - 0.25 * np.einsum('ijab,ibja->', t2aa, eris.ovov)
                + 1.0  * np.einsum('ijab,iajb->', t2ab, eris.ovOV)
                + 0.25 * np.einsum('ijab,iajb->', t2bb, eris.OVOV)
                - 0.25 * np.einsum('ijab,ibja->', t2bb, eris.OVOV))
        self.emp2 = emp2.real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, (t1a, t1b), (t2aa, t2ab, t2bb)

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nmo is None:
            nmo = self.nmo
        if nocc is None:
            nocc = self.nocc
        return vector_to_amplitudes(vec, nmo, nocc)

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            raise NotImplementedError
        assert self.mo_coeff is not None
        assert self.mo_occ is not None

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
            kernel(self, eris, t1, t2,
                   max_cycle=self.max_cycle,
                   tol=self.conv_tol,
                   tolnormt=self.conv_tol_normt,
                   verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _make_eris_incore(self, mo_coeff)

    def _finalize(self):
        logger.info(self, '%s converged = %s', self.__class__.__name__, self.converged)
        logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    energy = energy
    update_amps = update_amps


# ---------------------------------------------------------------------------
# ERIs container
# ---------------------------------------------------------------------------

class _ChemistsERIs(pytree.PytreeNode, pyscf_uccsd._ChemistsERIs):
    _dynamic_attr = ERI_Tracers

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = (
            mo_coeff[0][:, mo_idx[0]],
            mo_coeff[1][:, mo_idx[1]],
        )

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.focka = reduce(np.dot,
                            (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(np.dot,
                            (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)

        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        return self


def _make_eris_incore(mycc, mo_coeff=None):
    """Build in-core ERIs for UCCD using pyscfad-differentiable ao2mo."""
    log = logger.new_logger(mycc)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = eris.nocc
    moa, mob = eris.mo_coeff
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    eri_s4 = mycc._scf._eri   # stored in 4-fold (nao_pair, nao_pair) symmetry

    eri_aa = ao2mo.incore.general(eri_s4, (moa,) * 4, compact=False)
    eri_bb = ao2mo.incore.general(eri_s4, (mob,) * 4, compact=False)
    eri_ab = ao2mo.incore.general(eri_s4, (moa, moa, mob, mob), compact=False)

    eri_aa = eri_aa.reshape(nmoa, nmoa, nmoa, nmoa)
    eri_bb = eri_bb.reshape(nmob, nmob, nmob, nmob)
    eri_ab = eri_ab.reshape(nmoa, nmoa, nmob, nmob)
    eri_ba = eri_ab.transpose(2, 3, 0, 1)   # (rs|pq) = (pq|rs) for real ERIs

    # alpha-alpha blocks
    eris.oooo = eri_aa[:nocca, :nocca, :nocca, :nocca]
    eris.ovoo = eri_aa[:nocca, nocca:, :nocca, :nocca]
    eris.ovov = eri_aa[:nocca, nocca:, :nocca, nocca:]
    eris.oovv = eri_aa[:nocca, :nocca, nocca:, nocca:]
    eris.ovvo = eri_aa[:nocca, nocca:, nocca:, :nocca]
    eris.ovvv = eri_aa[:nocca, nocca:, nocca:, nocca:]
    eris.vvvv = eri_aa[nocca:, nocca:, nocca:, nocca:]

    # beta-beta blocks
    eris.OOOO = eri_bb[:noccb, :noccb, :noccb, :noccb]
    eris.OVOO = eri_bb[:noccb, noccb:, :noccb, :noccb]
    eris.OVOV = eri_bb[:noccb, noccb:, :noccb, noccb:]
    eris.OOVV = eri_bb[:noccb, :noccb, noccb:, noccb:]
    eris.OVVO = eri_bb[:noccb, noccb:, noccb:, :noccb]
    eris.OVVV = eri_bb[:noccb, noccb:, noccb:, noccb:]
    eris.VVVV = eri_bb[noccb:, noccb:, noccb:, noccb:]

    # alpha-beta (ab) blocks
    eris.ooOO = eri_ab[:nocca, :nocca, :noccb, :noccb]
    eris.ovOO = eri_ab[:nocca, nocca:, :noccb, :noccb]
    eris.ovOV = eri_ab[:nocca, nocca:, :noccb, noccb:]
    eris.ooVV = eri_ab[:nocca, :nocca, noccb:, noccb:]
    eris.ovVO = eri_ab[:nocca, nocca:, noccb:, :noccb]
    eris.ovVV = eri_ab[:nocca, nocca:, noccb:, noccb:]
    eris.vvVV = eri_ab[nocca:, nocca:, noccb:, noccb:]

    # beta-alpha (ba) blocks
    eris.OOoo = eri_ba[:noccb, :noccb, :nocca, :nocca]
    eris.OVoo = eri_ba[:noccb, noccb:, :nocca, :nocca]
    eris.OVov = eri_ba[:noccb, noccb:, :nocca, nocca:]
    eris.OOvv = eri_ba[:noccb, :noccb, nocca:, nocca:]
    eris.OVvo = eri_ba[:noccb, noccb:, nocca:, :nocca]
    eris.OVvv = eri_ba[:noccb, noccb:, nocca:, nocca:]
    eris.VVvv = eri_ba[noccb:, noccb:, nocca:, nocca:]

    log.timer('UCCD integral transformation')
    del log
    return eris
