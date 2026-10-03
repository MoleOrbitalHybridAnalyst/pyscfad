# Copyright 2025-2026 The PySCFAD Authors
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

import copy
import numpy
import pytest
import jax
from pyscfad import numpy as np
from pyscfad.gto import MoleLite as Mole
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1Param
from pyscfad.xtb.alchem import AlchemGFN1XTB
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges

@pytest.fixture
def setup():
    basis = xtb_basis.get_basis_filename()
    param = GFN1Param()
    # H3O+ (Bohr); the last H is the alchemical proton
    numbers = numpy.array([8, 1, 1, 1])
    coords = np.array(
        [
            [ 0.00000, 0.00000,  0.00000],
            [ 1.43355, 0.00000, -0.95296],
            [ 1.43355, 0.00000,  0.95296],
            [-0.80000, 0.10000,  1.60000],
        ]
    )
    yield basis, param, numbers, coords

@pytest.fixture
def mm_setup():
    a = np.eye(3) * 12.0
    mm_coords = np.array([[5.0, 0.5, 0.3], [-4.0, 1.0, -2.0]])
    mm_charges = np.array([0.4, -0.4])
    mm_radii = np.array([1.2, 1.2])
    yield a, mm_coords, mm_charges, mm_radii

# NOTE energies are jitted: eager execution compiles op by op and is much slower
def _ref_energy(basis, param, numbers, coords, charge, qmmm_fn=None):
    @jax.jit
    def energy():
        mol = Mole(numbers=numbers, coords=coords, basis=basis, charge=charge, verbose=0)
        mf = GFN1XTB(mol, param=param)
        mf.param = copy.copy(mf.param)
        if qmmm_fn is not None:
            mf = qmmm_fn(mf)
        return mf.kernel()
    return energy()

def _make_alchem_energy(basis, param, numbers, diis=None, qmmm_fn=None, **alchem_kwargs):
    def energy(coords, lam):
        mol = Mole(numbers=numbers, coords=coords, basis=basis, charge=1, verbose=0,
                   trace_coords=True)
        mf = AlchemGFN1XTB(mol, param, lam=lam, alchem_atoms=(3,), **alchem_kwargs)
        if qmmm_fn is not None:
            mf = qmmm_fn(mf)
        mf.diis = diis
        return mf.kernel()
    return jax.jit(energy), jax.jit(jax.grad(energy, argnums=(0, 1)))

def _check_grads(energy, grad, coords, lam=0.3, eps=1e-4, tol=1e-6):
    """Nuclear (along a random direction) and lambda gradients against finite differences."""
    g_coords, g_lam = grad(coords, lam)

    g_fd = (energy(coords, lam + eps) - energy(coords, lam - eps)) / (2 * eps)
    assert abs(g_lam - g_fd) < tol

    d = numpy.random.default_rng(1).standard_normal(coords.shape)
    d /= numpy.linalg.norm(d)
    g_fd = (energy(coords + eps * d, lam) - energy(coords - eps * d, lam)) / (2 * eps)
    assert abs(np.sum(g_coords * d) - g_fd) < tol

def test_alchem_proton_endpoints_and_grad(setup):
    basis, param, numbers, coords = setup
    e_h2o = _ref_energy(basis, param, numbers[:3], coords[:3], 0)
    e_h3o = _ref_energy(basis, param, numbers, coords, 1)
    for diis in (None, "qbroyden"):
        energy, grad = _make_alchem_energy(basis, param, numbers, diis)
        assert abs(energy(coords, 0.) - e_h2o) < 1e-8
        assert abs(energy(coords, 1.) - e_h3o) < 1e-8
        _check_grads(energy, grad, coords)

def test_alchem_proton_separate_penalty_switch(setup):
    basis, param, numbers, coords = setup

    # default penalty switch is 1 - onsite
    e_default, _ = _make_alchem_energy(basis, param, numbers, penalty=.5)
    e_explicit, _ = _make_alchem_energy(basis, param, numbers, penalty=.5,
                                        switch={"penalty": lambda x: 1 - x})
    assert abs(e_default(coords, .3) - e_explicit(coords, .3)) < 1e-12

    # separate on-site scaling g and penalty weight p keep exact endpoints
    switch = {"onsite": lambda x: x * x, "penalty": lambda x: (1 - x)**2}
    energy, grad = _make_alchem_energy(basis, param, numbers, penalty=.5, switch=switch)
    e_h2o = _ref_energy(basis, param, numbers[:3], coords[:3], 0)
    e_h3o = _ref_energy(basis, param, numbers, coords, 1)
    assert abs(energy(coords, 0.) - e_h2o) < 1e-8
    assert abs(energy(coords, 1.) - e_h3o) < 1e-8
    _check_grads(energy, grad, coords)

def test_alchem_proton_qmmm_multipoles(setup, mm_setup):
    basis, param, numbers, coords = setup
    a, mm_coords, mm_charges, mm_radii = mm_setup

    # Ewald meshes set array shapes, so compute them up front and pass them as
    # concrete values; sharing them also makes the lam=0 reference comparable
    @jax.jit
    def ewald_params():
        mol = Mole(numbers=numbers, coords=coords, basis=basis, charge=1, verbose=0)
        mf = add_mm_charges(GFN1XTB(mol, param=param), mm_coords, a, mm_charges, mm_radii,
                            unit="Bohr", mm_ew_rcut=5.0)
        return mf.mm_ew_eta, mf.mm_ew_mesh, mf.qm_ew_mesh
    mm_ew_eta, mm_ew_mesh, qm_ew_mesh = ewald_params()
    ew_kwargs = {
        "mm_ew_rcut": 5.0,
        "mm_ew_eta": float(mm_ew_eta),
        "mm_ew_mesh": numpy.asarray(mm_ew_mesh),
        "qm_ew_mesh": numpy.asarray(qm_ew_mesh),
    }

    def qmmm_fn(mf):
        # turn on the QM-MM dipolar and quadrupolar couplings
        natm = mf.mol.natm
        mf.param.dipgam = np.full(natm, 0.6)
        mf.param.quadgam = np.full(natm, 0.4)
        return add_mm_charges(mf, mm_coords, a, mm_charges, mm_radii,
                              unit="Bohr", **ew_kwargs)

    energy, grad = _make_alchem_energy(basis, param, numbers, qmmm_fn=qmmm_fn)

    # lam=1 is plain GFN1 by construction (checked in the test above)
    e_h2o = _ref_energy(basis, param, numbers[:3], coords[:3], 0, qmmm_fn)
    assert abs(energy(coords, 0.) - e_h2o) < 1e-6

    _check_grads(energy, grad, coords)
