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
from pyscfad.xtb.alchem import AlchemGFN1XTB, AlchemGFN1KXTB
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges
from pyscfad.xtb.kxtb import GFN1KXTB
from pyscfad.xtb.util import ke_cutoff_ewald
from pyscfad.pbc.gto import CellLite as Cell
from pyscfad.ml.gto import MolePad, make_basis_array
from pyscfad.ml.xtb import make_param_array
from pyscfad.ml.xtb import alchem as alchem_pad
from pyscfad.ml.pbc.gto import CellPad
from pyscfad.ml.pbc.gto.cell_pad import make_image_grid

# NH4+ (Bohr); the last H is the alchemical proton
NH4_NUMBERS = numpy.array([7, 1, 1, 1, 1])
NH4_COORDS = numpy.array([[0., 0., 0.], [1.9, 0., 0.], [-0.63, 1.79, 0.],
                          [-0.63, -0.9, 1.55], [-0.7, -0.85, -1.6]])
EXP_PENALTY = {"penalty": 2.5, "switch": {"penalty": lambda x: np.exp(-20 * x)}}

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

@pytest.fixture
def pad_setup():
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)
    yield basis, param

def _pad_batch(systems, natm):
    """Pad (numbers, coords, alchemical atom) triples to ``natm`` atoms."""
    numbers, coords, masks = [], [], []
    for n, c, ia in systems:
        npad = natm - len(n)
        numbers.append(numpy.hstack([n, numpy.zeros(npad, dtype=int)]))
        coords.append(numpy.vstack([c, numpy.zeros((npad, 3))]))
        mask = numpy.zeros(natm, dtype=bool)
        mask[ia] = True
        masks.append(mask)
    return np.asarray(numpy.array(numbers)), np.asarray(numpy.array(coords)), np.asarray(numpy.array(masks))

def test_alchem_proton_pad(setup, pad_setup):
    """Batched padded molecules (H3O+, NH4+) with their own proton and lam
    reproduce the unpadded alchemical energies and gradients."""
    basis, param, numbers, coords = setup
    pbasis, pparam = pad_setup
    systems = ((numbers, coords, 3), (NH4_NUMBERS, NH4_COORDS, 4))
    pnumbers, pcoords, masks = _pad_batch(systems, 5)
    lams = np.array([0.3, 0.8])

    for diis in (None, "qbroyden"):
        def energy_pad(numbers, coords, mask, lam):
            mol = MolePad(numbers, coords, basis=pbasis, charge=1)
            mf = alchem_pad.AlchemGFN1XTB(mol, pparam, lam=lam, alchem_mask=mask, **EXP_PENALTY)
            mf.diis = diis
            return mf.kernel()
        e, (g_coords, g_lam) = jax.jit(jax.vmap(
            jax.value_and_grad(energy_pad, argnums=(1, 3))))(pnumbers, pcoords, masks, lams)

        for b, (n, c, ia) in enumerate(systems):
            def energy(c, lam):
                mol = Mole(numbers=n, coords=c, basis=basis, charge=1, verbose=0)
                mf = AlchemGFN1XTB(mol, param, lam=lam, alchem_atoms=(ia,), **EXP_PENALTY)
                mf.diis = diis
                return mf.kernel()
            e0, (gc0, gl0) = jax.jit(jax.value_and_grad(energy, argnums=(0, 1)))(c, lams[b])
            assert abs(e[b] - e0) < 1e-10
            assert abs(g_lam[b] - gl0) < 1e-8
            assert abs(g_coords[b][:len(n)] - gc0).max() < 1e-8
            assert numpy.all(numpy.asarray(g_coords[b][len(n):]) == 0)

@pytest.fixture
def cell_setup(setup):
    """Small cubic cell for H3O+/NH4+ with shared static lattice-sum settings."""
    basis, *_ = setup
    a = numpy.eye(3) * 9.
    rcut = 12.
    cells = [Cell(numbers=n, coords=c, a=a, basis=basis, rcut=rcut, precision=1e-6)
             for n, c in ((numpy.array([8, 1, 1, 1]), numpy.zeros((4, 3))),
                          (NH4_NUMBERS, NH4_COORDS))]
    nimgs = tuple(int(x) for x in numpy.max([numpy.asarray(c.nimgs) for c in cells], axis=0))
    mesh = tuple(int(x) for x in numpy.max(
        [numpy.asarray(c.cutoff_to_mesh(ke_cutoff_ewald(GFN1KXTB.ewald_alpha, 1e-6 * float(c.vol))))
         for c in cells], axis=0))
    kpts = cells[0].make_kpts([2, 1, 1])
    yield a, rcut, nimgs, mesh, kpts

def _kxtb_scf(mf, mesh, diis):
    mf.ewald_mesh = mesh
    mf.diis = diis
    mf.conv_tol = 1e-11
    return mf.kernel()

def test_alchem_proton_kxtb(setup, cell_setup):
    """Periodic H3O+ (2 k-points; charged at lam=1, neutral at lam=0): endpoints
    equal GFN1KXTB of H3O+ and H2O, and the gradients match finite differences."""
    basis, param, numbers, coords = setup
    a, rcut, nimgs, mesh, kpts = cell_setup
    nk = len(kpts)
    cell_kw = {"a": a, "basis": basis, "rcut": rcut, "nimgs": nimgs, "precision": 1e-6}

    for diis in ("anderson", "qbroyden"):
        @jax.jit
        def e_ref():
            cells = (Cell(numbers=numbers, coords=coords, charge=nk, **cell_kw),
                     Cell(numbers=numbers[:3], coords=coords[:3], charge=0, **cell_kw))
            return [_kxtb_scf(GFN1KXTB(c, param=param, kpts=kpts), mesh, diis) for c in cells]
        e_h3o, e_h2o = e_ref()

        def energy(coords, lam):
            # cell.charge is that of the k-point supercell (nk cells)
            cell = Cell(numbers=numbers, coords=coords, charge=nk, **cell_kw)
            mf = AlchemGFN1KXTB(cell, param, lam=lam, alchem_atoms=(3,), kpts=kpts)
            return _kxtb_scf(mf, mesh, diis)
        energy, grad = jax.jit(energy), jax.jit(jax.grad(energy, argnums=(0, 1)))
        assert abs(energy(coords, 0.) - e_h2o) < 1e-8
        assert abs(energy(coords, 1.) - e_h3o) < 1e-8
        _check_grads(energy, grad, coords)

def test_alchem_proton_kxtb_pad(setup, pad_setup, cell_setup):
    """Batched padded cells (H3O+, NH4+) reproduce the unpadded periodic alchemy."""
    basis, param, numbers, coords = setup
    pbasis, pparam = pad_setup
    a, rcut, nimgs, mesh, kpts = cell_setup
    nk = len(kpts)
    systems = ((numbers, coords, 3), (NH4_NUMBERS, NH4_COORDS, 4))
    pnumbers, pcoords, masks = _pad_batch(systems, 5)
    lams = np.array([0.3, 0.8])
    Ls = np.asarray(make_image_grid(numpy.asarray(nimgs)), dtype=np.float64) @ a

    def energy_pad(numbers, coords, mask, lam):
        cell = CellPad(numbers, coords, basis=pbasis, a=a, Ls=Ls, rcut=rcut,
                       precision=1e-6, charge=nk)
        mf = alchem_pad.AlchemGFN1KXTB(cell, pparam, lam=lam, alchem_mask=mask, kpts=kpts,
                                       **EXP_PENALTY)
        return _kxtb_scf(mf, mesh, "anderson")
    e, (g_coords, g_lam) = jax.jit(jax.vmap(
        jax.value_and_grad(energy_pad, argnums=(1, 3))))(pnumbers, pcoords, masks, lams)

    for b, (n, c, ia) in enumerate(systems):
        def energy(c, lam):
            cell = Cell(numbers=n, coords=c, a=a, basis=basis, rcut=rcut, nimgs=nimgs,
                        precision=1e-6, charge=nk)
            mf = AlchemGFN1KXTB(cell, param, lam=lam, alchem_atoms=(ia,), kpts=kpts,
                                **EXP_PENALTY)
            return _kxtb_scf(mf, mesh, "anderson")
        e0, (gc0, gl0) = jax.jit(jax.value_and_grad(energy, argnums=(0, 1)))(c, lams[b])
        assert abs(e[b] - e0) < 1e-10
        assert abs(g_lam[b] - gl0) < 1e-8
        assert abs(g_coords[b][:len(n)] - gc0).max() < 1e-8
        assert numpy.all(numpy.asarray(g_coords[b][len(n):]) == 0)
