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

"""
Alchemical GFN1-XTB for creating/annihilating bare protons.

A scalar ``lam`` interpolates between the system without the alchemical atoms
(``lam=0``) and the system with them (``lam=1``). The alchemical atoms carry no
electrons, i.e., the number of electrons is independent of ``lam``. The molecule
must therefore be built with the charge of the ``lam=1`` end state.

The following quantities are switched by ``lam``:

* ``ovlp``: AO overlap between an alchemical atom and any other atom
  (this also scales the corresponding hcore elements, and QM/MM multipole integrals);
* ``onsite``: on-site hcore block of alchemical atoms, interpolated to ``penalty``;
* ``charge``: reference shell occupations (the "nuclear" part of the Mulliken charges);
* ``rep``: effective nuclear charges in the repulsion energy;
* ``cn``: contribution of alchemical atoms to the coordination numbers.

At ``lam=0`` the alchemical AOs are exactly decoupled and empty (for a large enough
``penalty``), so the energy equals that of the system without the alchemical atoms.

.. warning::
    Preliminary implementation. The curvature of ``dE/dlam`` (and hence the choice of
    ``penalty`` and switching functions) has not been assessed yet, and the padded
    (:mod:`pyscfad.ml.xtb`) and periodic (:mod:`pyscfad.xtb.kxtb`) variants are not supported.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy

from pyscfad import numpy as np
from pyscfad.xtb import util
from pyscfad.xtb.xtb import GFN1XTB
from pyscfad.xtb.param import GFN1MolParam
from pyscfad.xtb.data.elements import N_VALENCE

if TYPE_CHECKING:
    from typing import Callable
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.gto import MoleLite
    from pyscfad.xtb.param import GFN1Param

SWITCH_KEYS = ("ovlp", "onsite", "charge", "rep", "cn")

class AlchemGFN1XTB(GFN1XTB):
    """GFN1-XTB with alchemical bare protons.

    Args:
        mol: Molecule containing the alchemical atoms, with the charge of the ``lam=1`` state.
        param: GFN1 parameters (:class:`~pyscfad.xtb.param.GFN1Param`).
        lam: Alchemical coupling parameter in [0, 1]. Can be traced.
        alchem_atoms: Indices of the alchemical atoms.
        penalty: On-site orbital energy (Eh) of the alchemical AOs at ``lam=0``.
            It only needs to keep those orbitals above the Fermi level.
        switch: Optional mapping from the keys ``ovlp``, ``onsite``, ``charge``,
            ``rep`` and ``cn`` to switching functions ``s(lam)``. Defaults to linear.

    Notes:
        All ``lam``-dependent quantities are built at construction;
        create a new object for each ``lam``.
    """
    def __init__(
        self,
        mol: MoleLite,
        param: GFN1Param,
        lam: ArrayLike = 1.,
        alchem_atoms: tuple[int, ...] = (),
        penalty: float = 1.,
        switch: dict[str, Callable] | None = None,
        **kwargs,
    ):
        if not hasattr(param, "to_mol_param"):
            raise TypeError("AlchemGFN1XTB requires a GFN1Param object.")
        switch = {} if switch is None else dict(switch)
        unknown = set(switch) - set(SWITCH_KEYS)
        if unknown:
            raise KeyError(f"Unknown switch keys {unknown}; allowed {SWITCH_KEYS}")

        self.lam = lam
        self.alchem_atoms = tuple(int(i) for i in alchem_atoms)
        self.penalty = penalty
        s = {k: switch.get(k, lambda x: x)(lam) for k in SWITCH_KEYS}
        self._switch_values = s

        alch_atm = numpy.zeros(mol.natm, dtype=bool)
        alch_atm[list(self.alchem_atoms)] = True
        alch_bas = alch_atm[util.atom_to_bas_indices(mol)]
        atm_to_ao_id = util.atom_to_ao_indices(mol)
        alch_ao = alch_atm[atm_to_ao_id]
        same_atom = atm_to_ao_id[:,None] == atm_to_ao_id[None,:]
        self._alch_atm = alch_atm

        # AO-pair factors for overlap-like quantities
        w_ao = np.where(alch_ao, s["ovlp"], 1.)
        self._ovlp_fac = np.where(same_atom, 1., w_ao[:,None] * w_ao[None,:])
        # on-site blocks of alchemical atoms
        self._onsite_mask = same_atom & alch_ao[:,None]

        mol_param = GFN1MolParam(mol, param,
                                 cn_weights=np.where(alch_atm, s["cn"], 1.))
        self._refocc_alch = np.sum(np.where(alch_bas, mol_param.refocc, 0.))
        mol_param.refocc = np.where(alch_bas, s["charge"] * mol_param.refocc, mol_param.refocc)
        mol_param.zeff = np.where(alch_atm, s["rep"] * mol_param.zeff, mol_param.zeff)

        super().__init__(mol, param=mol_param, **kwargs)

    @property
    def tot_charge(self) -> Array:
        # sum(refocc(lam)) - nelectron
        return self.mol.charge - (1 - self._switch_values["charge"]) * self._refocc_alch

    def mask_ao_pairs(self, a: Array) -> Array:
        return a * self._ovlp_fac

    def get_ovlp(self, mol: MoleLite | None = None) -> Array:
        return self.mask_ao_pairs(super().get_ovlp(mol))

    def _get_EHT_factor(self, mol: MoleLite | None = None) -> Array:
        h1 = super()._get_EHT_factor(mol)
        s = self._switch_values["onsite"]
        return np.where(self._onsite_mask, s * h1 + (1 - s) * self.penalty, h1)

    def dip_moment(
        self,
        mol: MoleLite | None = None,
        dm: ArrayLike | None = None,
        unit: str = "Debye",
        origin: ArrayLike | None = None,
        verbose: int | None = None,
        charges: ArrayLike | None = None,
    ) -> Array:
        if mol is None:
            mol = self.mol
        if charges is None:
            charges = np.asarray([N_VALENCE.get(elem) for elem in mol.elements],
                                 dtype=np.floatx)
            charges = np.where(self._alch_atm, self._switch_values["charge"] * charges, charges)
        return super().dip_moment(mol=mol, dm=dm, unit=unit, origin=origin, verbose=verbose,
                                  charges=charges)
