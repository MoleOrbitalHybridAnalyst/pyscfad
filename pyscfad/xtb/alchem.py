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
* ``onsite`` (g) and ``penalty`` (p): on-site hcore block of alchemical atoms,
  ``g(lam) * K + p(lam) * penalty`` with ``K`` the physical on-site factor
  (``p`` defaults to ``1 - g``, i.e., interpolation between ``penalty`` and ``K``);
* ``charge``: reference shell occupations (the "nuclear" part of the Mulliken charges);
* ``rep``: effective nuclear charges in the repulsion energy;
* ``cn``: contribution of alchemical atoms to the coordination numbers.

At ``lam=0`` the alchemical AOs are exactly decoupled and empty (for a large enough
``penalty``), so the energy equals that of the system without the alchemical atoms.

The switches are implemented once in :class:`AlchemMixin` and combined with the
XTB classes:

* :class:`AlchemGFN1XTB`: molecules.

.. warning::
    Preliminary implementation. The defaults (linear switches, ``penalty=1``) are kept
    for backward compatibility; a penalty that decays quickly, e.g., ``penalty=2.5`` with
    ``switch={"penalty": lambda x: np.exp(-20 * x)}``, gives a safer ``lam=0`` limit at a
    similar curvature of ``dE/dlam``.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import copy

import numpy

from pyscfad import numpy as np
from pyscfad.xtb import util
from pyscfad.xtb.xtb import GFN1XTB
from pyscfad.xtb.param import cn_d3
from pyscfad.xtb.data.elements import N_VALENCE_ARRAY

if TYPE_CHECKING:
    from typing import Any, Callable
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.gto import MoleLite

SWITCH_KEYS = ("ovlp", "onsite", "penalty", "charge", "rep", "cn")

def _linear(x):
    return x

def switch_values(lam: ArrayLike, switch: dict[str, Callable] | None = None) -> dict[str, Array]:
    """Evaluate the switching functions at ``lam`` (see :class:`AlchemMixin`)."""
    switch = {} if switch is None else dict(switch)
    unknown = set(switch) - set(SWITCH_KEYS)
    if unknown:
        raise KeyError(f"Unknown switch keys {unknown}; allowed {SWITCH_KEYS}")
    onsite = switch.get("onsite", _linear)
    defaults = {"penalty": lambda x: 1 - onsite(x)}
    return {k: switch.get(k, defaults.get(k, _linear))(lam) for k in SWITCH_KEYS}

def make_alchem_mask(
    natm: int,
    alchem_atoms: tuple[int, ...] | None = None,
    alchem_mask: ArrayLike | None = None,
) -> Array:
    """Per-atom boolean mask of the alchemical atoms.

    Either ``alchem_atoms`` (static indices) or ``alchem_mask`` (``(natm,)`` booleans,
    can be traced, e.g., for batched padded molecules) must be given.
    """
    if alchem_mask is not None:
        if alchem_atoms:
            raise ValueError("Give either alchem_atoms or alchem_mask, not both.")
        return np.asarray(alchem_mask, dtype=bool)
    mask = numpy.zeros(natm, dtype=bool)
    mask[list(alchem_atoms or ())] = True
    return mask


class AlchemMixin:
    """Alchemical bare atoms for XTB methods.

    Mixed in before an XTB class (e.g., ``class A(AlchemMixin, GFN1XTB)``).
    Concrete classes call :meth:`_alchem_setup` in ``__init__`` and pass the
    returned parameters to the XTB constructor.

    The mixin overrides the hooks used by the molecular (and padded) XTB code:
    :meth:`get_ovlp`, :meth:`_get_EHT_factor`, :meth:`mask_ao_pairs`,
    :attr:`tot_charge` and :meth:`dip_moment`. Classes whose Hamiltonian is built
    differently (e.g., with k-points) override those parts using
    :attr:`_ovlp_fac`, :attr:`_onsite_mask` and :attr:`_switch_values`.

    Args (of :meth:`_alchem_setup`):
        mol: Molecule containing the alchemical atoms, with the charge of the ``lam=1`` state.
        param: GFN1 parameters (anything with ``to_mol_param``).
        lam: Alchemical coupling parameter in [0, 1]. Can be traced.
        alchem_atoms: Indices of the alchemical atoms.
        alchem_mask: Alternatively, a ``(natm,)`` boolean mask of the alchemical atoms.
            Can be traced.
        penalty: On-site orbital energy (Eh) of the alchemical AOs at ``lam=0``.
            It only needs to keep those orbitals above the Fermi level.
        switch: Optional mapping from the keys ``ovlp``, ``onsite``, ``penalty``,
            ``charge``, ``rep`` and ``cn`` to switching functions ``s(lam)``.
            All default to linear, except ``penalty``, which defaults to
            ``1 - onsite(lam)``. The on-site block of the alchemical atoms is
            ``onsite(lam) * K + penalty(lam) * penalty``; for exact decoupling at
            ``lam=0`` and the physical block at ``lam=1``, use ``onsite(0) = 0``,
            ``onsite(1) = 1``, ``penalty(0) = 1`` and ``penalty(1) = 0``.
    """
    def _alchem_setup(
        self,
        mol: MoleLite,
        param: Any,
        lam: ArrayLike = 1.,
        alchem_atoms: tuple[int, ...] | None = None,
        alchem_mask: ArrayLike | None = None,
        penalty: float = 1.,
        switch: dict[str, Callable] | None = None,
    ) -> Any:
        """Set up the alchemical switches; returns the switched molecular parameters."""
        if not hasattr(param, "to_mol_param"):
            raise TypeError("Alchemical XTB requires element parameters with to_mol_param, "
                            "e.g., GFN1Param or GFN1ParamArray.")
        self.lam = lam
        self.alchem_atoms = None if alchem_atoms is None else tuple(int(i) for i in alchem_atoms)
        self.penalty = penalty
        s = self._switch_values = switch_values(lam, switch)

        alch_atm = make_alchem_mask(mol.natm, alchem_atoms, alchem_mask)
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

        mol_param = copy.copy(param.to_mol_param(mol))
        self._refocc_alch = np.sum(np.where(alch_bas, mol_param.refocc, 0.))
        mol_param.refocc = np.where(alch_bas, s["charge"] * mol_param.refocc, mol_param.refocc)
        mol_param.zeff = np.where(alch_atm, s["rep"] * mol_param.zeff, mol_param.zeff)
        mol_param.CN = cn_d3(mol, kcn=param.kcn_d3, weights=np.where(alch_atm, s["cn"], 1.))
        return mol_param

    @property
    def tot_charge(self) -> Array:
        # sum(refocc(lam)) - nelectron
        return super().tot_charge - (1 - self._switch_values["charge"]) * self._refocc_alch

    def mask_ao_pairs(self, a: Array) -> Array:
        return a * self._ovlp_fac

    def get_ovlp(self, mol: MoleLite | None = None) -> Array:
        return self.mask_ao_pairs(super().get_ovlp(mol))

    def _get_EHT_factor(self, mol: MoleLite | None = None) -> Array:
        h1 = super()._get_EHT_factor(mol)
        g = self._switch_values["onsite"]
        p = self._switch_values["penalty"]
        return np.where(self._onsite_mask, g * h1 + p * self.penalty, h1)

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
            charges = N_VALENCE_ARRAY[np.asarray(mol.atom_charges())].astype(np.floatx)
            charges = np.where(self._alch_atm, self._switch_values["charge"] * charges, charges)
        return super().dip_moment(mol=mol, dm=dm, unit=unit, origin=origin, verbose=verbose,
                                  charges=charges)


class AlchemGFN1XTB(AlchemMixin, GFN1XTB):
    """GFN1-XTB with alchemical bare protons.

    See :class:`AlchemMixin` for the arguments.

    Notes:
        All ``lam``-dependent quantities are built at construction;
        create a new object for each ``lam``.
    """
    def __init__(
        self,
        mol: MoleLite,
        param: Any,
        lam: ArrayLike = 1.,
        alchem_atoms: tuple[int, ...] | None = None,
        penalty: float = 1.,
        switch: dict[str, Callable] | None = None,
        alchem_mask: ArrayLike | None = None,
        **kwargs,
    ):
        mol_param = self._alchem_setup(mol, param, lam=lam, alchem_atoms=alchem_atoms,
                                       alchem_mask=alchem_mask, penalty=penalty, switch=switch)
        super().__init__(mol, param=mol_param, **kwargs)

