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
Alchemical GFN1-XTB with padding (for batched calculations).

See :mod:`pyscfad.xtb.alchem` for the alchemical model.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscfad.xtb.alchem import AlchemMixin
from pyscfad.ml.xtb.xtb_pad import GFN1XTB

if TYPE_CHECKING:
    from typing import Callable
    from pyscfad.typing import ArrayLike
    from pyscfad.ml.gto import MolePad
    from pyscfad.ml.xtb.param import GFN1ParamArray


class AlchemGFN1XTB(AlchemMixin, GFN1XTB):
    """Padded GFN1-XTB with alchemical bare protons.

    The alchemical atoms are given by ``alchem_mask``, a ``(natm,)`` boolean array
    over the padded atoms, so molecules with different alchemical atoms can be
    batched (e.g., with :func:`jax.vmap`) together with ``lam``.
    See :class:`~pyscfad.xtb.alchem.AlchemMixin` for the other arguments.
    """
    def __init__(
        self,
        mol: MolePad,
        param: GFN1ParamArray,
        lam: ArrayLike = 1.,
        alchem_mask: ArrayLike | None = None,
        penalty: float = 1.,
        switch: dict[str, Callable] | None = None,
        alchem_atoms: tuple[int, ...] | None = None,
        **kwargs,
    ):
        mol_param = self._alchem_setup(mol, param, lam=lam, alchem_atoms=alchem_atoms,
                                       alchem_mask=alchem_mask, penalty=penalty, switch=switch)
        super().__init__(mol, param=mol_param, **kwargs)

