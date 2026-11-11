#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Carryover (adstock) and saturation transforms used by :class:`papilon.mmm.MMM`.

All adstock kernels are normalized so their weights sum to one: a constant
spend level x produces a steady-state effective spend of exactly x.
"""

from __future__ import annotations

import numpy as np

__all__ = ["geometric_adstock", "delayed_adstock", "hill"]


def _convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=float), kernel)[: len(x)]


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    """Carryover that peaks in the week of spend and decays by ``alpha`` each week.

    Parameters
    ----------
    x : array of spend per period.
    alpha : retention rate in (0, 1).
    l_max : number of lags in the kernel, including lag 0.
    """
    kernel = alpha ** np.arange(l_max)
    return _convolve(x, kernel / kernel.sum())


def delayed_adstock(x: np.ndarray, alpha: float, peak: float, l_max: int = 8) -> np.ndarray:
    """Carryover whose effect peaks ``peak`` periods after spend.

    The weight at lag l is ``alpha ** ((l - peak) ** 2)``, the delayed adstock of
    Jin et al. (2017), normalized to sum to one.
    """
    lags = np.arange(l_max)
    kernel = alpha ** ((lags - peak) ** 2)
    return _convolve(x, kernel / kernel.sum())


def hill(x: np.ndarray, half_sat: float) -> np.ndarray:
    """Diminishing returns: 0 at zero, 0.5 at ``half_sat``, approaching 1."""
    x = np.asarray(x, dtype=float)
    return x / (x + half_sat)
