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
"""Deprecation helpers. See docs/api-stability.md for the policy they implement."""
from __future__ import annotations

import functools
import warnings

__all__ = ["deprecated", "warn_deprecated"]


def warn_deprecated(what: str, since: str, removed_in: str, alternative: str | None = None, stacklevel: int = 3) -> None:
    msg = f"{what} is deprecated since Papilon {since} and will be removed in {removed_in}."
    if alternative:
        msg += f" Use {alternative} instead."
    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)


def deprecated(since: str, removed_in: str, alternative: str | None = None):
    """Mark a function as deprecated: it keeps working and warns once per call site."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            warn_deprecated(f"{fn.__module__}.{fn.__qualname__}", since, removed_in, alternative)
            return fn(*args, **kwargs)
        inner.__deprecated__ = {"since": since, "removed_in": removed_in, "alternative": alternative}
        return inner
    return wrap
