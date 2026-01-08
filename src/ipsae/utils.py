"""Utility functions for ipSAE."""

from collections.abc import Iterable
from typing import Any, overload

import numpy as np


def ptm_func(x: float, d0: float) -> float:
    """Calculate the TM-score term: 1 / (1 + (x/d0)^2).

    Args:
        x: Distance or error value.
        d0: Normalization factor.

    Returns:
    -------
        The calculated term.

    """
    return 1.0 / (1 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)


def calc_d0(length: int | float, pair_type: str) -> float:
    """Calculate the normalization factor d0 for TM-score.

    Formula from Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702-710 (2004).

    d0 = 1.24 * (L - 15)^(1/3) - 1.8
    Minimum value is 1.0 (or 2.0 for nucleic acids).

    Args:
        length: Length (number of residues).
        pair_type: Type of chain pair ('protein' or 'nucleic_acid').

    Returns:
    -------
        The calculated d0 value.

    """
    length = max(length, 27)
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def calc_d0_array(length: list[float] | np.ndarray, pair_type: str) -> np.ndarray:
    """Vectorized version of calc_d0.

    Args:
        length: Array of lengths.
        pair_type: Type of chain pair ('protein' or 'nucleic_acid').

    Returns:
    -------
        Array of calculated d0 values.

    """
    length_arr = np.array(length, dtype=float)
    length_arr = np.maximum(27, length_arr)
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    return np.maximum(min_value, 1.24 * (length_arr - 15) ** (1.0 / 3.0) - 1.8)


@overload
def init_chainpairdict_zeros(
    chainlist: list[str] | np.ndarray, zero: int
) -> dict[str, dict[str, int]]: ...


@overload
def init_chainpairdict_zeros(
    chainlist: list[str] | np.ndarray, zero: float
) -> dict[str, dict[str, float]]: ...


@overload
def init_chainpairdict_zeros(
    chainlist: list[str] | np.ndarray, zero: str
) -> dict[str, dict[str, str]]: ...


def init_chainpairdict_zeros(
    chainlist: Iterable[Any],
    zero: Any = 0,
) -> dict[Any, dict[Any, Any]]:
    """Initialize a nested dictionary for chain pairs with zero values."""
    return {c1: {c2: zero for c2 in chainlist if c1 != c2} for c1 in chainlist}


def contiguous_ranges(numbers: set[int]) -> str | None:
    """Format a set of numbers into a string of contiguous ranges.

    This is for printing out residue ranges in PyMOL scripts.

    Example: {1, 2, 3, 5, 7, 8} -> "1-3+5+7-8"

    Args:
        numbers: A set of integers.

    Returns:
    -------
        A formatted string representing the ranges, or None if empty.

    """
    if not numbers:
        return None
    sorted_numbers: list[int] = sorted(numbers)
    start = sorted_numbers[0]
    end = start
    ranges: list[str] = []

    def format_range(s, e) -> str:
        return f"{s}" if s == e else f"{s}-{e}"

    for number in sorted_numbers[1:]:
        if number == end + 1:
            end = number
        else:
            ranges.append(format_range(start, end))
            start = end = number
    ranges.append(format_range(start, end))
    return "+".join(ranges)
