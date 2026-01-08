"""Chain group parsing and index management."""

import numpy as np


def chain_group_name(chain_group: list[str]) -> str:
    """Generate a display name for a chain group.

    Args:
        chain_group: List of chain identifiers.

    Returns:
        String representation of the chain group (e.g., "H+L" for ["H", "L"]).

    """
    return "+".join(chain_group)


def parse_chain_groups(
    chain_groups_str: str,
    unique_chains: np.ndarray | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Parse chain groups string into a list of chain group pairs.

    Format: "A/H+L,A/H,A/L" means:
        - Calculate scores between chain A and chains H+L (treated as one group)
        - Calculate scores between chain A and chain H
        - Calculate scores between chain A and chain L

    Use "..." to include all default individual chain permutations.
    For example, "A/H+L,..." with chains A, H, L will include A/H+L plus all
    individual chain pairs (A/H, A/L, H/A, H/L, L/A, L/H).

    Args:
        chain_groups_str: Comma-separated chain group pairs in format "group1/group2".
            Each group can contain multiple chains joined with "+", e.g., "H+L".
            Use "..." to include all default individual chain permutations.
        unique_chains: Array of unique chain identifiers, required when using "...".

    Returns:
        List of tuples, where each tuple contains two lists of chain identifiers.
        For example, "A/H+L,A/H" returns [(['A'], ['H', 'L']), (['A'], ['H'])].
        Duplicates are automatically removed.

    """
    result = []
    seen: set[str] = set()  # Track seen pairs to remove duplicates
    has_ellipsis = False

    pairs = chain_groups_str.split(",")
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        # Check for ellipsis token
        if pair == "...":
            has_ellipsis = True
            continue

        if "/" not in pair:
            raise ValueError(
                f"Invalid chain group pair format: '{pair}'. Expected 'group1/group2'."
            )
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid chain group pair format: '{pair}'. Expected exactly one '/' separator."
            )
        group1_str, group2_str = parts
        group1 = sorted(g for c in group1_str.split("+") if (g := c.strip()))
        group2 = sorted(g for c in group2_str.split("+") if (g := c.strip()))
        if not group1 or not group2:
            raise ValueError(
                f"Invalid chain group pair: '{pair}'. Both groups must contain at least one chain."
            )
        if set(group1) & set(group2):
            raise ValueError(
                f"Invalid chain group pair: '{pair}'. Groups must not share chains."
            )

        # Add to result if not a duplicate (check both directions)
        pair_key_forward = f"{chain_group_name(group1)}/{chain_group_name(group2)}"
        pair_key_reverse = f"{chain_group_name(group2)}/{chain_group_name(group1)}"
        if pair_key_forward not in seen:
            seen.add(pair_key_forward)
            seen.add(pair_key_reverse)  # Mark reverse as seen too
            result.append((group1, group2))
            result.append((group2, group1))

    # Handle ellipsis - add all individual chain permutations
    # Use nested loops to match original behavior ordering:
    # For chains [A,B,C]: A->B, A->C, B->A, B->C, C->A, C->B
    if has_ellipsis:
        if unique_chains is None:
            raise ValueError(
                "Cannot use '...' without providing unique_chains. "
                "The '...' token requires knowledge of available chains."
            )

        for c1 in unique_chains:
            for c2 in unique_chains:
                if c1 == c2:
                    continue
                pair_key = f"{c1}/{c2}"
                if pair_key not in seen:
                    seen.add(pair_key)
                    result.append(([c1], [c2]))

    return result


def get_chain_group_indices(chains: np.ndarray, chain_group: list[str]) -> np.ndarray:
    """Get indices of residues belonging to any chain in the given chain group.

    Args:
        chains: Array of chain identifiers for each residue.
        chain_group: List of chain identifiers forming the group.

    Returns:
        Array of indices for residues in the chain group.

    """
    mask = np.isin(chains, chain_group)
    return np.where(mask)[0]
