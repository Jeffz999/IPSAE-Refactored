"""PyMOL script output for ipSAE results."""

from pathlib import Path

from ipsae.models import ScoreResults


def write_pymol_outputs(results: ScoreResults, output_prefix: str | Path) -> None:
    """Write the calculated results to a PyMOL script file.

    Creates:
    - {output_prefix}.pml: PyMOL script for visualization.

    Args:
        results: The ScoreResults object containing the data to write.
        output_prefix: The prefix for the output filenames (including path).

    """
    with Path(f"{output_prefix}.pml").open("w") as f:
        f.writelines(results.pymol_script)
