"""Command line interface for ipSAE."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from ipsae.models import InputModelType


@dataclass
class CliArgs:
    """Parsed command line arguments."""

    pae_file: Path
    structure_file: Path
    pae_cutoff: float
    dist_cutoff: float
    model_type: InputModelType
    output_dir: Path | None
    chain_groups: str | None
    csv: bool


def guess_model_type(pae_file: Path, structure_file: Path) -> InputModelType:
    """Guess the model type based on file extensions.

    Args:
        pae_file: Path to the PAE file.
        structure_file: Path to the structure file.

    Returns:
        The guessed model type as an InputModelType enum.

    Raises:
        ValueError: If the model type cannot be determined.

    """
    if structure_file.suffix == ".pdb":
        return InputModelType.AF2
    elif structure_file.suffix == ".cif":
        if pae_file.suffix == ".json":
            return InputModelType.AF3
        elif pae_file.suffix == ".npz":
            return InputModelType.Boltz1  # boltz2 is the same
        elif pae_file.suffix == ".npy":
            return InputModelType.Chai1

    raise ValueError(
        f"Could not determine model type from inputs: {pae_file}, {structure_file}"
    )


def parse_cli_args() -> CliArgs:
    """Parse command line arguments.

    Returns:
        A CliArgs object with the parsed arguments.

    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Calculate ipSAE, pDockQ, pDockQ2, and LIS scores for protein structure models."
    )
    parser.add_argument("pae_file", help="Path to PAE file (json, npz, pkl)")
    parser.add_argument("structure_file", help="Path to structure file (pdb, cif)")
    parser.add_argument("pae_cutoff", type=float, help="PAE cutoff")
    parser.add_argument("dist_cutoff", type=float, help="Distance cutoff")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save outputs. Prints results to stdout if not passed.",
    )
    parser.add_argument(
        "-t",
        "--model-type",
        help="Model type: af2, af3, boltz1, boltz2, chai-1 (auto-detected if not provided).",
        default="unknown",
    )
    parser.add_argument(
        "-g",
        "--chain-groups",
        help=(
            "Calculate scores between specified chain groups instead of all pairs. "
            "Format: 'group1/group2,group1/group3,...' where groups can contain "
            "multiple chains joined with '+'. Use '...' to include all default "
            "individual chain permutations. Example: 'A/H+L,...' calculates "
            "scores between chain A and chains H+L, plus all individual chain pairs."
        ),
        default=None,
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output summary results in CSV format.",
    )

    input_args = parser.parse_args()

    # Normalize paths and prepare typed args
    pae_path = Path(input_args.pae_file).expanduser().resolve()
    if not pae_path.exists():
        raise FileNotFoundError(f"PAE file not found: {pae_path}")
    struct_path = Path(input_args.structure_file).expanduser().resolve()
    if not struct_path.exists():
        raise FileNotFoundError(f"Structure file not found: {struct_path}")
    out_dir = (
        Path(input_args.output_dir).expanduser().resolve()
        if input_args.output_dir is not None
        else None
    )

    # Guess model type from file extensions
    if input_args.model_type != "unknown":
        model_type = InputModelType.from_string(input_args.model_type)
    else:
        model_type = guess_model_type(pae_path, struct_path)

    return CliArgs(
        pae_file=pae_path,
        structure_file=struct_path,
        pae_cutoff=input_args.pae_cutoff,
        dist_cutoff=input_args.dist_cutoff,
        model_type=model_type,
        output_dir=out_dir,
        chain_groups=input_args.chain_groups,
        csv=input_args.csv,
    )
