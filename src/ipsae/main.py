"""Main entry point for ipSAE."""

import logging
import os
from pathlib import Path

from ipsae.cli import parse_cli_args
from ipsae.io.csv import write_csv_outputs
from ipsae.io.pymol import write_pymol_outputs
from ipsae.io.text import write_text_outputs
from ipsae.models import (
    ChainPairScoreResults,
    InputModelType,
    PerResScoreResults,
    ScoreResults,
)
from ipsae.parsers.chains import parse_chain_groups
from ipsae.parsers.pae import load_pae_data
from ipsae.parsers.structure import load_structure
from ipsae.scoring import calculate_scores

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger("ipSAE")


def ipsae(
    pae_file: Path,
    structure_file: Path,
    pae_cutoff: float,
    dist_cutoff: float,
    model_type: str | None = None,
    chain_groups: str | None = None,
    ligand_pae_cutoff: float = 3.0,
    ligand_dist_cutoff: float = 4.0,
) -> ScoreResults:
    """Calculate ipSAE, pDockQ, pDockQ2, and LIS scores for protein structure models.

    Args:
        pae_file: Path to the PAE file (json, npz, pkl).
        structure_file: Path to the structure file (pdb, cif).
        pae_cutoff: Cutoff for PAE to consider a residue pair "good".
        dist_cutoff: Distance cutoff for contact definition.
        model_type: Type of the model. If None, auto-detects based on file extensions.
        chain_groups: Optional string to parse chain groups from. If provided,
            this takes precedence over chain_groups. Use "..." to include all
            default individual chain permutations. Default is None, which behaves
            the same as passing "...".

    Returns:
        A ScoreResults object containing all calculated scores and output strings.
        The main attributes are chain_pair_scores, by_res_scores, and pymol_script.

    """
    # Validate model type
    if model_type is None:
        from ipsae.cli import guess_model_type

        model_type_enum = guess_model_type(pae_file, structure_file)
    else:
        model_type_enum = InputModelType.from_string(model_type)

    # Load data
    structure_data = load_structure(structure_file)
    pae_data = load_pae_data(pae_file, structure_data, model_type_enum)

    # Parse chain groups if string is provided
    if chain_groups is None:
        chain_groups = "..."
    parsed_chain_groups = parse_chain_groups(chain_groups, structure_data.unique_chains)
    logger.debug(f"Parsed chain groups: {parsed_chain_groups}")

    # Calculate scores
    pdb_stem = structure_file.stem
    results = calculate_scores(
        structure=structure_data,
        pae_data=pae_data,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        ligand_pae_cutoff=ligand_pae_cutoff,
        ligand_dist_cutoff=ligand_dist_cutoff,
        label=pdb_stem,
        chain_groups=parsed_chain_groups,
    )
    return results


def main():
    """Entry point for the script.

    Parses command line arguments, loads data, calculates scores, and writes outputs.
    """
    args = parse_cli_args()
    logger.debug(f"Parsed CLI args: {args}")
    logger.info(f"Detected model type: {args.model_type}")
    if args.chain_groups:
        logger.info(f"Chain groups: {args.chain_groups}")

    scores = ipsae(
        pae_file=args.pae_file,
        structure_file=args.structure_file,
        pae_cutoff=args.pae_cutoff,
        dist_cutoff=args.dist_cutoff,
        model_type=args.model_type.value,
        chain_groups=args.chain_groups,
        ligand_pae_cutoff=args.ligand_pae_cutoff,
        ligand_dist_cutoff=args.ligand_dist_cutoff,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        pae_str = str(int(args.pae_cutoff)).zfill(2)
        dist_str = str(int(args.dist_cutoff)).zfill(2)
        pdb_stem = args.structure_file.stem
        output_prefix = args.output_dir / f"{pdb_stem}_{pae_str}_{dist_str}"

        write_text_outputs(scores, output_prefix)
        write_pymol_outputs(scores, output_prefix)
        if args.csv:
            write_csv_outputs(scores, output_prefix)

        logger.info(
            f"Success! Outputs written to {output_prefix}{{.txt,_byres.txt,.pml{',.csv' if args.csv else ''}}}"
        )
    else:
        # Print summary to stdout
        print("#" * 90 + "\n# Per-residue scores\n" + "#" * 90)
        print(PerResScoreResults.header_line())
        print("\n".join(x.to_formatted_line() for x in scores.by_res_scores))

        print("\n\n" + "#" * 90 + "\n# Summary\n" + "#" * 90)
        print("\n" + ChainPairScoreResults.header_line(), end="")
        for summary in scores.chain_pair_scores:
            line_end = "\n" if summary.Type == "max" else ""
            print(summary.to_formatted_line(end="\n"), end=line_end)

        print("\n\n" + "#" * 90 + "\n# PyMOL script\n" + "#" * 90)
        print("".join(scores.pymol_script))


if __name__ == "__main__":
    main()
