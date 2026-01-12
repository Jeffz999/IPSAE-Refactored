"""CSV output for ipSAE results."""

import csv
from dataclasses import asdict
from pathlib import Path

from ipsae.models import ScoreResults


def write_csv_outputs(results: ScoreResults, output_prefix: str | Path) -> None:
    """Write the calculated summary results to a CSV file.

    Creates:
    - {output_prefix}.csv: Summary table of scores in CSV format.

    Args:
        results: The ScoreResults object containing the data to write.
        output_prefix: The prefix for the output filenames (including path).

    """
    csv_file = Path(f"{output_prefix}.csv")

    if not results.chain_pair_scores:
        return

    # Get field names from the first result
    fieldnames = list(asdict(results.chain_pair_scores[0]).keys())

    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for summary in results.chain_pair_scores:
            writer.writerow(asdict(summary))

    # Write ligand scores if they exist
    if results.ligand_scores:
        ligand_csv_file = Path(f"{output_prefix}_ligands.csv")
        fieldnames_lig = list(asdict(results.ligand_scores[0]).keys())

        with ligand_csv_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_lig)
            writer.writeheader()
            for ligand_res in results.ligand_scores:
                writer.writerow(asdict(ligand_res))
