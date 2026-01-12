"""Text output for ipSAE results."""

from pathlib import Path

from ipsae.models import (
    ChainPairScoreResults,
    LigandScoreResults,
    PerResScoreResults,
    ScoreResults,
)


def write_text_outputs(results: ScoreResults, output_prefix: str | Path) -> None:
    """Write the calculated results to text files.

    Creates two files:
    - {output_prefix}.txt: Summary table of scores.
    - {output_prefix}_byres.txt: Detailed per-residue scores.

    Args:
        results: The ScoreResults object containing the data to write.
        output_prefix: The prefix for the output filenames (including path).

    """
    # Overwrite files by default
    chain_pair_scores_file = Path(f"{output_prefix}.txt")
    with chain_pair_scores_file.open("w") as f:
        f.write("\n" + ChainPairScoreResults.header_line())
        for i, summary in enumerate(results.chain_pair_scores):
            line_str = summary.to_formatted_line()
            f.write(f"{line_str}\n")
            # Add blank line after max rows to separate chain pair groups, but not at the end
            if summary.Type == "max" and i < len(results.chain_pair_scores) - 1:
                f.write("\n")

    # For per-residue scores, overwrite each time
    with Path(f"{output_prefix}_byres.txt").open("w") as f:
        f.write(PerResScoreResults.header_line())
        f.writelines(
            res_line.to_formatted_line(end="\n") for res_line in results.by_res_scores
        )

    # Write ligand scores if they exist
    if results.ligand_scores:
        ligand_scores_file = Path(f"{output_prefix}_ligands.txt")
        with ligand_scores_file.open("w") as f:
            f.write(LigandScoreResults.header_line())
            for ligand_res in results.ligand_scores:
                f.write(ligand_res.to_formatted_line(end="\n"))
