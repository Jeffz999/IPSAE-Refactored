"""Text output for ipSAE results."""

from pathlib import Path

from ipsae.models import ChainPairScoreResults, PerResScoreResults, ScoreResults


def write_text_outputs(results: ScoreResults, output_prefix: str | Path) -> None:
    """Write the calculated results to text files.

    Creates two files:
    - {output_prefix}.txt: Summary table of scores.
    - {output_prefix}_byres.txt: Detailed per-residue scores.

    Args:
        results: The ScoreResults object containing the data to write.
        output_prefix: The prefix for the output filenames (including path).

    """
    # Append to file if it exists, since we may be processing multiple models
    # or comparing different input parameters
    chain_pair_scores_file = Path(f"{output_prefix}.txt")
    if chain_pair_scores_file.exists():
        existing_chain_pair_lines = set(
            chain_pair_scores_file.read_text().strip().splitlines()
        )
    else:
        existing_chain_pair_lines = set()
        chain_pair_scores_file.write_text("\n" + ChainPairScoreResults.header_line())

    with chain_pair_scores_file.open("a") as f:
        for summary in results.chain_pair_scores:
            line_str = summary.to_formatted_line()
            if line_str not in existing_chain_pair_lines:
                f.write(f"{line_str}\n")
                # Add blank line after max rows to separate chain pair groups
                if summary.Type == "max":
                    f.write("\n")

    # For per-residue scores, overwrite each time
    with Path(f"{output_prefix}_byres.txt").open("w") as f:
        f.write(PerResScoreResults.header_line())
        f.writelines(
            res_line.to_formatted_line(end="\n") for res_line in results.by_res_scores
        )
