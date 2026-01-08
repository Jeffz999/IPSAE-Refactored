"""Tests for CSV output generation."""

import csv
from pathlib import Path

from ipsae.io.csv import write_csv_outputs
from ipsae.models import ChainPairScoreResults, ScoreResults


def test_write_csv_outputs(tmp_path):
    """Test that write_csv_outputs correctly generates a CSV file."""
    # Create dummy results
    summary = ChainPairScoreResults(
        Chn1="A",
        Chn2="B",
        PAE=10.0,
        Dist=10.0,
        Type="asym",
        ipSAE=0.5,
        ipSAE_d0chn=0.4,
        ipSAE_d0dom=0.3,
        ipTM_af=0.6,
        ipTM_d0chn=0.5,
        pDockQ=0.7,
        pDockQ2=0.8,
        LIS=0.9,
        n0res=10,
        n0chn=20,
        n0dom=15,
        d0res=5.0,
        d0chn=6.0,
        d0dom=5.5,
        nres1=5,
        nres2=5,
        dist1=3,
        dist2=3,
        Model="test_model",
    )
    results = ScoreResults(
        ipsae_scores={},
        iptm_scores={},
        pdockq_scores={},
        pdockq2_scores={},
        lis_scores={},
        metrics={},
        by_res_scores=[],
        chain_pair_scores=[summary],
        pymol_script=[],
    )

    output_prefix = tmp_path / "test_output"
    write_csv_outputs(results, output_prefix)

    csv_file = Path(f"{output_prefix}.csv")
    assert csv_file.exists()

    with csv_file.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Chn1"] == "A"
        assert rows[0]["Chn2"] == "B"
        assert float(rows[0]["ipSAE"]) == 0.5
        assert rows[0]["Model"] == "test_model"
