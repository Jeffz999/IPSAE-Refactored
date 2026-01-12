import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys
from ipsae.models import StructureData, PAEData, LigandAtom, Residue
from ipsae.scoring import calculate_ligand_scores

# Path to ligand test data
LIGAND_DATA_DIR = Path(__file__).parent / "data" / "ligand_test"

def test_calculate_ligand_scores_logic():
    """Unit test for ligand scoring logic using synthetic data."""
    # Create synthetic protein residues (2 residues)
    # Residue 1 at (0,0,0), Residue 2 at (10,10,10)
    res1 = Residue(atom_num=1, coor=np.array([0.0, 0.0, 0.0]), res="ALA", chainid="A", resnum=1, residue_str="A_ALA_1")
    res2 = Residue(atom_num=2, coor=np.array([10.0, 10.0, 10.0]), res="ALA", chainid="A", resnum=2, residue_str="A_ALA_2")
    
    # Create synthetic ligand atoms (2 atoms)
    # Atom 1 at (2,0,0) - close to Residue 1
    # Atom 2 at (12,10,10) - close to Residue 2
    lig1 = LigandAtom(atom_num=3, coor=np.array([2.0, 0.0, 0.0]), res="LIG", chainid="L", resnum=1, atom_name="C1", element="C", global_pae_idx=2)
    lig2 = LigandAtom(atom_num=4, coor=np.array([12.0, 10.0, 10.0]), res="LIG", chainid="L", resnum=1, atom_name="C2", element="C", global_pae_idx=3)
    
    # Protein atoms for contact calculation
    prot_atom1 = LigandAtom(atom_num=1, coor=np.array([0.0, 0.0, 0.0]), res="ALA", chainid="A", resnum=1, atom_name="CA", element="C", global_pae_idx=0)
    prot_atom2 = LigandAtom(atom_num=2, coor=np.array([10.0, 10.0, 10.0]), res="ALA", chainid="A", resnum=2, atom_name="CA", element="C", global_pae_idx=1)

    structure = StructureData(
        residues=[res1, res2],
        cb_residues=[res1, res2],
        chains=np.array(["A", "A"]),
        unique_chains=np.array(["A", "L"]),
        token_mask=np.array([True, True, False, False]), # 2 protein + 2 ligand tokens
        residue_types=np.array(["ALA", "ALA"]),
        coordinates=np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]),
        distances=np.zeros((2, 2)),
        chain_pair_type={"A": {"L": "protein"}},
        numres=2,
        ligand_atoms=[lig1, lig2],
        protein_atoms=[prot_atom1, prot_atom2]
    )
    
    # Create synthetic PAE matrix (4x4)
    # Indices: 0,1 (protein), 2,3 (ligand)
    full_pae_matrix = np.array([
        [0, 5, 2, 15], # Prot 1 to others (close to Lig 1: PAE=2)
        [5, 0, 15, 2], # Prot 2 to others (close to Lig 2: PAE=2)
        [2, 15, 0, 5], # Lig 1 to others
        [15, 2, 5, 0]  # Lig 2 to others
    ])
    pae = PAEData(
        pae_matrix=full_pae_matrix[:2, :2],
        full_pae_matrix=full_pae_matrix,
        atom_plddts=np.array([90.0, 80.0, 90.0, 80.0]),
        plddt=np.array([90.0, 80.0]),
        cb_plddt=np.array([90.0, 80.0]),
        iptm_dict={"A": {"L": 0.8}}
    )
    
    # Calculate scores
    # Distances: 
    # Lig 1 to Prot 1: 2.0 (Passes < 4.0)
    # Lig 2 to Prot 2: 2.0 (Passes < 4.0)
    # PAEs:
    # Lig 1 to Prot 1: 2.0 (Passes < 3.0)
    # Lig 2 to Prot 2: 2.0 (Passes < 3.0)
    
    scores = calculate_ligand_scores(structure, pae, ligand_dist_cutoff=4.0, ligand_pae_cutoff=3.0)
    
    assert len(scores) == 1
    res = scores[0]
    assert res.LigandChn == "L"
    assert res.npair == 2
    assert res.nligatoms == 2
    assert res.nres == 2
    assert res.pLDDT == 85.0 # (90+80)/2
    assert res.PAE == 3.0 # Cutoff used
    assert res.Dist == 4.0 # Cutoff used

@pytest.mark.skipif(
    not list(LIGAND_DATA_DIR.glob("*.cif")),
    reason="No ligand CIF files found in tests/data/ligand_test",
)
def test_ligand_scoring_integration(tmp_path):
    """Integration test using real files if provided by user."""
    cif_file = next(LIGAND_DATA_DIR.glob("*.cif"))
    # Assume PAE file is either .json or .json.gz with same base name
    pae_file = next(LIGAND_DATA_DIR.glob("*.json*"))
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipsae.ipsae",
            str(pae_file),
            str(cif_file),
            "10", # PAE cutoff
            "10", # Dist cutoff
            "-o",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    
    # Check if ligand output files exist
    ligand_txt = list(output_dir.glob("*_ligands.txt"))
    ligand_csv = list(output_dir.glob("*_ligands.csv"))
    
    assert len(ligand_txt) > 0, "Ligand TXT output missing"
    assert len(ligand_csv) > 0, "Ligand CSV output missing"
    
    # Verify content of TXT output
    with open(ligand_txt[0]) as f:
        content = f.read()
        assert "Ligand" in content
        assert "npair" in content
