"""Structure parsing logic for PDB and mmCIF files using Biopython."""

import logging
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

from ipsae.constants import NUC_RESIDUE_SET, RESIDUE_SET
from ipsae.models import LigandAtom, Residue, StructureData
from ipsae.utils import init_chainpairdict_zeros

logger = logging.getLogger("ipSAE")


def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> dict[str, str]:
    """Classify chains as 'protein' or 'nucleic_acid' based on residue types for d0 calculation.

    Args:
        chains: Array of chain identifiers.
        residue_types: Array of residue names.

    Returns:
    -------
        Dictionary mapping chain ID to type ('protein' or 'nucleic_acid').

    """
    chain_types = {}
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        nuc_count = sum(r in NUC_RESIDUE_SET for r in chain_residues)
        chain_types[chain] = "nucleic_acid" if nuc_count > 0 else "protein"
    return chain_types


def load_structure(struct_path: Path) -> StructureData:
    """Parse a PDB or mmCIF file to extract structure data using Biopython.

    Reads the file to identify residues, coordinates (CA and CB), and chains.
    Calculates the pairwise distance matrix between residues.
    Classifies chain pairs as protein-protein, protein-nucleic acid, etc.
    Also identifies ligand atoms and their corresponding PAE indices.

    Args:
        struct_path: Path to the PDB or mmCIF file.

    Returns:
    -------
        A StructureData object containing the parsed information.

    """
    is_cif = struct_path.suffix == ".cif"
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("model", str(struct_path))
        if structure is None:
            raise ValueError(f"Failed to parse structure file {struct_path}")
    except Exception as e:
        logger.error(f"Failed to parse structure file {struct_path}: {e}")
        raise

    model = next(structure.get_models())

    residues = []
    cb_residues = []
    chains_list = []
    token_mask = []
    ligand_atoms = []
    protein_atoms = []

    global_pae_idx = 0

    for chain in model:
        for residue in chain:
            res_name = residue.get_resname().strip()
            # Check if it's a standard residue (Protein or Nucleic Acid)
            # AF3/Boltz might use HETATM for modified residues, but they are still 1 token.
            if res_name in RESIDUE_SET:
                # 1 token per residue
                token_mask.append(1)

                # Representative atoms for ipSAE/pDockQ
                ca_atom = None
                if "CA" in residue:
                    ca_atom = residue["CA"]
                elif "C1'" in residue:
                    ca_atom = residue["C1'"]

                if ca_atom:
                    res_obj = Residue(
                        atom_num=ca_atom.get_serial_number(),
                        coor=ca_atom.get_coord(),
                        res=res_name,
                        chainid=chain.id,
                        resnum=residue.id[1],
                        residue_str=f"{res_name:3}   {chain.id:3} {residue.id[1]:4}",
                    )
                    residues.append(res_obj)
                    chains_list.append(chain.id)

                cb_atom = None
                if "CB" in residue:
                    cb_atom = residue["CB"]
                elif "C3'" in residue:
                    cb_atom = residue["C3'"]
                elif res_name == "GLY" and "CA" in residue:
                    cb_atom = residue["CA"]

                if cb_atom:
                    cb_res_obj = Residue(
                        atom_num=cb_atom.get_serial_number(),
                        coor=cb_atom.get_coord(),
                        res=res_name,
                        chainid=chain.id,
                        resnum=residue.id[1],
                        residue_str=f"{res_name:3}   {chain.id:3} {residue.id[1]:4}",
                    )
                    cb_residues.append(cb_res_obj)

                # Collect all non-H atoms for ligand-protein contact calculation
                for atom in residue:
                    if atom.element == "H":
                        continue
                    protein_atoms.append(
                        LigandAtom(
                            atom_num=atom.get_serial_number(),
                            coor=atom.get_coord(),
                            res=res_name,
                            chainid=chain.id,
                            resnum=residue.id[1],
                            atom_name=atom.get_name(),
                            element=atom.element,
                            global_pae_idx=global_pae_idx,
                        )
                    )

                global_pae_idx += 1
            else:
                # Ligand: 1 token per atom (excluding Hydrogens)
                for atom in residue:
                    if atom.element == "H":
                        continue
                    token_mask.append(0)
                    ligand_atoms.append(
                        LigandAtom(
                            atom_num=atom.get_serial_number(),
                            coor=atom.get_coord(),
                            res=res_name,
                            chainid=chain.id,
                            resnum=residue.id[1],
                            atom_name=atom.get_name(),
                            element=atom.element,
                            global_pae_idx=global_pae_idx,
                        )
                    )
                    global_pae_idx += 1

    # Convert structure information to numpy arrays
    numres = len(residues)
    coordinates = np.array([r.coor for r in cb_residues])
    chains = np.array(chains_list)

    # Unique chains in order of appearance
    unique_chains_list = []
    for c in chains_list:
        if c not in unique_chains_list:
            unique_chains_list.append(c)
    # Also add ligand chains
    for la in ligand_atoms:
        if la.chainid not in unique_chains_list:
            unique_chains_list.append(la.chainid)
    unique_chains = np.array(unique_chains_list)

    token_array = np.array(token_mask)
    residue_types = np.array([r.res for r in residues])

    chain_dict = classify_chains(chains, residue_types)
    # Add ligand chains to chain_dict
    for la in ligand_atoms:
        if la.chainid not in chain_dict:
            chain_dict[la.chainid] = "ligand"

    chain_pair_type = init_chainpairdict_zeros(unique_chains_list, "0")
    for c1 in unique_chains_list:
        for c2 in unique_chains_list:
            if c1 == c2:
                continue
            if chain_dict.get(c1) == "ligand" or chain_dict.get(c2) == "ligand":
                chain_pair_type[c1][c2] = "ligand"
            elif (
                chain_dict.get(c1) == "nucleic_acid"
                or chain_dict.get(c2) == "nucleic_acid"
            ):
                chain_pair_type[c1][c2] = "nucleic_acid"
            else:
                chain_pair_type[c1][c2] = "protein"

    # Distance matrix (for protein-protein)
    if len(coordinates) > 0:
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.sqrt((diff**2).sum(axis=2))
    else:
        distances = np.zeros((0, 0), dtype=float)

    return StructureData(
        residues=residues,
        cb_residues=cb_residues,
        chains=chains,
        unique_chains=unique_chains,
        token_mask=token_array,
        residue_types=residue_types,
        coordinates=coordinates,
        distances=distances,
        chain_pair_type=chain_pair_type,
        numres=numres,
        ligand_atoms=ligand_atoms,
        protein_atoms=protein_atoms,
    )
