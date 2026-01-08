"""Structure parsing logic for PDB and mmCIF files."""

import logging
from pathlib import Path

import numpy as np

from ipsae.constants import NUC_RESIDUE_SET, RESIDUE_SET
from ipsae.models import Residue, StructureData
from ipsae.utils import init_chainpairdict_zeros

logger = logging.getLogger("ipSAE")


def parse_pdb_atom_line(line: str) -> dict | None:
    """Parse a line from a PDB file.

    Args:
        line: A line from a PDB file starting with ATOM or HETATM.

    Returns:
    -------
        A dictionary containing atom details, or None if parsing fails.

    """
    try:
        atom_num = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21].strip()
        residue_seq_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except ValueError as e:
        logger.debug(f"Failed to parse PDB line: {line.strip()} with error: {e}")
        return None
    else:
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": residue_seq_num,
            "x": x,
            "y": y,
            "z": z,
        }


def parse_cif_atom_line(line: str, fielddict: dict[str, int]) -> dict | None:
    """Parse a line from an mmCIF file.

    Note that ligands do not have residue numbers, but modified residues do.
    We return `None` for ligands.

    Args:
        line: A line from an mmCIF file.
        fielddict: Dictionary mapping field names to column indices.

    Returns:
    -------
        A dictionary containing atom details, or None if parsing fails or it's a ligand.

    """
    linelist = line.split()
    try:
        residue_seq_num_str = linelist[fielddict["label_seq_id"]]
        if residue_seq_num_str == ".":
            return None  # ligand

        atom_num = int(linelist[fielddict["id"]])
        atom_name = linelist[fielddict["label_atom_id"]]
        residue_name = linelist[fielddict["label_comp_id"]]
        chain_id = linelist[fielddict["label_asym_id"]]
        residue_seq_num = int(residue_seq_num_str)
        x = float(linelist[fielddict["Cartn_x"]])
        y = float(linelist[fielddict["Cartn_y"]])
        z = float(linelist[fielddict["Cartn_z"]])
    except (ValueError, IndexError, KeyError) as e:
        logger.debug(f"Failed to parse mmCIF line: {line.strip()} with error: {e}")
        return None
    else:
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": residue_seq_num,
            "x": x,
            "y": y,
            "z": z,
        }


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
    """Parse a PDB or mmCIF file to extract structure data.

    Reads the file to identify residues, coordinates (CA and CB), and chains.
    Calculates the pairwise distance matrix between residues.
    Classifies chain pairs as protein-protein, protein-nucleic acid, etc.

    Args:
        struct_path: Path to the PDB or mmCIF file.

    Returns:
    -------
        A StructureData object containing the parsed information.

    """
    residues = []
    cb_residues = []
    chains_list = []

    # For af3 and boltz1: need mask to identify CA atom tokens in plddt vector and pae matrix;
    # Skip ligand atom tokens and non-CA-atom tokens in PTMs (those not in RESIDUE_SET)
    token_mask = []
    atomsitefield_dict = {}
    atomsitefield_num = 0

    is_cif = struct_path.suffix == ".cif"

    with struct_path.open() as f:
        for raw_line in f:
            # mmCIF _atom_site loop headers
            if raw_line.startswith("_atom_site."):
                line = raw_line.strip()
                parts = line.split(".")
                if len(parts) == 2:
                    atomsitefield_dict[parts[1]] = atomsitefield_num
                    atomsitefield_num += 1

            # Atom coordinates
            if raw_line.startswith(("ATOM", "HETATM")):
                if is_cif:
                    atom = parse_cif_atom_line(raw_line, atomsitefield_dict)
                else:
                    atom = parse_pdb_atom_line(raw_line)

                if atom is None:  # Ligand or parse error
                    token_mask.append(0)
                    continue

                # CA or C1' (nucleic acid)
                if (atom["atom_name"] == "CA") or ("C1" in atom["atom_name"]):
                    token_mask.append(1)
                    res_obj = Residue(
                        atom_num=atom["atom_num"],
                        coor=np.array([atom["x"], atom["y"], atom["z"]]),
                        res=atom["residue_name"],
                        chainid=atom["chain_id"],
                        resnum=atom["residue_seq_num"],
                        residue_str=f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    )
                    residues.append(res_obj)
                    chains_list.append(atom["chain_id"])

                # CB or C3' or GLY CA
                if (
                    (atom["atom_name"] == "CB")
                    or ("C3" in atom["atom_name"])
                    or (atom["residue_name"] == "GLY" and atom["atom_name"] == "CA")
                ):
                    res_obj = Residue(
                        atom_num=atom["atom_num"],
                        coor=np.array([atom["x"], atom["y"], atom["z"]]),
                        res=atom["residue_name"],
                        chainid=atom["chain_id"],
                        resnum=atom["residue_seq_num"],
                        residue_str=f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    )
                    cb_residues.append(res_obj)

                # Non-CA/C1' atoms in standard residues -> token 0
                # Nucleic acids and non-CA atoms in PTM residues to tokens (as 0), whether labeled as "HETATM" (af3) or as "ATOM" (boltz1)
                if (
                    (atom["atom_name"] != "CA")
                    and ("C1" not in atom["atom_name"])
                    and (atom["residue_name"] not in RESIDUE_SET)
                ):
                    token_mask.append(0)

    # Convert structure information to numpy arrays
    numres = len(residues)
    coordinates = np.array([r.coor for r in cb_residues])
    chains = np.array(chains_list)
    _, uniq_chain_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(uniq_chain_idx)]
    token_array = np.array(token_mask)
    residue_types = np.array([r.res for r in residues])

    chain_dict = classify_chains(chains, residue_types)
    unique_chains_list = unique_chains.tolist()
    chain_pair_type = init_chainpairdict_zeros(unique_chains_list, "0")
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            if chain_dict[c1] == "nucleic_acid" or chain_dict[c2] == "nucleic_acid":
                chain_pair_type[c1][c2] = "nucleic_acid"
            else:
                chain_pair_type[c1][c2] = "protein"

    # Distance matrix
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
    )
