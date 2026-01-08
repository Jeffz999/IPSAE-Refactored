"""PAE loading logic for various model types."""

import json
import logging
from pathlib import Path

import numpy as np

from ipsae.models import InputModelType, PAEData, StructureData
from ipsae.utils import init_chainpairdict_zeros

logger = logging.getLogger("ipSAE")


def load_obj_from_file(file_path: Path):
    """Load an object from a file (numpy .npy/.npz/.pkl or JSON).

    Args:
        file_path: Path to the file.

    Returns:
        The loaded object (numpy array or dictionary).

    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix in {".npy", ".npz"}:
        return np.load(file_path)
    elif file_path.suffix == ".pkl":
        return np.load(file_path, allow_pickle=True)
    elif file_path.suffix == ".json":
        with file_path.open() as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_pae_data(
    pae_path: Path, structure_data: StructureData, model_type: InputModelType
) -> PAEData:
    """Load PAE, pLDDT, and other scores from various file formats.

    In addition to `pae_path` passed by the user, we also try to load score
    files based on inferred model types.

    | Model type | File type | Filename pattern                          |
    |-----------:|:----------|:------------------------------------------|
    | AF3 server | Structure | fold_[name]_model_0.cif                   |
    |            | PAE       | fold_[name]_full_data_0.json              |
    |            | ipTM      | fold_[name]_summary_confidences_0.json    |
    | AF3 local  | Structure | model1.cif                                |
    |            | PAE       | confidences.json                          |
    |            | ipTM      | summary_confidences.json                  |
    | Boltz      | Structure | [name]_model_0.cif                        |
    |            | PAE       | pae_[name]_model_0.npz                    |
    |            | ipTM      | confidence_[name]_model_0.json            |
    |            | plDDT     | plddt_[name]_model_0.npz                  |
    | AF2        | Structure | *.pdb                                     |
    |            | PAE       | *.json                                    |
    | Chai-1     | Structure | pred.model_idx_0.cif                      |
    |            | PAE       | pae.model_idx_0.npy                       |
    |            | plDDT     | plddt.model_idx_0.npy                     |
    |            | ipTM      | scores.model_idx_0.npz                    |

    NOTE: support for Chai-1 models require generating PAE and plDDT files separately.
        plDDT needs to be extracted from mmCIF files, or dumped after chai_lab.chai1.run_inference.
        pae needs to be dumped from chai_lab.chai1.run_inference into (N, N) npy files.
        ptm, iptm, per_chain_pair_iptm are in the scores npz files.

    Args:
        pae_path: Path to the PAE file.
        structure_data: Parsed structure data (needed for mapping atoms/residues).
        model_type: Type of model ('af2', 'af3', 'boltz1').

    Returns:
    -------
        A PAEData object containing the loaded scores.

    """
    if not pae_path.exists():
        raise FileNotFoundError(f"PAE file not found: {pae_path}")

    unique_chains = structure_data.unique_chains
    numres = structure_data.numres
    token_array = structure_data.token_mask
    mask_bool = token_array.astype(bool)

    # Initialize scores to be loaded
    pae_matrix = np.zeros((numres, numres))
    plddt = np.zeros(numres)
    cb_plddt = np.zeros(numres)
    iptm_dict = init_chainpairdict_zeros(unique_chains.tolist(), 0.0)
    iptm_val = -1.0
    ptm_val = -1.0

    if model_type is InputModelType.AF2:
        # Load all scores from input PAE file
        data = load_obj_from_file(pae_path)

        iptm_val = float(data.get("iptm", -1.0))
        ptm_val = float(data.get("ptm", -1.0))

        if "plddt" in data:
            plddt = np.array(data["plddt"])
            cb_plddt = np.array(data["plddt"])  # for pDockQ
        else:
            logger.warning(
                f"pLDDT scores not found in {model_type.name} PAE file: {pae_path}"
            )

        if "pae" in data:
            pae_matrix = np.array(data["pae"])
        elif "predicted_aligned_error" in data:
            pae_matrix = np.array(data["predicted_aligned_error"])
        else:
            logger.warning(
                f"PAE matrix not found in {model_type.name} PAE file: {pae_path}"
            )

    elif model_type is InputModelType.Boltz1 or model_type is InputModelType.Boltz2:
        # Load pLDDT if file exists
        plddt_path = pae_path.with_name(pae_path.name.replace("pae_", "plddt_", 1))
        if plddt_path.exists():
            logger.debug(f"Loading {model_type.name} pLDDT from file: {plddt_path}")
            data_plddt = load_obj_from_file(plddt_path)
            # Boltz plddt is 0-1, convert to 0-100
            plddt_boltz = np.array(100.0 * data_plddt["plddt"])

            # Filter by token mask
            plddt = plddt_boltz[np.ix_(mask_bool)]
            cb_plddt = plddt_boltz[np.ix_(mask_bool)]
        else:
            logger.warning(f"{model_type.name} pLDDT file not found: {plddt_path}")
            ntokens = np.sum(token_array)
            plddt = np.zeros(ntokens)
            cb_plddt = np.zeros(ntokens)

        # Load PAE matrix
        data_pae = load_obj_from_file(pae_path)
        pae_full = np.array(data_pae["pae"])
        pae_matrix = pae_full[np.ix_(mask_bool, mask_bool)]

        # Load ipTM scores if summary file exists
        summary_path = pae_path.with_name(
            pae_path.name.replace("pae", "confidence")
        ).with_suffix(".json")
        if summary_path.exists():
            logger.debug(
                f"Loading {model_type.name} confidence summary from file: {summary_path}"
            )
            data_summary = load_obj_from_file(summary_path)
            if "pair_chains_iptm" in data_summary:
                boltz_iptm = data_summary["pair_chains_iptm"]
                # Map indices to chains
                for i, c1 in enumerate(unique_chains):
                    for j, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        # Keys in json are strings of indices
                        iptm_dict[c1][c2] = boltz_iptm[str(i)][str(j)]
        else:
            logger.warning(
                f"{model_type.name} confidence summary file not found: {summary_path}"
            )

    elif model_type is InputModelType.AF3:
        data = load_obj_from_file(pae_path)

        atom_plddts = np.array(data["atom_plddts"])

        # Derive atom indices from structure data
        # Cbeta plDDTs are needed for pDockQ
        ca_indices = [r.atom_num - 1 for r in structure_data.residues]
        cb_indices = [r.atom_num - 1 for r in structure_data.cb_residues]

        plddt = atom_plddts[ca_indices]
        cb_plddt = atom_plddts[cb_indices]

        # Get pairwise residue PAE matrix by identifying one token per protein residue.
        # Modified residues have separate tokens for each atom, so need to pull out Calpha atom as token
        if "pae" in data:
            pae_full = np.array(data["pae"])
            pae_matrix = pae_full[np.ix_(mask_bool, mask_bool)]
        else:
            raise ValueError(
                f"PAE matrix not found in {model_type.name} PAE file: {pae_path}"
            )

        # Get iptm matrix from AF3 summary_confidences file
        summary_path = None
        pae_filename = pae_path.name
        if "confidences" in pae_filename:  # AF3 local
            summary_path = pae_path.with_name(
                pae_filename.replace("confidences", "summary_confidences")
            )
        elif "full_data" in pae_filename:  # AF3 server
            summary_path = pae_path.with_name(
                pae_filename.replace("full_data", "summary_confidences")
            )

        if summary_path and summary_path.exists():
            logger.debug(
                f"Loading {model_type.name} summary confidences from file: {summary_path}"
            )
            data_summary = load_obj_from_file(summary_path)
            if "chain_pair_iptm" in data_summary:
                af3_iptm = data_summary["chain_pair_iptm"]
                for i, c1 in enumerate(unique_chains):
                    for j, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        iptm_dict[c1][c2] = af3_iptm[i][j]
            else:
                logger.warning(
                    f"{model_type.name} summary confidences file missing key 'chain_pair_iptm': {summary_path}"
                )
        elif summary_path:
            logger.warning(
                f"{model_type.name} summary confidences file not found: {summary_path}"
            )
        else:
            logger.warning(
                f"Could not determine {model_type.name} summary confidences file path from PAE file: {pae_path}"
            )

    elif model_type is InputModelType.Chai1:
        # Load pLDDT if file exists
        plddt_path = pae_path.with_name(pae_path.name.replace("pae.", "plddt.", 1))
        if plddt_path.exists():
            logger.debug(f"Loading {model_type.name} pLDDT from file: {plddt_path}")
            data_plddt = load_obj_from_file(plddt_path)
            # Chai-1 plddt is 0-1, convert to 0-100
            plddt_chai1 = np.array(100.0 * data_plddt)

            # Filter by token mask
            plddt = plddt_chai1[np.ix_(mask_bool)]
            cb_plddt = plddt_chai1[np.ix_(mask_bool)]
        else:
            logger.warning(f"{model_type.name} pLDDT file not found: {plddt_path}")
            ntokens = np.sum(token_array)
            plddt = np.zeros(ntokens)
            cb_plddt = np.zeros(ntokens)

        # Load PAE matrix
        pae_full = load_obj_from_file(pae_path)
        pae_matrix = pae_full[np.ix_(mask_bool, mask_bool)]

        # Load ipTM scores if summary file exists
        summary_path = pae_path.with_name(
            pae_path.name.replace("pae.", "scores.", 1)
        ).with_suffix(".npz")
        if summary_path.exists():
            logger.debug(
                f"Loading {model_type.name} confidence summary from file: {summary_path}"
            )
            data_summary = load_obj_from_file(summary_path)
            ptm_val = float(data_summary.get("ptm", -1.0))
            iptm_val = float(data_summary.get("iptm", -1.0))

            if "per_chain_pair_iptm" in data_summary:  # (1, N_chains, N_chains)
                boltz_iptm = data_summary["per_chain_pair_iptm"][0]
                # Map indices to chains
                for i, c1 in enumerate(unique_chains):
                    for j, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        # Keys in json are strings of indices
                        iptm_dict[c1][c2] = boltz_iptm[i][j]
        else:
            logger.warning(
                f"{model_type.name} confidence summary file not found: {summary_path}"
            )

    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    return PAEData(
        pae_matrix=pae_matrix,
        plddt=plddt,
        cb_plddt=cb_plddt,
        iptm_dict=iptm_dict,
        ptm=ptm_val,
        iptm=iptm_val,
    )
