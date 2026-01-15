"""Core scoring engine for ipSAE, pDockQ, and LIS."""

import math

import numpy as np

from ipsae.constants import CHAIN_COLOR, LIS_PAE_CUTOFF
from ipsae.models import (
    ChainPairScoreResults,
    LigandScoreResults,
    PAEData,
    PerResScoreResults,
    Residue,
    ScoreResults,
    StructureData,
)
from ipsae.parsers.chains import (
    chain_group_name,
    get_chain_group_indices,
    parse_chain_groups,
)
from ipsae.utils import (
    calc_d0,
    calc_d0_array,
    contiguous_ranges,
    ptm_func_vec,
)


def calculate_pdockq_scores(
    chains: np.ndarray,
    chain_pairs: list[tuple[list[str], list[str]]],
    distances: np.ndarray,
    pae_matrix: np.ndarray,
    cb_plddt: np.ndarray,
    pdockq_dist_cutoff: float = 8.0,
):
    """Calculate pDockQ and pDockQ2 scores for specified chain pairs."""
    pDockQ: dict[str, dict[str, float]] = {}
    pDockQ2: dict[str, dict[str, float]] = {}

    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        # Initialize nested dict if needed
        if g1 not in pDockQ:
            pDockQ[g1] = {}
            pDockQ2[g1] = {}

        # Get indices for chain groups
        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            pDockQ[g1][g2] = 0.0
            pDockQ2[g1][g2] = 0.0
            continue

        # Submatrix of distances
        dists_sub = distances[np.ix_(g1_indices, g2_indices)]
        valid_mask = dists_sub <= pdockq_dist_cutoff
        npairs = np.sum(valid_mask)

        if npairs > 0:
            # Identify interface residues on g1 and g2
            # Any column in valid_mask with at least one True
            g1_interface_mask = valid_mask.any(axis=1)
            g1_interface_indices = g1_indices[g1_interface_mask]
            g2_interface_mask = valid_mask.any(axis=0)
            g2_interface_indices = g2_indices[g2_interface_mask]

            mean_plddt = cb_plddt[
                np.hstack([g1_interface_indices, g2_interface_indices])
            ].mean()
            x = mean_plddt * np.log10(npairs)
            pDockQ[g1][g2] = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

            # Also find sub-matrix for pDockQ2 calculation
            pae_sub = pae_matrix[np.ix_(g1_indices, g2_indices)]
            pae_valid = pae_sub[valid_mask]
            pae_ptm_sum = ptm_func_vec(pae_valid, 10.0).sum()

            mean_ptm = pae_ptm_sum / npairs
            x = mean_plddt * mean_ptm
            pDockQ2[g1][g2] = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
        else:
            pDockQ[g1][g2] = 0.0
            pDockQ2[g1][g2] = 0.0

    return pDockQ, pDockQ2


def calculate_lis(
    chains: np.ndarray,
    chain_pairs: list[tuple[list[str], list[str]]],
    pae_matrix: np.ndarray,
):
    """Calculate LIS scores for specified chain pairs."""
    LIS: dict[str, dict[str, float]] = {}

    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        # Initialize nested dict if needed
        if g1 not in LIS:
            LIS[g1] = {}

        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            LIS[g1][g2] = 0.0
            continue

        pae_sub = pae_matrix[np.ix_(g1_indices, g2_indices)]
        valid_pae = pae_sub[pae_sub <= LIS_PAE_CUTOFF]
        if valid_pae.size > 0:
            scores = (LIS_PAE_CUTOFF - valid_pae) / LIS_PAE_CUTOFF
            LIS[g1][g2] = float(np.mean(scores))
        else:
            LIS[g1][g2] = 0.0

    return LIS


def aggregate_byres_scores(
    residues: list[Residue],
    pae_cutoff: int | float,
    dist_cutoff: int | float,
    chain_pairs: list[tuple[list[str], list[str]]],
    ipsae_d0res_byres: dict[str, dict[str, np.ndarray]],
    ipsae_d0chn_byres: dict[str, dict[str, np.ndarray]],
    ipsae_d0dom_byres: dict[str, dict[str, np.ndarray]],
    iptm_d0chn_byres: dict[str, dict[str, np.ndarray]],
    n0res_byres: dict[str, dict[str, np.ndarray]],
    d0res_byres: dict[str, dict[str, np.ndarray]],
    unique_residues_chain1: dict[str, dict[str, set]],
    unique_residues_chain2: dict[str, dict[str, set]],
    dist_unique_residues_chain1: dict[str, dict[str, set]],
    dist_unique_residues_chain2: dict[str, dict[str, set]],
    pae_data: PAEData,
    pDockQ: dict[str, dict[str, float]],
    pDockQ2: dict[str, dict[str, float]],
    LIS: dict[str, dict[str, float]],
    n0chn: dict[str, dict[str, int]],
    n0dom: dict[str, dict[str, int]],
    d0chn: dict[str, dict[str, float]],
    d0dom: dict[str, dict[str, float]],
    label: str,
) -> tuple[list[ChainPairScoreResults], list[str], dict[str, dict[str, float]]]:
    """Aggregate per-residue scores into chain-pair-specific scores.

    Returns:
        A tuple containing:
        - List of ChainPairScoreResults objects with chain-pair scores.
        - List of PyMOL script lines.
        - Dictionary of metrics for each chain pair.

    """
    # Store results in a structured way
    results_metrics: dict[str, dict[str, float]] = {}

    chain_pair_scores: list[ChainPairScoreResults] = []

    pymol_lines = [
        "# Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS      n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"
    ]

    # Helper to get max info
    def get_max_info(values_array, c1, c2):
        """Get max value and corresponding residue info from by-residue arrays."""
        vals = values_array[c1][c2]
        if np.all(vals == 0):
            return 0.0, "None", 0
        idx = np.argmax(vals)
        return vals[idx], residues[idx].residue_str, idx

    # Build a map from group names to chain lists
    group_to_chains: dict[str, list[str]] = {}
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)
        group_to_chains[g1] = group1
        group_to_chains[g2] = group2

    # Identify all unique unordered pairs from chain_pairs
    unique_pair_set: set[tuple[str, str]] = set()
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)
        if g1 != g2:
            pair_key: tuple[str, str] = tuple(sorted((g1, g2)))  # type: ignore
            unique_pair_set.add(pair_key)

    # Helper to get max across both directions
    def get_max_of_pair(arr, k1, k2):
        v1, _, i1 = get_max_info(arr, k1, k2)
        v2, _, i2 = get_max_info(arr, k2, k1)
        if v1 >= v2:
            return v1, i1, k1, k2
        return v2, i2, k2, k1

    # Process each unique pair: output both asym directions + max together
    for pair_key in sorted(unique_pair_set):
        g_a, g_b = pair_key  # Alphabetically sorted

        # Check if data exists for both directions
        if g_a not in ipsae_d0res_byres or g_b not in ipsae_d0res_byres.get(g_a, {}):
            continue
        if g_b not in ipsae_d0res_byres or g_a not in ipsae_d0res_byres.get(g_b, {}):
            continue

        # Process both asym directions: g_a -> g_b, then g_b -> g_a
        for g1, g2 in [(g_a, g_b), (g_b, g_a)]:
            group1 = group_to_chains[g1]
            group2 = group_to_chains[g2]

            # Asym values
            ipsae_res_val, _, ipsae_res_idx = get_max_info(ipsae_d0res_byres, g1, g2)
            ipsae_chn_val, _, _ = get_max_info(ipsae_d0chn_byres, g1, g2)
            ipsae_dom_val, _, _ = get_max_info(ipsae_d0dom_byres, g1, g2)
            iptm_chn_val, _, _ = get_max_info(iptm_d0chn_byres, g1, g2)

            # Get n0res/d0res at max index
            n0res_val = n0res_byres[g1][g2][ipsae_res_idx]
            d0res_val = d0res_byres[g1][g2][ipsae_res_idx]

            # Counts
            res1_cnt = len(unique_residues_chain1[g1][g2])
            res2_cnt = len(unique_residues_chain2[g1][g2])
            dist1_cnt = len(dist_unique_residues_chain1[g1][g2])
            dist2_cnt = len(dist_unique_residues_chain2[g1][g2])

            # ipTM AF - for chain groups, try individual chains first
            iptm_af = 0.0
            if len(group1) == 1 and len(group2) == 1:
                # Individual chains - try to get from iptm_dict
                single_c1, single_c2 = group1[0], group2[0]
                if (
                    single_c1 in pae_data.iptm_dict
                    and single_c2 in pae_data.iptm_dict.get(single_c1, {})
                ):
                    iptm_af = pae_data.iptm_dict[single_c1][single_c2]
            if iptm_af == 0.0 and pae_data.iptm != -1.0:
                iptm_af = pae_data.iptm  # Fallback to global

            summary_result = ChainPairScoreResults(
                Chn1=g1,
                Chn2=g2,
                PAE=pae_cutoff,
                Dist=dist_cutoff,
                Type="asym",
                ipSAE=float(ipsae_res_val),
                ipSAE_d0chn=float(ipsae_chn_val),
                ipSAE_d0dom=float(ipsae_dom_val),
                ipTM_af=float(iptm_af),
                ipTM_d0chn=float(iptm_chn_val),
                pDockQ=float(pDockQ[g1][g2]),
                pDockQ2=float(pDockQ2[g1][g2]),
                LIS=float(LIS[g1][g2]),
                n0res=int(n0res_val),
                n0chn=int(n0chn[g1][g2]),
                n0dom=int(n0dom[g1][g2]),
                d0res=float(d0res_val),
                d0chn=float(d0chn[g1][g2]),
                d0dom=float(d0dom[g1][g2]),
                nres1=res1_cnt,
                nres2=res2_cnt,
                dist1=dist1_cnt,
                dist2=dist2_cnt,
                Model=label,
            )
            chain_pair_scores.append(summary_result)
            pymol_lines.append("# " + summary_result.to_formatted_line(end="\n"))

            # Store in results dict
            results_metrics[f"{g1}_{g2}"] = {
                "ipsae": float(ipsae_res_val),
                "iptm": float(iptm_af),
                "pdockq": float(pDockQ[g1][g2]),
                "pdockq2": float(pDockQ2[g1][g2]),
                "lis": float(LIS[g1][g2]),
            }

            # PyMOL script generation - handle multi-chain groups
            # Use '-' instead of '+' in alias names for compatibility
            pymol_g1 = g1.replace("+", "-")
            pymol_g2 = g2.replace("+", "-")
            chain_pair_name = f"color_{pymol_g1}_{pymol_g2}"

            # Ranges
            r1_ranges = contiguous_ranges(unique_residues_chain1[g1][g2])
            r2_ranges = contiguous_ranges(unique_residues_chain2[g1][g2])

            # Build color commands for each chain in the groups
            color_cmds = ["color gray80, all"]
            for chain_id in group1:
                color = CHAIN_COLOR.get(chain_id, "magenta")
                if r1_ranges:
                    color_cmds.append(
                        f"color {color}, chain {chain_id} and resi {r1_ranges}"
                    )
            for chain_id in group2:
                color = CHAIN_COLOR.get(chain_id, "marine")
                if r2_ranges:
                    color_cmds.append(
                        f"color {color}, chain {chain_id} and resi {r2_ranges}"
                    )

            pymol_lines.append(f"alias {chain_pair_name}, {'; '.join(color_cmds)}\n\n")

        # Now add the max row for this unique pair
        # Original code uses larger chain first (c1 > c2), then outputs (c2, c1)
        # We already have g_a < g_b alphabetically sorted
        # So to match: use g_b (larger) as g1, g_a (smaller) as g2
        g1, g2 = g_b, g_a

        # Calculate max values
        ipsae_res_max, idx_res, mk1_res, mk2_res = get_max_of_pair(
            ipsae_d0res_byres, g1, g2
        )
        ipsae_chn_max, _, _, _ = get_max_of_pair(ipsae_d0chn_byres, g1, g2)
        ipsae_dom_max, _, _, _ = get_max_of_pair(ipsae_d0dom_byres, g1, g2)
        iptm_chn_max, _, _, _ = get_max_of_pair(iptm_d0chn_byres, g1, g2)

        # n0/d0 for max
        n0res_max = n0res_byres[mk1_res][mk2_res][idx_res]
        d0res_max = d0res_byres[mk1_res][mk2_res][idx_res]

        # n0dom/d0dom for max
        v1_dom, _, _ = get_max_info(ipsae_d0dom_byres, g1, g2)
        v2_dom, _, _ = get_max_info(ipsae_d0dom_byres, g2, g1)
        if v1_dom >= v2_dom:
            n0dom_max = n0dom[g1][g2]
            d0dom_max = d0dom[g1][g2]
        else:
            n0dom_max = n0dom[g2][g1]
            d0dom_max = d0dom[g2][g1]

        # iptm af max - for individual chains, use iptm_dict; for groups, use global
        iptm_af_1 = pae_data.iptm_dict.get(g1, {}).get(g2, 0.0)
        iptm_af_2 = pae_data.iptm_dict.get(g2, {}).get(g1, 0.0)
        if iptm_af_1 == 0 and pae_data.iptm != -1.0:
            iptm_af_1 = pae_data.iptm
        if iptm_af_2 == 0 and pae_data.iptm != -1.0:
            iptm_af_2 = pae_data.iptm
        iptm_af_max = max(iptm_af_1, iptm_af_2)

        pdockq2_max = max(pDockQ2[g1][g2], pDockQ2[g2][g1])
        lis_avg = (LIS[g1][g2] + LIS[g2][g1]) / 2.0

        # Residue counts (max of cross pairs)
        res1_max = max(
            len(unique_residues_chain2[g1][g2]), len(unique_residues_chain1[g2][g1])
        )
        res2_max = max(
            len(unique_residues_chain1[g1][g2]), len(unique_residues_chain2[g2][g1])
        )
        dist1_max = max(
            len(dist_unique_residues_chain2[g1][g2]),
            len(dist_unique_residues_chain1[g2][g1]),
        )
        dist2_max = max(
            len(dist_unique_residues_chain1[g1][g2]),
            len(dist_unique_residues_chain2[g2][g1]),
        )

        # Output with swapped chain names (Chn1=g2, Chn2=g1) to match original
        summary_result = ChainPairScoreResults(
            Chn1=g2,
            Chn2=g1,
            PAE=pae_cutoff,
            Dist=dist_cutoff,
            Type="max",
            ipSAE=float(ipsae_res_max),
            ipSAE_d0chn=float(ipsae_chn_max),
            ipSAE_d0dom=float(ipsae_dom_max),
            ipTM_af=float(iptm_af_max),
            ipTM_d0chn=float(iptm_chn_max),
            pDockQ=float(pDockQ[g1][g2]),
            pDockQ2=float(pdockq2_max),
            LIS=float(lis_avg),
            n0res=int(n0res_max),
            n0chn=int(n0chn[g1][g2]),
            n0dom=int(n0dom_max),
            d0res=float(d0res_max),
            d0chn=float(d0chn[g1][g2]),
            d0dom=float(d0dom_max),
            nres1=res1_max,
            nres2=res2_max,
            dist1=dist1_max,
            dist2=dist2_max,
            Model=label,
        )
        chain_pair_scores.append(summary_result)
        pymol_lines.append("# " + summary_result.to_formatted_line(end="\n") + "\n")

    return chain_pair_scores, pymol_lines, results_metrics


def calculate_ligand_scores(
    structure: StructureData,
    pae_data: PAEData,
    label: str = "model",
    ligand_dist_cutoff: float = 4.0,
    ligand_pae_cutoff: float = 3.0,
    d0_scaling: float = 1.24,
) -> tuple[list[LigandScoreResults], list[str]]:
    """Calculate binding metrics for ligands including normalized ipSAE.

    For each ligand, we find protein atoms within ligand_dist_cutoff.
    We then filter these contacts by PAE < ligand_pae_cutoff.
    """
    if not structure.ligand_atoms or not structure.protein_atoms:
        return [], []

    from scipy.spatial import KDTree

    # Protein atoms for KDTree
    protein_coords = np.array([pa.coor for pa in structure.protein_atoms])
    protein_tree = KDTree(protein_coords)

    # Group ligand atoms by chain
    ligand_chains = {}
    for la in structure.ligand_atoms:
        if la.chainid not in ligand_chains:
            ligand_chains[la.chainid] = []
        ligand_chains[la.chainid].append(la)

    results = []
    pymol_lines = []
    for chain_id, atoms in ligand_chains.items():
        # Ligand atoms for this chain
        lig_coords = np.array([a.coor for a in atoms])
        lig_pae_indices = [a.global_pae_idx for a in atoms]

        # Find protein atoms within distance cutoff
        neighbors_list = protein_tree.query_ball_point(lig_coords, ligand_dist_cutoff)

        valid_lig_atoms = set()
        valid_prot_residues = set()

        sum_plddt = 0.0
        sum_normalized_score = 0.0  # Accumulator for 1 / (1 + (PAE/d0)^2)
        total_valid_pairs = 0

        for i, prot_indices in enumerate(neighbors_list):
            lig_idx = lig_pae_indices[i]

            # pLDDT for this ligand atom
            if lig_idx < len(pae_data.atom_plddts):
                sum_plddt += pae_data.atom_plddts[lig_idx]

            for p_tree_idx in prot_indices:
                prot_atom = structure.protein_atoms[p_tree_idx]
                prot_idx = prot_atom.global_pae_idx

                # Get Raw PAE
                # Note: PAE matrix is [aligned_residue, scored_residue]
                # ChimeraX uses pae.value(ra1, ra2) which is matrix[ra1_idx, ra2_idx]
                # Here we use lig_idx as aligned, prot_idx as scored.
                pae_val = pae_data.full_pae_matrix[lig_idx, prot_idx]

                # Filter by confidence
                if pae_val <= ligand_pae_cutoff:
                    # Normalized Score Sum (for ipSAE)
                    # Formula: 1 / (1 + (PAE / d0)^2)
                    norm_score = 1.0 / (1.0 + (pae_val / d0_scaling) ** 2)
                    sum_normalized_score += norm_score

                    valid_prot_residues.add((prot_atom.chainid, prot_atom.resnum))
                    valid_lig_atoms.add(i)
                    total_valid_pairs += 1

        # Final Averaging
        # Note: ChimeraX often averages pLDDT over the selection.
        # If the user feedback implies avg_plddt is for the ligand, we keep it as is.
        # For AF3/Boltz, we should only average over atoms that have PAE data.
        num_atoms_with_plddt = sum(1 for i in range(len(atoms)) if lig_pae_indices[i] < len(pae_data.atom_plddts))
        avg_plddt = (sum_plddt / num_atoms_with_plddt) if num_atoms_with_plddt > 0 else 0.0

        if total_valid_pairs > 0:
            # ipSAE is the average normalized score over all valid contact pairs
            final_norm_ipsae = float(sum_normalized_score / total_valid_pairs)
        else:
            final_norm_ipsae = 0.0  # Min score

        results.append(
            LigandScoreResults(
                LigandChn=chain_id,
                PAE=ligand_pae_cutoff,
                Dist=ligand_dist_cutoff,
                ipSAE=final_norm_ipsae,
                pLDDT=float(avg_plddt),
                nligatoms=len(valid_lig_atoms),
                nres=len(valid_prot_residues),
                npair=total_valid_pairs,
                Model=label,
            )
        )
    
    # PyMOL script generation for ligands
    if results:
        pymol_lines.append("\n# Ligand visualization\n")
        pymol_lines.append("show sticks, hetatm\n")
        for res in results:
            if res.nligatoms > 0:
                pymol_lines.append(
                    f"# Ligand {res.LigandChn}: ipSAE={res.ipSAE:.3f}, n_contacts={res.npair}\n"
                )
                # We don't have the specific residue ranges here easily, but we can color the whole ligand chain
                pymol_lines.append(f"color yellow, chain {res.LigandChn}\n")

    return results, pymol_lines


def calculate_scores(
    structure: StructureData,
    pae_data: PAEData,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    ligand_pae_cutoff: float = 3.0,
    ligand_dist_cutoff: float = 4.0,
    label: str = "model",
    chain_groups: list[tuple[list[str], list[str]]] | None = None,
) -> ScoreResults:
    """Calculate chain-pair-specific ipSAE, ipTM, pDockQ, pDockQ2, and LIS scores.

    This is the main calculation engine. It iterates over all chain pairs and computes:
    - ipSAE: Inter-protein Predicted Aligned Error score.
    - ipTM: Inter-protein Template Modeling score.
    - pDockQ: Predicted DockQ score (Bryant et al.).
    - pDockQ2: Improved pDockQ score (Zhu et al.).
    - LIS: Local Interaction Score (Kim et al.).

    Nomenclature:
    - iptm_d0chn: calculate iptm from PAEs with no PAE cutoff
        d0 = numres in chain pair = len(chain1) + len(chain2)
    - ipsae_d0chn: calculate ipsae from PAEs with PAE cutoff
        d0 = numres in chain pair = len(chain1) + len(chain2)
    - ipsae_d0dom: calculate ipsae from PAEs with PAE cutoff
        d0 from number of residues in chain1 and chain2 that have interchain PAE<cutoff
    - ipsae_d0res: calculate ipsae from PAEs with PAE cutoff
        d0 from number of residues in chain2 that have interchain PAE<cutoff given residue in chain1

    for each chain_pair iptm/ipsae, there is (for example)
    - ipsae_d0res_byres: by-residue array;
    - ipsae_d0res_asym: asymmetric pair value (A->B is different from B->A)
    - ipsae_d0res_max: maximum of A->B and B->A value
    - ipsae_d0res_asymres: identify of residue that provides each asym maximum
    - ipsae_d0res_maxres: identify of residue that provides each maximum over both chains

    - n0num: number of residues in whole complex provided by AF2 model
    - n0chn: number of residues in chain pair = len(chain1) + len(chain2)
    - n0dom: number of residues in chain pair that have good PAE values (<cutoff)
    - n0res: number of residues in chain2 that have good PAE residues for each residue of chain1

    Args:
        structure: Parsed structure data.
        pae_data: Loaded PAE and pLDDT data.
        pae_cutoff: Cutoff for PAE to consider a residue pair "good" (default: 10.0).
        dist_cutoff: Distance cutoff for contact definition (default: 10.0).
        label: Filename prefix (for output labeling).
        chain_groups: Optional list of chain group pairs to calculate scores for.
            If None, all chain pairs are calculated.

    Returns:
    -------
        A ScoreResults object containing all calculated scores and output strings.

    """
    chains = structure.chains
    unique_chains = structure.unique_chains
    distances = structure.distances
    pae_matrix = pae_data.pae_matrix
    cb_plddt = pae_data.cb_plddt

    # Get unique chain group names for dictionary keys
    if chain_groups is None:
        chain_pairs = parse_chain_groups("...", unique_chains)
    else:
        chain_pairs = chain_groups
    chain_group_names = list(
        {chain_group_name(g) for pair in chain_pairs for g in pair}
    )

    # Calculate pDockQ and LIS scores
    pDockQ, pDockQ2 = calculate_pdockq_scores(
        chains, chain_pairs, distances, pae_matrix, cb_plddt
    )
    LIS = calculate_lis(chains, chain_pairs, pae_matrix)

    # --- Ligand Scores ---
    ligand_scores, ligand_pymol_lines = calculate_ligand_scores(
        structure, pae_data, label, ligand_dist_cutoff, ligand_pae_cutoff
    )

    # --- ipTM / ipSAE ---
    residues = structure.residues
    plddt = pae_data.plddt
    numres = structure.numres

    # Initialize containers using chain group names
    def init_chainpair_dict_zeros(default_val):
        return {
            c1: {c2: default_val for c2 in chain_group_names if c1 != c2}
            for c1 in chain_group_names
        }

    def init_chainpair_dict_npzeros():
        return {
            c1: {c2: np.zeros(numres) for c2 in chain_group_names if c1 != c2}
            for c1 in chain_group_names
        }

    def init_chainpair_dict_set():
        return {
            c1: {c2: set() for c2 in chain_group_names if c1 != c2}
            for c1 in chain_group_names
        }

    iptm_d0chn_byres = init_chainpair_dict_npzeros()
    ipsae_d0chn_byres = init_chainpair_dict_npzeros()
    ipsae_d0dom_byres = init_chainpair_dict_npzeros()
    ipsae_d0res_byres = init_chainpair_dict_npzeros()

    n0chn: dict[str, dict[str, int]] = init_chainpair_dict_zeros(0)
    d0chn: dict[str, dict[str, float]] = init_chainpair_dict_zeros(0.0)
    n0dom: dict[str, dict[str, int]] = init_chainpair_dict_zeros(0)
    d0dom: dict[str, dict[str, float]] = init_chainpair_dict_zeros(0.0)
    n0res_byres = init_chainpair_dict_npzeros()
    d0res_byres = init_chainpair_dict_npzeros()

    unique_residues_chain1 = init_chainpair_dict_set()
    unique_residues_chain2 = init_chainpair_dict_set()
    dist_unique_residues_chain1 = init_chainpair_dict_set()
    dist_unique_residues_chain2 = init_chainpair_dict_set()

    # Helper to determine pair type for a chain group pair
    def get_pair_type(group1: list[str], group2: list[str]) -> str:
        for c1 in group1:
            for c2 in group2:
                if (
                    c1 in structure.chain_pair_type
                    and c2 in structure.chain_pair_type.get(c1, {})
                ):
                    if structure.chain_pair_type[c1][c2] == "nucleic_acid":
                        return "nucleic_acid"
        return "protein"

    # First pass: d0chn
    # Calculate ipTM/ipSAE with and without PAE cutoff
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            continue

        pair_type = get_pair_type(group1, group2)

        n0chn[g1][g2] = len(g1_indices) + len(g2_indices)  # Total #res in group1+2
        d0chn[g1][g2] = calc_d0(n0chn[g1][g2], pair_type)

        # Precompute PTM matrix for this d0
        ptm_matrix_d0chn = ptm_func_vec(
            pae_matrix[np.ix_(g1_indices, g2_indices)], d0chn[g1][g2]
        )

        # ipTM uses all of group 2, ipSAE uses PAE cutoff
        iptm_d0chn_byres[g1][g2][g1_indices] = ptm_matrix_d0chn.mean(axis=1)

        valid_pairs_mask = pae_matrix[np.ix_(g1_indices, g2_indices)] < pae_cutoff
        ipsae_d0chn_byres[g1][g2][g1_indices] = np.ma.masked_where(
            ~valid_pairs_mask, ptm_matrix_d0chn
        ).mean(axis=1)

        # n0res and d0res by residue
        n0res_byres[g1][g2][g1_indices] = valid_pairs_mask.sum(axis=1)
        d0res_byres[g1][g2][g1_indices] = calc_d0_array(
            n0res_byres[g1][g2][g1_indices], pair_type
        )

        # Track unique residues contributing to the ipSAE for g1, g2
        g1_contrib_residues = set(
            residues[g1_indices[i]].resnum
            for i in np.where(valid_pairs_mask.any(axis=1))[0]
        )
        unique_residues_chain1[g1][g2].update(g1_contrib_residues)

        g2_contrib_residues = set(
            residues[g2_indices[j]].resnum
            for j in np.where(valid_pairs_mask.any(axis=0))[0]
        )
        unique_residues_chain2[g1][g2].update(g2_contrib_residues)

        # Track unique residues contributing to ipTM in interface
        valid_dist_mask = (valid_pairs_mask) & (
            distances[np.ix_(g1_indices, g2_indices)] < dist_cutoff
        )
        g1_dist_contrib_residues = set(
            residues[g1_indices[i]].resnum
            for i in np.where(valid_dist_mask.any(axis=1))[0]
        )
        dist_unique_residues_chain1[g1][g2].update(g1_dist_contrib_residues)

        g2_dist_contrib_residues = set(
            residues[g2_indices[j]].resnum
            for j in np.where(valid_dist_mask.any(axis=0))[0]
        )
        dist_unique_residues_chain2[g1][g2].update(g2_dist_contrib_residues)

    # Second pass: d0dom and d0res
    by_res_lines: list[PerResScoreResults] = []
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            continue

        pair_type = get_pair_type(group1, group2)

        n0dom[g1][g2] = len(unique_residues_chain1[g1][g2]) + len(
            unique_residues_chain2[g1][g2]
        )
        d0dom[g1][g2] = calc_d0(n0dom[g1][g2], pair_type)

        valid_pairs_mask = pae_matrix[np.ix_(g1_indices, g2_indices)] < pae_cutoff
        ptm_matrix_d0dom = ptm_func_vec(
            pae_matrix[np.ix_(g1_indices, g2_indices)], d0dom[g1][g2]
        )
        ipsae_d0dom_byres[g1][g2][g1_indices] = np.ma.masked_where(
            ~valid_pairs_mask, ptm_matrix_d0dom
        ).mean(axis=1)

        ptm_matrix_d0res = ptm_func_vec(
            pae_matrix[np.ix_(g1_indices, g2_indices)],
            d0res_byres[g1][g2][g1_indices][:, np.newaxis],
        )
        ipsae_d0res_byres[g1][g2][g1_indices] = np.ma.masked_where(
            ~valid_pairs_mask, ptm_matrix_d0res
        ).mean(axis=1)

        # Output line generation
        for i in g1_indices:
            by_res_lines.append(
                PerResScoreResults(
                    i=int(i + 1),
                    AlignChn=str(residues[i].chainid),
                    ScoredChn=str(g2),
                    AlignResNum=residues[i].resnum,
                    AlignResType=residues[i].res,
                    AlignRespLDDT=float(plddt[i]),
                    n0chn=int(n0chn[g1][g2]),
                    n0dom=int(n0dom[g1][g2]),
                    n0res=int(n0res_byres[g1][g2][i]),
                    d0chn=d0chn[g1][g2],
                    d0dom=d0dom[g1][g2],
                    d0res=float(d0res_byres[g1][g2][i]),
                    pTM_pae=float(iptm_d0chn_byres[g1][g2][i]),
                    pSAE_d0chn=float(ipsae_d0chn_byres[g1][g2][i]),
                    pSAE_d0dom=float(ipsae_d0dom_byres[g1][g2][i]),
                    pSAE=float(ipsae_d0res_byres[g1][g2][i]),
                )
            )

    # Aggregate results (Asym and Max)
    # We need to store these to generate the summary table

    # Store results in a structured way
    chain_pair_scores, pymol_lines, results_metrics = aggregate_byres_scores(
        residues,
        pae_cutoff,
        dist_cutoff,
        chain_pairs,
        ipsae_d0res_byres,
        ipsae_d0chn_byres,
        ipsae_d0dom_byres,
        iptm_d0chn_byres,
        n0res_byres,
        d0res_byres,
        unique_residues_chain1,
        unique_residues_chain2,
        dist_unique_residues_chain1,
        dist_unique_residues_chain2,
        pae_data,
        pDockQ,
        pDockQ2,
        LIS,
        n0chn,
        n0dom,
        d0chn,
        d0dom,
        label,
    )

    # Add ligand pymol lines
    pymol_lines.extend(ligand_pymol_lines)

    return ScoreResults(
        ipsae_scores=ipsae_d0res_byres,
        iptm_scores=iptm_d0chn_byres,
        pdockq_scores=pDockQ,
        pdockq2_scores=pDockQ2,
        lis_scores=LIS,
        metrics=results_metrics,
        by_res_scores=by_res_lines,
        chain_pair_scores=chain_pair_scores,
        ligand_scores=ligand_scores,
        pymol_script=pymol_lines,
    )
