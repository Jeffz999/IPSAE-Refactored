import json
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from scipy.spatial import KDTree
import sys
import glob
import os

def get_af3_scores(cif_path, ligand_chain='D'):
    base_name = os.path.basename(cif_path).replace('_model.cif', '')
    summary_path = cif_path.replace('_model.cif', '_summary_confidences.json')
    pae_path = cif_path.replace('_model.cif', '_confidences.json')

    # --- 1. Get ipTM (Replicating ChimeraX logic) ---
    # ChimeraX logic: cp_iptm[-1][:-1] 
    # (Assumes Ligand is the LAST chain in the summary matrix)
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        cp_iptm = summary['chain_pair_iptm']
        # Get row for last chain, all columns except self
        ligand_row = cp_iptm[-1][:-1] 
        avg_iptm = np.mean(ligand_row)
    except Exception:
        avg_iptm = 0.0

    # --- 2. Load PAE Matrix ---
    try:
        with open(pae_path) as f:
            pae_data = json.load(f)
        pae_matrix = np.array(pae_data['pae'])
    except Exception:
        print(f"Skipping {base_name}: Missing PAE file.")
        return None

    # --- 3. Parse Structure ---
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("model", str(cif_path))
        if structure is None:
            raise ValueError(f"Failed to parse structure file {cif_path}")
    except Exception as e:
        print(f"Failed to parse structure file {cif_path}: {e}")
        raise
    
    model = structure[0]
    
    # Map residues to PAE indices
    all_atoms = []
    global_res_idx = 0
    
    # AF3 PAE order corresponds to residue order in CIF
    for chain in model:
        for res in chain:
            for atom in res:
                all_atoms.append({
                    'atom_obj': atom,
                    'coord': atom.get_coord(),
                    'plddt': atom.get_bfactor(),
                    'global_pae_idx': global_res_idx,
                    'chain_id': chain.id,
                    'res_id': f"{chain.id}_{res.id[1]}" # Unique ID for protein residues
                })
            global_res_idx += 1

    ligand_atoms = [a for a in all_atoms if a['chain_id'] == ligand_chain]
    protein_atoms = [a for a in all_atoms if a['chain_id'] != ligand_chain]

    if not ligand_atoms:
        return None

    # --- 4. Calculate Metrics ---
    # Metric A: Average Ligand pLDDT
    ave_plddt = np.mean([a['plddt'] for a in ligand_atoms])

    # Metric B: Contacts (Dist < 4A AND PAE < 3)
    # Build tree for protein
    prot_coords = [a['coord'] for a in protein_atoms]
    tree = KDTree(prot_coords)
    
    # Query ligand against protein
    lig_coords = [a['coord'] for a in ligand_atoms]
    neighbors_list = tree.query_ball_point(lig_coords, r=4.0)

    # Counters matching ChimeraX output
    passed_contacts = 0          # matches 'npair'
    contact_lig_atom_indices = set() # matches 'nligatoms'
    contact_prot_res_ids = set()     # matches 'nres'

    for i, neighbors in enumerate(neighbors_list):
        if not neighbors: continue
        
        l_data = ligand_atoms[i]
        l_idx = l_data['global_pae_idx']
        
        for p_idx_in_list in neighbors:
            p_data = protein_atoms[p_idx_in_list]
            p_idx = p_data['global_pae_idx']

            # CHECK PAE < 3.0 (Matching ChimeraX code "maxPae 3")
            if pae_matrix[l_idx, p_idx] < 3.0:
                passed_contacts += 1
                contact_lig_atom_indices.add(i)
                contact_prot_res_ids.add(p_data['res_id'])

    return {
        'name': base_name,
        'iptm': avg_iptm,
        'plddt': ave_plddt,
        'nligatoms': len(contact_lig_atom_indices), # Unique ligand atoms in contact
        'nres': len(contact_prot_res_ids),           # Unique protein residues in contact
        'npair': passed_contacts                     # Total atomic contacts
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    # Handle command line args like the ChimeraX script
    if len(sys.argv) == 1:
        cif_files = [f for f in os.listdir() if f.endswith('_model.cif')]
    else:
        cif_files = []
        for path in sys.argv[1:]:
            cif_files.extend(glob.glob(path))
    
    cif_files.sort()

    print(f"{'Name':<30} {'ipTM':<6} {'pLDDT':<6} {'nlig':<5} {'nres':<5} {'npair':<5}")
    
    for cif_file in cif_files:
        res = get_af3_scores(cif_file)
        if res:
            print(f"{res['name']:<30} {res['iptm']:.2f}   {res['plddt']:.0f}    {res['nligatoms']:<5} {res['nres']:<5} {res['npair']:<5}")