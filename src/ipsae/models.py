"""Dataclasses and enums for ipSAE."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class InputModelType(Enum):
    """Supported model types for ipSAE calculation."""

    AF2 = "af2"
    AF3 = "af3"
    Boltz1 = "boltz1"
    Boltz2 = "boltz2"
    Chai1 = "chai-1"

    def __str__(self):
        """Return the string representation of the InputModelType."""
        return self.value

    @classmethod
    def from_string(cls, model_type_str: str) -> "InputModelType":
        """Create an InputModelType from a string with better error messages.

        Args:
            model_type_str: String representation of the model type.

        Returns:
            Corresponding InputModelType enum member.

        Raises:
            ValueError: If the string does not correspond to a valid model type.

        """
        valid_model_types = {member.value for member in cls}
        model_type_str_lower = model_type_str.lower()
        if model_type_str_lower not in valid_model_types:
            raise ValueError(
                f"Invalid model type specified: '{model_type_str}'. "
                f"Valid options are: {valid_model_types}."
            )
        return cls(model_type_str_lower)


@dataclass
class Residue:
    """Represents a residue with its coordinates and metadata.

    Attributes
    ----------
        atom_num: Atom serial number of the representative atom (CA or CB).
        coor: Numpy array of coordinates [x, y, z].
        res: Residue name (3-letter code).
        chainid: Chain identifier.
        resnum: Residue sequence number.
        residue_str: Formatted string representation for output.

    """

    atom_num: int
    coor: np.ndarray
    res: str
    chainid: str
    resnum: int
    residue_str: str


@dataclass
class LigandAtom:
    """Represents a ligand atom with its coordinates and metadata.

    Attributes
    ----------
        atom_num: Atom serial number.
        coor: Numpy array of coordinates [x, y, z].
        res: Residue name (3-letter code).
        chainid: Chain identifier.
        resnum: Residue sequence number.
        atom_name: Atom name (e.g., 'C1').
        element: Element symbol.
        global_pae_idx: Index in the full PAE matrix.

    """

    atom_num: int
    coor: np.ndarray
    res: str
    chainid: str
    resnum: int
    atom_name: str
    element: str
    global_pae_idx: int


@dataclass
class StructureData:
    """Container for parsed structure data.

    Attributes
    ----------
        residues: List of Residue objects (CA atoms).
            Used as the "master list" of residues for metadata (Residue Number, Chain ID, Residue Name).
        cb_residues: List of Residue objects (CB atoms, or CA for Glycine).
            Used for calculating distances, as sidechain (CB) distance is often more relevant for contacts.
        chains: Array of chain identifiers for each residue.
            Used for indexing to quickly find all residues in a specific chain.
        unique_chains: Array of unique chain identifiers.
            Used for iterating over every possible pair of chains.
        token_mask: Array indicating valid residues (1) vs ligands/others (0).
            Used to filter the PAE matrix, ensuring scores are only calculated for protein/DNA chains.
        residue_types: Array of residue names.
            Used for classifying chains (e.g., checking for nucleotides) to decide if it's protein or DNA/RNA.
        coordinates: Array of coordinates (CB atoms).
            The raw X, Y, Z positions used to calculate the distances matrix.
        distances: Pairwise distance matrix between residues.
            Critical geometric field used to decide if two residues are in "contact" (usually < 10 Angstroms).
        chain_pair_type: Dictionary mapping chain pairs to type ('protein' or 'nucleic_acid').
            Used for choosing the right normalization formula (d0) for scores.
        numres: Total number of residues.
            Used for sanity checks and initializing empty arrays.
        ligand_atoms: List of LigandAtom objects.
            Used for calculating ligand binding metrics.
        protein_atoms: List of LigandAtom objects (all atoms of protein/NA).
            Used for calculating ligand-protein contacts.

    """

    residues: list[Residue]  # [n_res]
    cb_residues: list[Residue]  # [n_res]
    chains: np.ndarray  # [n_res,]
    unique_chains: np.ndarray  # [n_chains,]
    token_mask: np.ndarray  # [n_res,]
    residue_types: np.ndarray  # [n_res,]
    coordinates: np.ndarray  # [n_res, 3]
    distances: np.ndarray  # [n_res, n_res]
    chain_pair_type: dict[str, dict[str, str]]
    numres: int
    ligand_atoms: list[LigandAtom]
    protein_atoms: list[LigandAtom]


@dataclass
class PAEData:
    """Container for PAE and confidence data.

    Attributes
    ----------
        pae_matrix: Predicted Aligned Error matrix (filtered for protein/NA).
        full_pae_matrix: Full Predicted Aligned Error matrix (including ligands).
        atom_plddts: Array of pLDDT scores for all tokens.
        plddt: Array of pLDDT scores (CA atoms).
        cb_plddt: Array of pLDDT scores (CB atoms).
        iptm_dict: Dictionary of ipTM scores for chain pairs.
        ptm: Global PTM score (if available).
        iptm: Global ipTM score (if available).

    """

    pae_matrix: np.ndarray
    full_pae_matrix: np.ndarray
    atom_plddts: np.ndarray
    plddt: np.ndarray
    cb_plddt: np.ndarray
    iptm_dict: dict[str, dict[str, float]]
    ptm: float = -1.0
    iptm: float = -1.0


@dataclass
class PerResScoreResults:
    """Container for per-residue score results.

    Attributes:
        i: residue in model (from 1 to total number of residues in model)
        AlignChn: chainid of aligned residue (in PAE calculation)
        ScoredChn: chainid of scored residues (with PAE less than cutoff)
        AlignResNum: residue number of aligned residue
        AlignResType: residue type of aligned residue (three letter code)
        AlignRespLDDT: plDDT of aligned residue
        n0chn: number of residues in d0 in ipSAE_d0chn calculation
        n0dom: number of residues in d0 in ipSAE_d0dom calculation
        n0res: number of residues for d0 in ipSAE calculation
        d0chn: d0 for ipSAE_d0chn
        d0dom: d0 for ipSAE_d0dom
        d0res: d0 for ipSAE
        pTM_pae: ipTM calculated from PAE matrix and d0 = sum of chain lengths
        pSAE_d0chn: residue-specific ipSAE calculated with PAE cutoff and d0 = sum of chain lengths (n0chn)
        pSAE_d0dom: residue-specific ipSAE calculated with PAE cutoff and d0 = total number of residues in both chains with any interchain PAE<cutoff (n0dom)
        pSAE: residue-specific ipSAE value for given PAE cutoff and d0 determined by number of residues in 2nd chain with PAE<cutoff (n0res)

    """

    i: int
    AlignChn: str
    ScoredChn: str
    AlignResNum: int
    AlignResType: str
    AlignRespLDDT: float
    n0chn: int
    n0dom: int
    n0res: int
    d0chn: float
    d0dom: float
    d0res: float
    pTM_pae: float
    pSAE_d0chn: float
    pSAE_d0dom: float
    pSAE: float

    def to_formatted_line(self, end: str = "") -> str:
        """Format the per-residue score results as a fixed-width string."""
        c1, c2 = self.AlignChn, self.ScoredChn
        return (
            f"{self.i:<8d}"
            f"{c1:<10}"
            f"{c2:<10}"
            f"{self.AlignResNum:4d}           "
            f"{self.AlignResType:3}        "
            f"{self.AlignRespLDDT:8.2f}         "
            f"{self.n0chn:5d}  "
            f"{self.n0dom:5d}  "
            f"{self.n0res:5d}  "
            f"{self.d0chn:8.3f}  "
            f"{self.d0dom:8.3f}  "
            f"{self.d0res:8.3f}   "
            f"{self.pTM_pae:8.4f}    "
            f"{self.pSAE_d0chn:8.4f}    "
            f"{self.pSAE_d0dom:8.4f}    "
            f"{self.pSAE:8.4f}{end}"
        )

    @staticmethod
    def header_line() -> str:
        """Return the header line for the per-residue score output."""
        return "i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE\n"


@dataclass
class ChainPairScoreResults:
    """Container for chain-pair summary score results.

    Attributes:
        Chn1: first chain identifier
        Chn2: second chain identifier
        PAE: PAE cutoff value
        Dist: Distance cutoff for CA-CA contacts
        Type: "asym" or "max"; asym means asymmetric ipTM/ipSAE values; max is maximum of asym values
        ipSAE: ipSAE value for given PAE cutoff and d0 determined by number of residues in 2nd chain with PAE<cutoff
        ipSAE_d0chn: ipSAE calculated with PAE cutoff and d0 = sum of chain lengths
        ipSAE_d0dom: ipSAE calculated with PAE cutoff and d0 = total number of residues in both chains with any interchain PAE<cutoff
        ipTM_af: AlphaFold ipTM values. For AF2, this is for whole complex from json file. For AF3, this is symmetric pairwise value from summary json file.
        ipTM_d0chn: ipTM (no PAE cutoff) calculated from PAE matrix and d0 = sum of chain lengths
        pDockQ: score from pLDDTs from Bryant, Pozotti, and Eloffson
        pDockQ2: score based on PAE, calculated pairwise from Zhu, Shenoy, Kundrotas, Elofsson
        LIS: Local Interaction Score based on transform of PAEs from Kim, Hu, Comjean, Rodiger, Mohr, Perrimon
        n0res: number of residues for d0 in ipSAE calculation
        n0chn: number of residues in d0 in ipSAE_d0chn calculation
        n0dom: number of residues in d0 in ipSAE_d0dom calculation
        d0res: d0 for ipSAE
        d0chn: d0 for ipSAE_d0chn
        d0dom: d0 for ipSAE_d0dom
        nres1: number of residues in chain1 with PAE<cutoff with residues in chain2
        nres2: number of residues in chain2 with PAE<cutoff with residues in chain1
        dist1: number of residues in chain 1 with PAE<cutoff and dist<cutoff from chain2
        dist2: number of residues in chain 2 with PAE<cutoff and dist<cutoff from chain1
        Model: AlphaFold filename

    """

    Chn1: str
    Chn2: str
    PAE: float
    Dist: float
    Type: str
    ipSAE: float
    ipSAE_d0chn: float
    ipSAE_d0dom: float
    ipTM_af: float
    ipTM_d0chn: float
    pDockQ: float
    pDockQ2: float
    LIS: float
    n0res: int
    n0chn: int
    n0dom: int
    d0res: float
    d0chn: float
    d0dom: float
    nres1: int
    nres2: int
    dist1: int
    dist2: int
    Model: str

    def to_formatted_line(self, end: str = "") -> str:
        """Format the summary result as a fixed-width string."""
        pae_str = str(int(self.PAE)).zfill(2)
        dist_str = str(int(self.Dist)).zfill(2)

        return (
            f"{self.Chn1:<5}{self.Chn2:<5} {pae_str:3}  {dist_str:3}  {self.Type:5} "
            f"{self.ipSAE:8.6f}    "
            f"{self.ipSAE_d0chn:8.6f}    "
            f"{self.ipSAE_d0dom:8.6f}    "
            f"{self.ipTM_af:5.3f}    "
            f"{self.ipTM_d0chn:8.6f}    "
            f"{self.pDockQ:8.4f}   "
            f"{self.pDockQ2:8.4f}   "
            f"{self.LIS:8.4f}   "
            f"{self.n0res:5d}  "
            f"{self.n0chn:5d}  "
            f"{self.n0dom:5d}  "
            f"{self.d0res:6.2f}  "
            f"{self.d0chn:6.2f}  "
            f"{self.d0dom:6.2f}  "
            f"{self.nres1:5d}   "
            f"{self.nres2:5d}   "
            f"{self.dist1:5d}   "
            f"{self.dist2:5d}   "
            f"{self.Model}{end}"
        )

    @staticmethod
    def header_line() -> str:
        """Return the header line for the summary output."""
        return "Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"


@dataclass
class LigandScoreResults:
    """Container for ligand binding score results.

    Attributes:
        LigandChn: chain identifier of the ligand
        PAE: PAE cutoff value
        Dist: Distance cutoff for contacts
        ipTM: Average ipTM for the ligand chain
        pLDDT: Average pLDDT for the ligand atoms
        nligatoms: number of unique ligand atoms in contact
        nres: number of unique protein residues in contact
        npair: total number of atomic contacts
        Model: AlphaFold filename

    """

    LigandChn: str
    PAE: float
    Dist: float
    ipTM: float
    pLDDT: float
    nligatoms: int
    nres: int
    npair: int
    Model: str

    def to_formatted_line(self, end: str = "") -> str:
        """Format the ligand result as a fixed-width string."""
        return (
            f"{self.LigandChn:<10} "
            f"{self.PAE:4.1f}  "
            f"{self.Dist:4.1f}  "
            f"{self.ipTM:8.4f}  "
            f"{self.pLDDT:8.2f}  "
            f"{self.nligatoms:8d}  "
            f"{self.nres:8d}  "
            f"{self.npair:8d}  "
            f"{self.Model}{end}"
        )

    @staticmethod
    def header_line() -> str:
        """Return the header line for the ligand summary output."""
        return "LigandChn   PAE   Dist     ipTM      pLDDT    nligatoms     nres    npair  Model\n"


@dataclass
class ScoreResults:
    """Container for calculated scores and output data.

    Attributes
    ----------
        ipsae_scores: Dictionary of ipSAE scores (by residue).
        iptm_scores: Dictionary of ipTM scores (by residue).
        pdockq_scores: Dictionary of pDockQ scores (by chain pair).
        pdockq2_scores: Dictionary of pDockQ2 scores (by chain pair).
        lis_scores: Dictionary of LIS scores (by chain pair).
        metrics: Dictionary of pDockQ, pDockQ2, and LIS scores for each chain pair.
        by_res_scores: Lists of per-residue scores.
        chain_pair_scores: List of chain-pair summary score results.
        ligand_scores: List of ligand binding score results.
        pymol_script: List of formatted strings for PyMOL script.

    """

    ipsae_scores: dict[str, dict[str, np.ndarray]]  # {c1: {c2: np.ndarray}}
    iptm_scores: dict[str, dict[str, np.ndarray]]  # {c1: {c2: np.ndarray}}
    pdockq_scores: dict[str, dict[str, float]]  # {c1: {c2: score}}
    pdockq2_scores: dict[str, dict[str, float]]  # {c1: {c2: score}}
    lis_scores: dict[str, dict[str, float]]  # {c1: {c2: score}}
    metrics: dict[str, dict[str, float]]  # {`<c1>_<c2>`: {metric_name: value}}

    by_res_scores: list[PerResScoreResults]
    chain_pair_scores: list[ChainPairScoreResults]
    ligand_scores: list[LigandScoreResults]
    pymol_script: list[str]
