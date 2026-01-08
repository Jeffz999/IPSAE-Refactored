"""Self-contained script for calculating the ipSAE score.

This module now serves as a backward-compatible wrapper around the modularized
ipsae package.
"""

from ipsae.main import ipsae, main
from ipsae.models import (
    ChainPairScoreResults,
    InputModelType,
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

if __name__ == "__main__":
    main()
