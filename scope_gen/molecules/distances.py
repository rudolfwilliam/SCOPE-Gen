from scope_gen.distances import Distance
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys, SanitizeMol, SanitizeFlags
from scope_gen.molecular_extensions.data.conversions import tuple_to_mol
from rdkit import Chem


class TanimotoDistance(Distance):
    def __init__(self, fingerprint="MACCSkeys"):
        self.fingerprint = fingerprint

    def __call__(self, x, y):
        # Convert x and y to molecular footprints if they aren't already
        if not isinstance(x, DataStructs.ExplicitBitVect):
            x = self._to_fingerprint(x)
        if not isinstance(y, DataStructs.ExplicitBitVect):
            y = self._to_fingerprint(y)

        return 1. - DataStructs.TanimotoSimilarity(x, y)
    
    def _to_fingerprint(self, x):
        # either x is molecule, SMILES string or (X, E) tuple
        assert isinstance(x, str) or isinstance(x, Chem.rdchem.Mol) or isinstance(x, tuple)

        if isinstance(x, str):
            # convert SMILES string to molecule
            x = Chem.MolFromSmiles(x)
        
        if isinstance(x, tuple):
            # convert (X, E) tuple to molecule
            x = tuple_to_mol(x)

        if isinstance(x, Chem.rdchem.Mol):
            # convert molecule to fingerprint
            SanitizeMol(x, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
            if self.fingerprint == "MACCSkeys":
                x = MACCSkeys.GenMACCSKeys(x)
            elif self.fingerprint == "Morgan":
                x = Chem.AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)
            else:
                raise ValueError(f"Invalid fingerprint type: {self.fingerprint}")
        
        return x
