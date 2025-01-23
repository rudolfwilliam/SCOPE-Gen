from rdkit.Chem.rdchem import BondType as BT

ATOM_DECODER = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
