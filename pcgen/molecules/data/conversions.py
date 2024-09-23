"""Convert from rdkit molecule to (X, E) data tuple and vice versa."""

from pcgen.molecules.data.meta_data import TYPES, BONDS, ATOM_DECODER
from pcgen.molecules.data.base import squeeze
from torch.functional import F
from rdkit import Chem
from rdkit.Chem import SanitizeMol, SanitizeFlags
from torch_geometric.data import Data

import torch
import src.utils

def is_valid_mol(mol):
    """Check if a given RDKit molecule object is valid."""
    # convert to molecule if not already
    if isinstance(mol, tuple):
        mol = tuple_to_mol(mol)
    if isinstance(mol, str):
        mol = smiles_to_mol(mol)
    if mol is None:
        return False
    try:
        # Attempt to sanitize the molecule without kekulization
        SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
        
        # Optionally, try to kekulize separately to identify issues
        try:
            Chem.Kekulize(mol)
        except Chem.KekulizeException:
            return False

        # Additional checks
        if mol.GetNumAtoms() == 0:
            return False
        
        # Check for disconnected components
        if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
            return False

        return True
    except (ValueError, Chem.rdchem.KekulizeException) as e:
        return False

def mol_to_tuple(mol):
    # if mol is a list with only one element, take the element
    if isinstance(mol, list) and len(mol) == 1:
        mol = mol[0]
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(TYPES[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BONDS[bond.GetBondType()] + 1]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(BONDS) + 1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(TYPES)).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=0)

    # Try to build the molecule again from the graph. If it fails, do not add it to the training set
    dense_data, node_mask = src.utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    dense_data = dense_data.mask(node_mask, collapse=True)
    X, E = dense_data.X, dense_data.E

    return X, E

def tuple_to_mol(mol):
        """
        Convert tuple (X, E) to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        mol = squeeze(mol)
        node_list, adjacency_matrix = mol
        # dictionary to map integer value to the char of atom
        atom_decoder = ATOM_DECODER

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            mol = None
        return mol

def mol_to_smiles(mol):
    """Convert rdkit molecule to SMILES string"""
    return Chem.MolToSmiles(mol)

def smiles_to_mol(smiles):
    """Convert SMILES string to rdkit molecule"""
    return Chem.MolFromSmiles(smiles)

def smiles_to_tuple(smiles):
    """Convert SMILES string to (X, E) data tuple"""
    mol = smiles_to_mol(smiles)
    return mol_to_tuple(mol)

def clean_and_convert_samples(samples):
    """Sanitize and remove duplicates from a list of mol samples. Then convert to mol objects"""
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    samples_mol = [tuple_to_mol(sample) for sample in samples]
    # sanitize molecules
    # Sanitize molecules and collect valid ones
    valid_samples_mol = []
    for sample in samples_mol:
        try:
            Chem.SanitizeMol(sample)
            valid_samples_mol.append(sample)
        except Exception as e:
            continue
    # remove molecules that consist of multiple disconnected components
    valid_samples_mol = [mol for mol in valid_samples_mol if len(Chem.GetMolFrags(mol, asMols=True)) == 1]
    # remove duplicates
    unique_mol_dict = {}
    for mol in valid_samples_mol:
        smiles = Chem.MolToSmiles(mol, canonical=True)  # Generate canonical SMILES
        if smiles not in unique_mol_dict:
            unique_mol_dict[smiles] = mol

    # Get the list of unique molecules
    unique_samples_mol = list(unique_mol_dict.values())
    return unique_samples_mol
