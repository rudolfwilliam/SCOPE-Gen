"""Script that contains functions to generate scaffolds from molecular data."""

from pcgen.molecules.data.conversions import tuple_to_mol, smiles_to_mol

import torch

def remove_atom(mol):
    from random import choice
    from pcgen.molecules.data.conversions import mol_to_tuple, clean_and_convert_samples
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    # convert to correct instance
    if isinstance(mol, tuple):
        mol = tuple_to_mol(mol)
    if isinstance(mol, str):
        mol = smiles_to_mol(mol)
    atom_count = mol.GetNumAtoms()
    # convert to graph representation
    mol_tuple = mol_to_tuple(mol)
    # remove one arbitrary atom from the molecule that results in a valid molecule
    valid = False
    excl = []
    while not valid:
        idxs = [i for i in range(0, atom_count) if i not in excl]
        if len(idxs) == 0:
            break
        idx = choice(idxs)
        # Remove the ith row
        X_mod = torch.cat((mol_tuple[0][:, :idx], mol_tuple[0][:, idx+1:]), dim=1)
        E_mod = torch.cat((mol_tuple[1][:, :idx, :], mol_tuple[1][:, idx+1:, :]), dim=1)
        # Remove the ith column
        E_mod = torch.cat((E_mod[:, :, :idx], E_mod[:, :, idx+1:]), dim=2)
        mol_tuple_cropped = (X_mod[0, ...], E_mod[0, ...])
        mol_cropped = clean_and_convert_samples([mol_tuple_cropped])
        if len(mol_cropped) == 0:
            excl.append(idx)
            continue
        else:
            valid = True
    return mol_cropped if valid else None

def extract_murcko_scaffold(mol):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import SanitizeFlags, SanitizeMol
    if isinstance(mol, tuple):
        mol = tuple_to_mol(mol)
    if isinstance(mol, str):
        mol = smiles_to_mol(mol)
    SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return scaffold
