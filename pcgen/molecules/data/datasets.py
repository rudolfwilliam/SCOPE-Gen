from pcgen.data.datasets import CountDataset, ConditionalDataset
from pcgen.molecules.data.fragment_gen import remove_atom
from pcgen.molecules.data.conversions import is_valid_mol
from pcgen.molecules.paths import DATA_DIR
from pcgen.molecules.data.conversions import smiles_to_mol, smiles_to_tuple
from pcgen.utils import override
from torch.utils.data import Dataset
import os
import pickle


def switch_representation(smiles, repr):
    if repr == "smiles":
        return smiles
    elif repr == "mol":
        return smiles_to_mol(smiles)
    elif repr == "tuple":
        return smiles_to_tuple(smiles)
    else:
        raise ValueError(f"Invalid representation: {repr}")


class MosesDataset(Dataset):
    def __init__(self, z=None, repr="smiles"):
        if z is None:
            with open(os.path.join(DATA_DIR, 'moses_test/moses_data.pkl'), 'rb') as file:
                z = pickle.load(file)
        else:
            assert all(isinstance(s, str) and is_valid_mol(s) for s in z)
        assert repr in ["smiles", "mol", "tuple"]
        self.smiles = z
        self.repr = repr
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return switch_representation(self.smiles[idx], self.repr)


class MosesScaffoldDataset(ConditionalDataset):
    def __init__(self, z, cond=None, repr="tuple"):
        if cond is None:
            print("No scaffolds provided, generating them via single-atom removal.")
            scaffolds = MosesScaffoldDataset._prepare_scaffolds(z)
            # add required atom count
            cond = [(scaffold, smiles_to_mol(instance).GetNumAtoms()) for scaffold, instance in zip(scaffolds, z)]
        assert repr in ["smiles", "mol", "tuple"]
        assert all(isinstance(s, str) and is_valid_mol(s) for s in z)
        # discard examples where valid scaffolds could not be generated (they are None)
        smiles = [s for s, c in zip(z, cond) if c is not None]
        cond = [c for c in cond if c is not None]
        self.repr = repr
        super().__init__(smiles, cond)

    @staticmethod
    def _prepare_scaffolds(z):
        scaffolds = []
        for i, mol in enumerate(z):
            scaffolds.append(remove_atom(mol))
            if i % 1000 == 0:
                print(f"Done with {i/len(z)} of all scaffold generations.")
        return scaffolds

    @override
    def __getitem__(self, idx):
        return switch_representation(self.z[idx], self.repr), (switch_representation(self.cond[idx][0], self.repr), self.cond[idx][1])


class MolCountDataset(CountDataset):
    def __init__(self, y, cond=None):
        super().__init__(y, cond)
        assert all(isinstance(l, int) for l in y)
        assert all(is_valid_mol(c) for c in cond)
    
    def __getitem__(self, idx):
        return self.y[idx], self.cond[idx]
    