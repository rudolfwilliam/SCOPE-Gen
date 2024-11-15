"""This script generates the data for the molecules dataset."""

from pcgen.molecules.paths import CKPT_DIR
from pcgen.molecules.utils import load_model_from_ckpt_path
from pcgen.molecules.data.datasets import MosesDataset
from pcgen.molecules.data.fragment_gen import remove_atom
from pcgen.molecules.data.conversions import mol_to_tuple, tuple_to_mol
from pcgen.molecules.data.conversions import clean_and_convert_samples
from pcgen.molecules.paths import DATA_DIR
from rdkit import Chem
from rdkit.Chem import QED
from pcgen.utils import set_seed
import torch
import numpy as np
import os

NUM_SAMPLES = 60
VERBOSE = True
CALIBRATION_SET_SIZE = 5000

def main():
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    
    # use GPU is cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_ckpt_path(CKPT_DIR + '/checkpoint_moses.ckpt', map_location=device)
    # load moses test set
    dataset = MosesDataset(repr='mol')
    # generate new data
    results = {
        'labels' : [],
        'scores' : [],
        'similarities' : []
    }
    if VERBOSE:
        print(f'Generating data for {CALIBRATION_SET_SIZE} examples')
    for i in range(len(dataset)):
        example = dataset[i]
        # generate a scaffold
        scaffold = remove_atom(dataset[i])
        if scaffold is None:
            continue
        # generate extensions
        samples = model.sample(num_samples=NUM_SAMPLES, cond=(mol_to_tuple(scaffold), example.GetNumAtoms()))
        samples = [tuple_to_mol(sample) for sample in samples]
        instance = {'labels' : [], 'scores' : [], 'similarities' : [], 'valids' : []}
        # all similarities are 0
        instance['similarities'] = np.zeros(shape=(NUM_SAMPLES, NUM_SAMPLES))
        instance['labels'] = np.zeros(NUM_SAMPLES)
        instance['scores'] = np.zeros(NUM_SAMPLES)
        instance['valids'] = [False] * NUM_SAMPLES
        for j, sample in enumerate(samples):
            # check if the sample is valid
            try:
                Chem.SanitizeMol(sample)
                score = QED.qed(sample)
                instance['scores'][j] = score
                # check if the sample is the same as the example
                if Chem.MolToSmiles(sample, canonical=True) == Chem.MolToSmiles(example, canonical=True):
                    instance['labels'][j] = 1
                instance['valids'][j] = True
            except Exception as e:
                pass
            if instance['valids'][j]:
                # check if the sample is a duplicate
                for k in range(j + 1):
                    if not instance['valids'][k]:
                        continue
                    try:
                        smiles_k = Chem.MolToSmiles(samples[k], canonical=True)
                    except Exception as e:
                        continue  # Skip if smiles_k can't be obtained
                    if Chem.MolToSmiles(sample, canonical=True) == smiles_k:
                        instance['similarities'][j, k] = 1.
        # mirror the similarity matrix
        instance['similarities'] = instance['similarities'].T
        results['labels'].append(instance['labels'])
        results['scores'].append(instance['scores'])
        results['similarities'].append(instance['similarities'])
        # save data
        if i % 10 == 0:
            if VERBOSE:
                print(f'Processed {i} examples')
            np.save(os.path.join(DATA_DIR, 'outputs', 'labels.npy'), results['labels'])
            np.save(os.path.join(DATA_DIR, 'outputs', 'scores.npy'), results['scores'])
            np.save(os.path.join(DATA_DIR, 'outputs', 'diversity.npy'), results['similarities'])


if __name__ == "__main__":
    set_seed(0)
    main()
