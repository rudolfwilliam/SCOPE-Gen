from pcgen.molecular_extensions.utils import load_model_from_ckpt_path
from pcgen.molecular_extensions.paths import CKPT_DIR, OUTPUT_DIR
from pcgen.molecular_extensions.data.datasets import MosesDataset
from pcgen.molecular_extensions.data.conversions import mol_to_tuple, clean_and_convert_samples
from pcgen.molecular_extensions.data.fragment_gen import extract_murcko_scaffold
from pcgen.molecular_extensions.distances import TanimotoDistance

import numpy as np

def main(n=100, epsilon=0.15):
    #test = moses.get_dataset("test")
    model = load_model_from_ckpt_path(CKPT_DIR + '/checkpoint_moses.ckpt', map_location='cuda')
    model.eval()
    ns = []
    dists = []
    test = MosesDataset(repr="mol")
    for i in range(n):
        mol = test[i]
        atom_count = len(mol.GetAtoms())
        mol = mol_to_tuple(test[i])
        scaffold = extract_murcko_scaffold(mol)
        dist_fun = TanimotoDistance()
        samples = model.sample(num_samples=512, cond=(mol_to_tuple(scaffold), atom_count))
        samples = clean_and_convert_samples(samples)
        # check if the original molecule is in the epsilon neighborhood of the sampless
        n = 1
        for sample in samples:
            # compute the tanimoto similarity between the original molecule and the sample macskeys
            dist = dist_fun(mol, sample)
            if dist <= epsilon:
                break
            n += 1
        if n == (len(samples) + 1):
            print("Original molecule not in the epsilon neighborhood of the samples")
            ns.append(float("inf"))
            dists.append(float("inf"))
        else:
            ns.append(n)
            dists.append(dist)
        if i % 10 == 0:
            print(f"Sample {i}")
            np.save(OUTPUT_DIR + "/ns.npy", ns)
            np.save(OUTPUT_DIR + "/dists.npy", dists)
    
    # save the results
    np.save("ns.npy", ns)
    np.save("dists.npy", dists)
    print(f"Average number of samples to find a molecule in the epsilon neighborhood: {sum(ns)/len(ns)}")


if __name__ == "__main__":
    main(n=100, epsilon=0.3)
