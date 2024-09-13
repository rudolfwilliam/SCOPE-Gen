"""This script removes duplicate answers."""

import numpy as np
import pickle 
import pickle
import os


def main(formatted_data_dir, output_dir, data_dir):
    full_path = os.path.join(data_dir, formatted_data_dir)
    with open(os.path.join(full_path, "data.pkl"), 'rb') as file:
        data = pickle.load(file)

    new_data = []

    for instance in data:
        new_instance = remove_dupl(instance)
        new_data.append(new_instance)

    return new_data

def remove_dupl(instance):
    labels = instance["labels"]
    similarities = instance["similarities"]
    if len(labels) == 0:
        return instance
    
    # Find indices of unique answers based on similarities
    unique_indices = []
    seen = set()
    
    for i in range(len(labels)):
        if i not in seen:
            unique_indices.append(i)
            # Mark similar answers to be removed from consideration
            for j in range(i + 1, len(labels)):
                if similarities[i, j] == 1:
                    seen.add(j)
    
    # Create new instance with filtered data
    new_instance = {
        "labels": labels[unique_indices],
        "scores": instance["scores"][unique_indices],
        "similarities": similarities[np.ix_(unique_indices, unique_indices)]
    }
    return new_instance


if __name__ == '__main__':
    main(formatted_data_dir = "processed", output_dir = "no_duplicates")
