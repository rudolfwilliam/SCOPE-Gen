"""Reformat the data to fit the convention used here."""

import numpy as np
import pickle
import os

def format_data(data_dir, output_dir, binarize_labels=False, label_threshold=0.35):
    losses = np.load(os.path.join(data_dir, "outputs", "losses.npy"), allow_pickle=True)
    scores = np.load(os.path.join(data_dir, "outputs", "scores.npy"), allow_pickle=True)
    similarities = np.load(os.path.join(data_dir, "outputs", "diversity.npy"), allow_pickle=True)

    formatted_data = []
    for losses_, scores_, similarities_ in zip(losses, scores, similarities):
        instance = {
            "labels": (1 - losses_) if not binarize_labels else ((1 - losses_) > label_threshold).astype(int),
            "scores": scores_,
            "similarities": similarities_
        }
        formatted_data.append(instance)

    output_dir = os.path.join(data_dir, output_dir)
    with open(output_dir, 'wb') as file:
        pickle.dump(formatted_data, file)

def labels_to_losses(data_dir):
    labels = np.load(os.path.join(data_dir, "outputs", "labels.npy"), allow_pickle=True)
    losses = 1 - labels
    output_dir = os.path.join(data_dir, "outputs", "losses.npy")
    np.save(output_dir, losses)

def invert(scores):
    return 1/(1 + 1e10*scores)

if __name__ == '__main__':
    format_data(data_dir = "outputs", output_dir = "processed/data.pkl")
