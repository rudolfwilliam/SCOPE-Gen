from pcgen.toy_example.data.distributions import ThreeGaussians
from pcgen.toy_example.paths import DATA_DIR
from pcgen.utils import set_seed
from pcgen.admissions import ProximalAdmission
import numpy as np


def generate_data(gen_model, data_set_size, min_prob_threshold=1e-6, epsilon=0.1, distance_func=None, first_adm_only=False, k_max=20):
    data_candidates = gen_model.sample(data_set_size*k_max)
    scores = gen_model.predictor(data_candidates)
    data_candidates = data_candidates.reshape(data_set_size, k_max, -1)
    scores = scores.reshape(data_set_size, k_max)
    similarity_matrices = np.stack([1/(1 + distance_func.distance_matrix(data_candidate)) for data_candidate in data_candidates])
    admission_funcs = [ProximalAdmission(epsilon=epsilon, distance_func=distance_func) for _ in range(data_set_size)]

    # sample actual data
    data_gt = generate_raw_data(data_set_size, min_prob_threshold=min_prob_threshold)
    # remove condition
    data_gt = [data_gt[i][0] for i in range(data_set_size)]

    # set ground truth for all of them
    for j in range(data_set_size):
        admission_funcs[j].gt = data_gt[j]

    admissibles = np.array([[float(admission_funcs[j](data_candidates[j, i])) for i in range(k_max)] for j in range(data_set_size)])
    if first_adm_only:
        def process_row(row):
            idxs = np.where(row == 1)[0]
            if idxs.size > 0:
                first_one_idx = idxs[0]
                # Create a new row that sets all elements after the first 1 to zero
                new_row = np.concatenate((row[:first_one_idx + 1], np.zeros(len(row) - first_one_idx - 1)))
            else:
                # If there is no 1 in the row, return the row as is
                new_row = row
            return new_row
        admissibles = np.array([process_row(row) for row in admissibles])
    instances = [{"scores": scores_, "similarities": similarity_matrix, "labels": admissibles_} for scores_, similarity_matrix, admissibles_ in zip(scores, similarity_matrices, admissibles)]

    return instances

def generate_raw_data(n_samples, min_prob_threshold=1e-6, save=False, file_name="data_gen_model"):
    dist = ThreeGaussians()
    data = dist.sample(n_samples, min_prob_threshold=min_prob_threshold)
    # data is unconditional, so we need to add None as the condition
    data = [(x, None) for x in data]
    if save:
        # save data as a pt file
        np.save(data, DATA_DIR + '/' + file_name + '.npy')
    return data
