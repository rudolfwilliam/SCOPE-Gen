from pcgen.toy_example.data.distributions import ThreeGaussians
from pcgen.toy_example.calibrate.compute_scores import compute_count_scores as compute_scores_func
from pcgen.toy_example.paths import CKPT_DIR, DATA_DIR
from pcgen.toy_example.models import ToyExampleLogisticModel
from pcgen.calibrate.base import compute_percentile, generate_prediction_set
from pcgen.scores.normalized_densities import normalized_densities_score as score
import torch

def main(alpha, scores_file_name, model_file_name, num_samples, sample_size, compute_scores=False):
    if compute_scores:
        # if scores have not been computed yet, compute them
        compute_scores_func(model_file_name, 'calibration_data', scores_file_name)
    scores = torch.load(DATA_DIR + '/' + scores_file_name + '.pt')
    # compute percentile alpha of scores
    perc = compute_percentile(alpha, scores)
    # choose n such that the score is 
    dist = ThreeGaussians()
    model = ToyExampleLogisticModel()
    model.load_state_dict(torch.load(CKPT_DIR + '/' + model_file_name + '.ckpt')["state_dict"])
    coverage = 0
    for _ in range(num_samples):
        scores = []
        model_sample = model.sample(sample_size)[0].detach()
        idxs = generate_prediction_set(score, model, model_sample, perc)
        # generate samples from the model
        true_example = dist.sample(num_samples=1).detach()
        # get index of sample that is closest to the true sample
        idx = torch.argmin(torch.abs(model_sample - true_example))
        mean_dist += torch.abs(model_sample - true_example).min()
        if idx in idxs:
            coverage += 1
        mean_pred_set_size += len(idxs)
    coverage = coverage / num_samples
    print(f"Estimated Mean Prediction Set Size: {mean_pred_set_size / num_samples}")
    print(f"Estimated Coverage: {coverage}")
    print(f"Estimated Mean Distance: {mean_dist / num_samples}")
        

if __name__ == '__main__':
    main(alpha=0.35, scores_file_name='scores', model_file_name='logistic-v1', num_samples=10000, sample_size=30, compute_scores=False)
    