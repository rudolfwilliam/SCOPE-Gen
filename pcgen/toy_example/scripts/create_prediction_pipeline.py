from pcgen.toy_example.data.datasets import ToyDataset
from pcgen.toy_example.models import GaussianMixtureModel
from pcgen.toy_example.paths import CONFIG_DIR, CKPT_DIR
from pcgen.toy_example.distances import L2Distance
from pcgen.nc_scores import SumScore
from pcgen.admissions import ProximalAdmission
from pcgen.algorithms.base import create_prediction_pipeline
from pcgen.calibrate.score_computation import ProximalCoverageScoreComputer
from pcgen.utils import set_seed, load_config_from_json
import torch
import pickle


def main(cfg):
    data = ToyDataset()
    gen_model = torch.load(CKPT_DIR + '/gmm.pt')
    distance_func = L2Distance()
    score_func = SumScore()
    admission_func = ProximalAdmission(epsilon=cfg["epsilon"], distance_func=distance_func)
    pipeline = create_prediction_pipeline(data=data, split_ratios=cfg["split_ratios"], gen_model=gen_model, predictor=gen_model.predictor, 
                                          score_computation_cls=ProximalCoverageScoreComputer, alphas=cfg["alphas"], score_func=score_func, 
                                          distance_func=distance_func, admission_func=admission_func, data_splitting=True, verbose=True)
    # store pipeline object to disc
    with open(CKPT_DIR + '/pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

if __name__ == "__main__":
    set_seed(0)
    cfg = load_config_from_json(CONFIG_DIR + "/default.json")
    main(cfg)
