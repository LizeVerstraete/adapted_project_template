from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from src.data_processing.metrics import generate_classic_metrics
from src.utils.utils import get_config
from src.data_processing.config import GenerateConf, MetricsConf


class Validator_Tester():
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False

    def validate(self, best_model=False, test_set=False):
        this_evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        this_evaluator.reset()

        conf = get_config('config_example_he2mt.yaml')
        generate_conf = GenerateConf(**conf['generate'])
        model = generate_conf.models.models[0]
        results_path = Path(conf["generate"]["results_path"]).absolute()
        image_outs = results_path / "CycleGan"
        generate_conf.weights = generate_conf.models.weights[0]
        model = model(generate_conf)
        #stores outputs in /results/masson_fake/CycleGan
        model.image_outs.mkdir(parents=True, exist_ok=True)
        model.eval()
        total_output_list = []
        with torch.no_grad():
            for imgs, imgs_path in tqdm(model.dataloader):
                output = model.post_transform(model.forward(imgs))
                total_output_list.append(output)
                model.save_outputs(output, imgs_path)
            total_output = np.concatenate(total_output_list, axis=0)
            output_losses = {}
            fakes = sorted(Path('results/masson_fake/').glob('*'))
            fakes = [f for f in fakes if f.is_dir()]
            for fake in fakes:
                m = MetricsConf(classic_metrics=['ssim'], center_crop=None,
                                source='data/he/', fake=fake)
                results = generate_classic_metrics(metrics_conf=m)
                ssim_scores = results.CycleGan.SSIM.array.to_numpy()
                #ssim_score = results.mean().values[0]
                output_losses.update({"SSIMs": ssim_scores})
            all_outputs= {"loss": output_losses,"pred": total_output}
            this_evaluator.process(all_outputs)

    def _get_loss_weights(self):
        pass
