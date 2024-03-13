from pathlib import Path

import cv2
import torch
import logging
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

from cleanfid.fid import get_batch_features
from torchmetrics import F1Score, CohenKappa, Accuracy
from collections import defaultdict
import torchmetrics
import numpy as np

from metric_calculation import update_df
from src.data_processing.adapted_metrics import unpaired_lab_WB
from src.data_processing.config import MetricsConf, GenerateConf
from src.data_processing.fid import calculate_fid
from src.data_processing.metrics import SSIM
from src.utils.utils import get_config


# def multiclass_acc(preds, truths):
#     """
#     Compute the multiclass accuracy w.r.t. groundtruth
#
#     :param preds: Float array representing the predictions, dimension (N,)
#     :param truths: Float/int array representing the groundtruth classes, dimension (N,)
#     :return: Classification accuracy
#     """
#     return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
def av_wasserstein(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    return np.mean([wasserstein_distance(hist1[i], hist2[i]) for i in range(3)])


class All_Evaluator:
    def __init__(self, config, dataloaders: dict):
        evaluator_class = globals()["General_Evaluator"]
        self.train_evaluator = evaluator_class(config, len(dataloaders.train_loader.dataset))
        self.val_evaluator = evaluator_class(config, len(dataloaders.train_loader.dataset))
        if hasattr(dataloaders, "test_loader"):
            self.test_evaluator = evaluator_class(config, len(dataloaders.test_loader.dataset))

class General_Evaluator:
    def __init__(self, config, total_instances: int):
        self.config = config
        self.total_instances = total_instances
        self.num_classes = config.model.args.num_classes
        self.reset()

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        logging.info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def set_early_stop(self):
        self.early_stop = True

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def reset(self):
        self.losses = []
        self.SSIMs = []
        self.lab_wds = []
        self.fid_scores = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w if self.config.model.args.multi_loss.multi_supervised_w[pred_key] != 0.0}
        self.labels = torch.empty(0,256,256,3).cuda()
        self.processed_instances = 0
        self.predictions = torch.empty(0,256,256,3).cuda()

    def process(self,predictions,label,loss):
        self.processed_instances += len(predictions)
        self.losses.append(loss)
        self.labels = torch.cat((self.labels,label),dim=0)
        self.predictions = torch.cat((self.predictions,predictions),dim=0)

    def mean_batch_loss(self):
        if len(self.losses)==0:
            return None, ""
        mean_batch_loss = {}
        for key in self.losses[0].keys():
            mean_batch_loss[key] = torch.stack([self.losses[i][key] for i in range(len(self.losses))]).mean().item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self):
        metrics = defaultdict(dict)
        #CALCULATE WB
        metrics["WB"] = unpaired_lab_WB(self.labels,self.predictions)

        #CALCULATE SSIM
        index=-1
        sum_ssims = 0
        avg_hists = []
        hist_labels = np.zeros((3,256))
        hist_predictions = np.zeros((3,256))

        for label in self.labels:
            index += 1
            label = label.cpu().numpy()
            prediction = self.predictions[index].cpu().numpy()
            sum_ssims += SSIM(label,prediction)
            #sum_ssims += ssim(cv2.cvtColor(label,cv2.COLOR_BGR2GRAY),cv2.cvtColor(prediction,cv2.COLOR_BGR2GRAY), multichannel=False)
        avg_ssim = sum_ssims/(index+1)
        metrics["SSIM"] = avg_ssim

        #CALCULATE FID -> doesn't work yet
        metrics["FID"] = calculate_fid(self.predictions, self.labels, 32)
        return metrics

    # def evaluate(self):
    #     total_preds, metrics = {}, defaultdict(dict)
    #
    #     conf = get_config('config_example_he2mt.yaml')
    #     generate_conf = GenerateConf(**conf['generate'])
    #     model = generate_conf.models.models[0]
    #     generate_conf.weights = generate_conf.models.weights[0]
    #     model = model(generate_conf)
    #     #features_predictions = get_batch_features(self.predictions,model=None,device=0)
    #
    #     fakes = sorted(Path('results/masson_fake/').glob('*'))
    #     fakes = [f for f in fakes if f.is_dir()]
    #     for fake in fakes:
    #         m = MetricsConf(classic_metrics=['ssim'], center_crop=None,
    #                         source='data/he/', fake=fake)
    #         results = generate_classic_metrics(metrics_conf=m)
    #         ssim_score = results.mean().values[0]
    #         metrics["SSIM"] = ssim_score
    #         #ssim_std = results.std().values[0]
    #         lab_wd = unpaired_lab_WD('data/he/', str(fake))
    #         metrics["WD"] = lab_wd
    #         fid_score = calculate_fid(['data/masson/', str(fake)], device=0,
    #                                   batch_size=32)
    #         metrics["FID"] = fid_score
    #         self.SSIMs.append(ssim_score)
    #         self.lab_wds.append(lab_wd)
    #         self.fid_scores.append(fid_score)
    #     metrics = dict(metrics)
    #     return metrics


    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        # Flag if its saved don't save it again on $save_every
        not_saved = True
        validate_with = self.config.early_stopping.get("validate_with", "loss")
        if validate_with == "loss":
            is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])
        elif validate_with == "accuracy":
            is_best = (metrics["acc"]["combined"] > best_logs["acc"]["combined"])
        else:
            raise ValueError("self.agent.config.early_stopping.validate_with should be either loss or accuracy")
        return is_best