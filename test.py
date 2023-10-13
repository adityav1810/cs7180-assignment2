"""
File: test.py

Description: 
    File to perform testing with a trained model. 
Authors:
    Author 1 (Aditya Varshney,varshney.ad@northeastern.edu, Northeastern University)
    Author 2 (Luv Verma, verma.lu@northeastern.edu , Northeastern University)

Citations and References:
    - Reference 1: https://github.com/matteo-rizzo/fc4-pytorch
    
"""


import os
from time import time

import numpy as np
import torch.utils.data

from utils import DEVICE
from modules.Evaluator import Evaluator
from ColorCheckerDataset import ColorCheckerDataset
from model import ModelFC4

MODEL_TYPE = "fc4_cwp"
SAVE_PRED = False
SAVE_CONF = False
USE_TRAINING_SET = False


def main():
    evaluator = Evaluator()
    model = ModelFC4()
    path_to_pred, path_to_pred_fold = None, None
    path_to_conf, path_to_conf_fold = None, None

    if SAVE_PRED:
        path_to_pred = os.path.join("test", "pred", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))

    if SAVE_CONF:
        path_to_conf = os.path.join("test", "conf", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))

    for num_fold in range(3):
        fold_evaluator = Evaluator()
        test_set = ColorCheckerDataset(train=USE_TRAINING_SET, folds_num=num_fold)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

        path_to_pretrained = os.path.join("trained_models", MODEL_TYPE, "fold_{}".format(num_fold))
        # path_to_pretrained = "/home/varshney.ad/cs7180/ImageEnhancement/trained_models/trained_models/fc4_cwp/fold_0"
        model.load(path_to_pretrained)
        model.evaluation_mode()

        if SAVE_PRED:
            path_to_pred_fold = os.path.join(path_to_pred, "fold_{}".format(num_fold))
            os.makedirs(path_to_pred_fold)

        if SAVE_CONF:
            path_to_conf_fold = os.path.join(path_to_conf, "fold_{}".format(num_fold))
            os.makedirs(path_to_conf_fold)

        print("\n *** FOLD {} *** \n".format(num_fold))
        print(" * Test set size: {}".format(len(test_set)))
        print(" * Using trained model stored at: {} \n".format(path_to_pretrained))

        with torch.no_grad():
            for i, (img, label, file_name) in enumerate(dataloader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred, _, conf = model.predict(img, return_steps=True)
                loss = model.get_loss(pred, label)
                fold_evaluator.add_error(loss.item())
                evaluator.add_error(loss.item())
                print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name[0], i, loss.item()))
                if SAVE_PRED:
                    np.save(os.path.join(path_to_pred_fold, file_name[0]), pred)
                if SAVE_CONF:
                    np.save(os.path.join(path_to_conf_fold, file_name[0]), conf)

        metrics = fold_evaluator.compute_metrics()
        print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
        print(" Median .......... : {:.4f}".format(metrics["median"]))
        print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
        print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
        print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
        print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))

    print("\n *** AVERAGE ACROSS FOLDS *** \n")
    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
    print(" Median .......... : {:.4f}".format(metrics["median"]))
    print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
    print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()