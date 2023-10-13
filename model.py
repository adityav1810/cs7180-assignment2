"""
File: model.py

Description: 
    This file contains the implementation of SqueezeNet with attention layers added using PyTorch with pretrained weights loaded.
    SqueezeNet is a deep neural network architecture that achieves a good balance between accuracy and model size.
    The model is loaded with pre-trained weights obtained from a pre-trained SqueezeNet model.

Authors:
    Author 1 (Aditya Varshney,varshney.ad@northeastern.edu, Northeastern University)
    Author 2 (Luv Verma, verma.lu@northeastern.edu , Northeastern University)

Citations and References:
    - Reference 1: https://github.com/matteo-rizzo/fc4-pytorch
    
"""

import os
from typing import Union, Tuple
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms import transforms
from torch.utils import model_zoo
import torch
import torch.nn as nn
import torch.nn.init as init
# from torch import nn, Tensor
from torch import Tensor
from torch.nn.functional import normalize
from utils import USE_CONFIDENCE_WEIGHTED_POOLING ,correct, rescale, scale,DEVICE
from modules.Loss import AngularLoss

model_urls = {
    1.0: 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    1.1: 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        attention = torch.bmm(query, key)
        attention = torch.nn.functional.softmax(attention, dim=1)
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out
    
class Fire(nn.Module):

    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
                          self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version: float = 1.0, num_classes: int = 1000):
        super().__init__()

        self.num_classes = num_classes
        self.attention1 = SelfAttention(96)
        self.attention2 = SelfAttention(256)
        self.attention3 = SelfAttention(512)
        self.attention_classifier = SelfAttention(512)

        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                self.attention1,
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                self.attention2,
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                self.attention3,
                Fire(512, 64, 256, 256),
            )
        elif version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}: 1.0 or 1.1 expected".format(version=version))

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            self.attention_classifier,
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


# class SqueezeNetLoader:
#     def __init__(self, version: float = 1.1):
#         self.__version = version
#         self.__model = SqueezeNet(self.__version)

#     def load(self, pretrained: bool = False) -> SqueezeNet:
#         """
#         Returns the specified version of SqueezeNet
#         @param pretrained: if True, returns a model pre-trained on ImageNet
#         """
#         if pretrained:
#             path_to_local = os.path.join("assets", "pretrained")
#             print("\n Loading local model at: {} \n".format(path_to_local))
#             os.environ['TORCH_HOME'] = path_to_local
#             self.__model.load_state_dict(model_zoo.load_url(model_urls[self.__version]))
#         return self.__model


class SqueezeNetLoader:
    def __init__(self, version: float = 1.1):
        self.__version = version
        self.__model = SqueezeNet(self.__version)

    def load(self, pretrained: bool = False) -> SqueezeNet:
        """
        Returns the specified version of SqueezeNet
        @param pretrained: if True, returns a model pre-trained on ImageNet
        """
        if pretrained:
            path_to_local = os.path.join("assets", "pretrained")
            print("\n Loading local model at: {} \n".format(path_to_local))
            os.environ['TORCH_HOME'] = path_to_local
            
            # Load the pre-trained weights into a temp model
            # pre_trained_weights = model_zoo.load_url(model_urls[self.__version])
            # print(os.getcwd())
            # pre_trained_weights = torch.load("trained_models/fc4_cwp/fold_0/model.pth")
            pre_trained_weights = torch.load("modelWAttention.pth")

            # Create a copy of the model's state dict
            model_dict = self.__model.state_dict()

            # 1. Filter out unnecessary keys from the pre-trained weights
            # 2. Remove any mismatching layers
            pretrained_dict = {}
            layer_names = []
            for k, v in pre_trained_weights.items():
                if(k in model_dict and model_dict[k].shape == v.shape):
                    # print(k,type(k),type(v),v.shape)
                    layer_names.append(k)
                    print(f"Freezing {[k]} weights ....\t")
                    self.__model.state_dict()[k] = v
                    self.__model.state_dict()[k].requires_grad = False
            # pretrained_dict = {k: v for k, v in pre_trained_weights.items() if k in model_dict and model_dict[k].shape == v.shape}
            
            # Update the current model's dict
            model_dict.update(pretrained_dict)
            # Load the new state dict
            self.__model.load_state_dict(model_dict)
            # dfs_freeze(self.__model,layer_names)
            
        return self.__model

    
class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        # self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])
        # print(self.backbone)
        modules = list(squeezenet.features.children())
        self.backbone = nn.Sequential(*modules)
        # Instantiate attention layer for FC4
        # self.attention_fc4 = SelfAttention(512)

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4 if USE_CONFIDENCE_WEIGHTED_POOLING else 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """

        x = self.backbone(x)
        # x = self.attention_fc4(x)
        # x = SelfAttention(x.shape[1])(x)  # Attention adding
        out = self.final_convs(x)
        print(f"Passing through feed forward of FC4 ")
        # print(f"Layers found {self.backbone} , {self.backbone.features[0].shape}")
        for _, i in enumerate(self.backbone):
            print(_,i)

        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            # Per-patch color estimates (first 3 dimensions)
            rgb = normalize(out[:, :3, :, :], dim=1)

            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]
            print(f"out dimensions : {out.shape}\n")

            # Confidence-weighted pooling
            pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

            return pred, rgb, confidence

        # Summation pooling
        pred = normalize(torch.sum(torch.sum(out, 2), 2), dim=1)

        return pred
    
class Model:
    def __init__(self):
        self._device = DEVICE
        self._criterion = AngularLoss(self._device)
        self._optimizer = None
        self._network = None

    def print_network(self):
        print("\n----------------------------------------------------------\n")
        print(self._network)
        print("\n----------------------------------------------------------\n")

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self._network))

    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self._criterion(pred, label)

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_log: str):
        torch.save(self._network.state_dict(), os.path.join(path_to_log, "model.pth"))

    def load(self, path_to_pretrained: str):
        # path_to_model = os.path.join(path_to_pretrained, "model.pth")
        self._network.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "adam"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)
    



class ModelFC4(Model):

    def __init__(self):
        super().__init__()
        self._network = FC4(1.0).to(self._device)

    def predict(self, img: Tensor, return_steps: bool = False) -> Union[Tensor, Tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, rgb, confidence = self._network(img)
            if return_steps:
                return pred, rgb, confidence
            return pred
        return self._network(img)

    def optimize(self, img: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(img)
        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def save_vis(self, model_output: dict, path_to_plot: str):
        model_output = {k: v.clone().detach().to(self._device) for k, v in model_output.items()}

        img, label, pred = model_output["img"], model_output["label"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]
        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)
        masked_original = scale(F.to_tensor(original).to(self._device).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_loss(pred, label)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')
        
    def getActivation(self,layer_name:str):
        for k,w in self._network:
            print(k,w)
