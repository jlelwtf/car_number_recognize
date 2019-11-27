import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
from datasets.mask_plate_dataset import MaskPlateDataset
from models.model import Model
from torch_vis.engine import train_one_epoch, evaluate
from torch_vis.transforms import ToRCNNTensor, RandomHorizontalFlip, Compose

'''
From https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''


class MaskRCNN(Model):

    def __init__(
            self,
            num_classes: int = 2,
            hidden_layer: int = 256,
            device: str = 'cuda'
    ):
        super().__init__(device)
        self._num_classes = num_classes
        self._hidden_layer = hidden_layer
        self._create_model()

    def _create_model(self):
        self._net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self._net.roi_heads.box_predictor.cls_score.in_features

        self._net.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self._num_classes
        )

        in_features_mask = self._net.roi_heads.mask_predictor.conv5_mask.in_channels

        self._net.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            self._hidden_layer,
            self._num_classes
        )
        self._net.to(self.device)

    @staticmethod
    def _get_transform(train: bool):
        transforms = [ToRCNNTensor()]
        if train:
            transforms.append(RandomHorizontalFlip(0.5))
        return Compose(transforms)

    def train(
            self,
            train_dataset_path: str,
            val_dataset_path: str,
            weight_path: str,
            batch_size: int = 2,
            num_epochs: int = 10,
            print_freq: int = 10,
    ):
        dataset = MaskPlateDataset(train_dataset_path, self._get_transform(True))
        dataset_val = MaskPlateDataset(val_dataset_path, self._get_transform(False))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn
        )

        params = [p for p in self._net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        for epoch in range(num_epochs):
            train_one_epoch(
                self._net, optimizer, data_loader,
                self.device, epoch, print_freq=print_freq
            )
            lr_scheduler.step()
            evaluate(self._net, data_loader_test, device=self.device)

        torch.save(self._net.state_dict(), weight_path)
