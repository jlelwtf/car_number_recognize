from typing import List

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from datasets.ocr_dataset import OCRDataset
from models.model import Model
from models.ocr_net import OCRNet
from torch_vis.transforms import Compose, TransformImageForOCR, TransformRuLabel, \
    ToOCRTensor


class OCRModel(Model):

    def __init__(
            self,
            symbol_list: List[str],
            sep_symbol='-',
            space_symbol='|',
            image_height=128,
            image_width=64,
            device='cuda',
    ):
        super().__init__(device=device)
        self.space_symbol = space_symbol
        self.sep_symbol = sep_symbol
        self._image_width = image_width
        self._image_height = image_height
        self._symbol_list = symbol_list
        self._net = OCRNet(num_classes=len(symbol_list) + 1)
        self._net.to(device)
        self._transforms = self._get_transforms()

    def _get_transforms(self):
        return Compose([
            TransformImageForOCR(self._image_height, self._image_width),
            ToOCRTensor(self._symbol_list, self._net.out_height,
                        self._net.out_height, len(self._symbol_list) - 1)
        ])

    def _get_target_lengths(self, target):
        target_lengths = []
        for seq in target:
            i = seq.shape[0] - 1
            while not seq[i] or i == 0:
                i -= 1
            target_lengths.append(i + 1)

        return torch.IntTensor(target_lengths)

    def _get_input_lengths(self, size):
        return torch.full(
                size=(size,),
                fill_value=self._net.out_height,
                dtype=torch.long
            )

    def _train_one_epoch(self, data_loader, loss_func, optimizer):
        self._net.train()
        count = 0
        losses = []
        for image_batch, target_batch in data_loader:

            input_lengths = self._get_input_lengths(image_batch.shape[0]).to(self.device)
            target_lengths = self._get_target_lengths(target_batch)
            target_lengths = target_lengths.to(self.device)

            count += len(image_batch)
            print(f'Train: {count}/{len(data_loader.dataset)}. Loss: ', end='')

            image_batch = image_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            predict = self._net.forward(image_batch)

            activation = torch.nn.LogSoftmax(dim=2)
            predict = activation(predict)

            predict = predict.permute(1, 0, 2)

            loss = loss_func(predict, target_batch, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss_val = loss.data.cpu()
            losses.append(loss_val)
            print(f'{loss_val}                              ', end='\r')
        loss_val = np.mean(losses)
        print(f'Train loss: {loss_val}                                           ')
        return loss_val

    def _evaluate(self, data_loader, loss_func):
        self._net.eval()

        loss_vals = []

        for image_batch, target_batch in data_loader:
            input_lengths = self._get_input_lengths(image_batch.shape[0]).to(self.device)
            target_lengths = self._get_target_lengths(target_batch).to(self.device)

            image_batch = image_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            predict = self._net.forward(image_batch)

            activation = torch.nn.LogSoftmax(dim=2)
            predict = activation(predict)

            predict = predict.permute(1, 0, 2)

            loss_vals.append(
                loss_func(predict, target_batch, input_lengths, target_lengths).data.cpu()
            )

        loss = np.mean(loss_vals)
        return loss

    def train(
            self,
            train_dataset_path: str,
            val_dataset_path: str,
            batch_size=32,
            num_epochs=10
    ):
        dataset = OCRDataset(train_dataset_path, self._transforms)
        dataset_val = OCRDataset(val_dataset_path, self._transforms)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        data_loader_val = DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        # optimizer = SGD(self._net.parameters(), lr=0.002)
        optimizer = torch.optim.Adam(self._net.parameters())

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        loss_func = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        flag = True
        val_los = 1
        epoch = 1
        while val_los > 0.8 and epoch <= num_epochs:
            print(f'Epoch {epoch}/{num_epochs}')
            loss = self._train_one_epoch(data_loader, loss_func, optimizer)
            if loss < 1.2 and flag:
                flag = False
                lr_scheduler.step(epoch)

            val_loss = self._evaluate(data_loader_val, loss_func)

            print(f'Evaluate: Loss: {val_loss}')
            print()
            epoch += 1

    def predict(self, x: List[torch.Tensor]):
        activation = torch.nn.Softmax(dim=2)
        return activation(super().predict(x))
