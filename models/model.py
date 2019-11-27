from typing import List

import torch


class Model:
    _net: torch.nn.Module

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)

    def _check_cpu(self):
        return self.device.type == 'cpu'

    def save(self, weight_path: str):
        if not self._check_cpu():
            self._net.to(torch.device('cpu'))

        torch.save(self._net.state_dict(), weight_path)

        if not self._check_cpu():
            self._net.to(self.device)

        self._net.to(self.device)

    def load(self, weight_path: str):
        if not self._check_cpu():
            self._net.to(torch.device('cpu'))

        self._net.load_state_dict(torch.load(weight_path))

        if not self._check_cpu():
            self._net.to(self.device)

        self._net.eval()

    def predict(self, x: List[torch.Tensor]) -> List[dict]:
        return self._net(x)
