from models.ocr_model import OCRModel
from torch_vis.transforms import Compose, TransformImageForOCR, TransformRuLabel, \
    ToOCRTensor


class RuOCRModel(OCRModel):

    def __init__(self, device):
        symbol_list = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "E", "H", "K", "M", "O", "P", "T",
            "X", "Y", "-", "|"
        ]

        super().__init__(symbol_list, device=device)
        self.sep_symbol = '-'
        self.space_symbol = '|'

    def _get_transforms(self):
        return Compose([
            TransformImageForOCR(self._image_height, self._image_width),
            TransformRuLabel(self.sep_symbol, self.space_symbol),
            ToOCRTensor(self._symbol_list, self._net.out_height,
                        self._net.out_height, len(self._symbol_list) - 1)
        ])
