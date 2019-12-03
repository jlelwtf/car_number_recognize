import argparse
import sys

from models.ru_ocr_model import RuOCRModel

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train_dataset_path', default='datasets/ocr_dataset/train')
arg_parser.add_argument('--val_dataset_path', default='datasets/ocr_dataset/val')
arg_parser.add_argument('--model_weights_path', default='models/data/ocr_model.pth')
arg_parser.add_argument('--device', default='cpu')
arg_parser.add_argument('--batch_size', default=32, type=int)
arg_parser.add_argument('--num_epochs', default=10, type=int)


if __name__ == '__main__':

    argv = arg_parser.parse_args(sys.argv[1:])

    model = RuOCRModel(device=argv.device)

    model.train(argv.train_dataset_path, argv.val_dataset_path,
                argv.batch_size, argv.num_epochs)

    model.save(argv.model_weights_path)
