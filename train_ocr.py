import argparse
import sys

from models.ru_ocr_model import RuOCRModel

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train_dataset_path', default='datasets/ocr_dataset/train')
arg_parser.add_argument('--val_dataset_path', default='datasets/ocr_dataset/val')
arg_parser.add_argument('--model_weights_path', default='models/data/ocr_model_improve.pth')
arg_parser.add_argument('--pretrained_model_path', default='')
arg_parser.add_argument('--device', default='cpu')
arg_parser.add_argument('--num_epochs', default=10, type=int)
arg_parser.add_argument('--lr', default=0.002, type=float)
arg_parser.add_argument('--decrease_lr', default=True, type=bool)


if __name__ == '__main__':

    argv = arg_parser.parse_args(sys.argv[1:])

    model = RuOCRModel(device=argv.device)

    random_weight_init = True

    if argv.pretrained_model_path:
        model.load(argv.pretrained_model_path)
        random_weight_init = False

    print(type(argv.decrease_lr))
    model.train(
        argv.train_dataset_path, argv.val_dataset_path,
        argv.num_epochs, argv.lr, argv.decrease_lr, random_weight_init
    )

    model.save(argv.model_weights_path)
