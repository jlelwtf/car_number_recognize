import os
from shutil import move

if __name__ == '__main__':
    os.system(
        'wget '
        'https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2019-08-30.zip '
        '-P datasets'
    )

    os.system(
        'unzip '
        'datasets/autoriaNumberplateOcrRu-2019-08-30.zip '
        '-d datasets'
    )

    move(
        'datasets/autoriaNumberplateOcrRu-2019-08-30 ',
        'datasets/ocr_dataset'
    )

    os.remove(
        'datasets/autoriaNumberplateOcrRu-2019-08-30.zip'
    )
