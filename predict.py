from pytorch_lightning.core import LightningModule
from transformer.model import TransformerModel

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--sentence', type=str)

    return parser.parse_args()


def main(checkpoint_path, sentence, num_steps):

    pretrained_model= TransformerModel.load_from_checkpoint(checkpoint_path)
    prediction = pretrained_model.predict(sentence, num_steps)

    print(sentence, ' --> ', prediction)




if __name__ == '__main__':
    main(**vars(get_args()))


