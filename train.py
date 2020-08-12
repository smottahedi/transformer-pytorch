"""Train model on CPU"""

import os
from argparse import ArgumentParser
from warnings import filterwarnings

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer

from transformer.model import TransformerModel


filterwarnings('ignore')


def get_args():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--seed', default=1, type=int)
    parent_parser.add_argument('--epochs', default=20, type=int)
    parent_parser.add_argument('--batch_size', default=64, type=int)
    parent_parser.add_argument('--learning_rate', default=0.00275, type=float)
    parent_parser.add_argument('--gpus', default=None)
    parent_parser.add_argument('--tpu_cores', default=None)
    parent_parser.add_argument('--num_workers', default=8)
    parser = TransformerModel.add_model_specific_args(parent_parser, root_dir)
    return parser.parse_args()


def main(args):
    
    wand_logger = WandbLogger(offline=False, project='Transformer', save_dir='./lightning_logs/')
    wand_logger.log_hyperparams(params=args)
    
    checkpoint = ModelCheckpoint(filepath='./lightning_logs/checkpoints/checkpoints',
                                 monitor='val_loss', verbose=0, save_top_k=2)
    
    model = TransformerModel(**vars(args))
    trainer = Trainer(
        logger=wand_logger,
        early_stop_callback=False,
        checkpoint_callback=checkpoint,
        # fast_dev_run=True,
        # overfit_pct=0.03,
        # profiler=True,
        auto_lr_find=False,
        # val_check_interval=1.0,
        # log_save_interval=50000,
        # row_log_interval=50000,
        max_epochs=args.epochs,
        min_epochs=1,
    )
    # lr_finder = trainer.lr_find(model)
    # print(lr_finder.results)
    trainer.fit(model)

if  __name__ == '__main__':
    main(get_args())