"""Transformer language model"""

import torch
from torch import long, nn

from pathlib import Path
from argparse import ArgumentParser
import csv
from typing import Dict, Optional
import logging
import copy

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


from pytorch_lightning.core import LightningModule

from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder


logger = logging.getLogger(__name__)



@DatasetReader.register("NMT")
class NMTDatasetReader(DatasetReader):
    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = True,
        target_add_start_token: bool = True,
        target_add_end_token: bool = True,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        quoting: int = csv.QUOTE_MINIMAL,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        self._start_token, self._end_token = self._source_tokenizer.tokenize(
            start_symbol + " " + end_symbol
        )
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(
                csv.reader(data_file, delimiter=self._delimiter,
                           quoting=self.quoting)
            ):
                source_sequence, target_sequence = row
                if len(source_sequence) == 0 or len(target_sequence) == 0:
                    continue
                yield self.text_to_instance(source_sequence, target_sequence)
        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None
    ) -> Instance:  # type: ignore
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, copy.deepcopy(self._start_token))
        if self._source_add_end_token:
            tokenized_source.append(copy.deepcopy(self._end_token))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))
            target_field = TextField(
                tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({"source_tokens": source_field})


def sequence_mask(inputs, valid_len, value=0):
    output = inputs.clone()
    for count, matrix in enumerate(output):
        matrix[int(valid_len[count]):] = value
    return output


class MaskedSoftMaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftMaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class TransformerModel(LightningModule):

    def __init__(self,
                 num_hiddens,
                 num_layers,
                 dropout,
                 num_heads,
                 ffn_num_hiddens,
                 train_path,
                 val_path,
                 learning_rate,
                 test_path,
                 cache_path,
                 **kwargs):
        super(TransformerModel, self).__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        if train_path.exists():
            print("reading training data")
            train_reader = NMTDatasetReader(cache_directory=cache_path / 'train', target_token_indexers={
                                            'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
                                            source_max_tokens=self.hparams['max_len'])
            self.train_dataset = train_reader.read(train_path)
        if val_path.exists():
            print("reading validation data")
            val_reader = NMTDatasetReader(cache_directory=cache_path / 'val', target_token_indexers={
                                          'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
                                          source_max_tokens=self.hparams['max_len'])
            self.val_dataset = val_reader.read(val_path)
        if test_path.exists():
            print("reading test data")
            test_reader = NMTDatasetReader(cache_directory=cache_path / 'test', target_token_indexers={
                                           'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
                                            source_max_tokens=self.hparams['max_len'])
            self.test_dataset = test_reader.read(test_path)

        self.vocab = Vocabulary.from_instances(
            self.train_dataset + self.val_dataset + self.test_dataset,
            min_count={
                'tokens': self.hparams['min_freq'], 'target_tokens': self.hparams['min_freq']}
        )

        self.train_dataset.index_with(self.vocab)
        self.val_dataset.index_with(self.vocab)
        self.test_dataset.index_with(self.vocab)

        self.src_embedding = Embedding(
            num_embeddings=self.vocab.get_vocab_size('tokens'), embedding_dim=self.hparams['embedding_dim'])
        self.src_embedder = BasicTextFieldEmbedder(
            token_embedders={'tokens': self.src_embedding})

        self.trg_embedding = Embedding(
            num_embeddings=self.vocab.get_vocab_size('target_tokens'), embedding_dim=self.hparams['embedding_dim'])
        self.trg_embedder = BasicTextFieldEmbedder(
            token_embedders={'target_tokens': self.trg_embedding})

        self.encoder = TransformerEncoder(self.src_embedder, num_hiddens, ffn_num_hiddens, num_heads, num_layers,
                                          dropout)
        self.decoder = TransformerDecoder(self.trg_embedder, self.vocab.get_vocab_size('target_tokens'),
                                          num_hiddens, ffn_num_hiddens, num_heads, num_layers,
                                          dropout)
        self.loss = MaskedSoftMaxCELoss()

    def forward(self, enc_input, dec_input, *args):

        enc_output = self.encoder(enc_input, *args)
        dec_state = self.decoder.init_state(enc_output, *args)
        return self.decoder(dec_input, dec_state)

    def predict(self, sentence, num_steps):

        tokens = [self.vocab.get_token_index(token, namespace='tokens') for token in sentence.split()]
        tokenize_sentence = {'tokens': {'tokens': torch.tensor([tokens])}}
        
        encoder_output = self.encoder(tokenize_sentence)
        decoder_state = self.decoder.init_state(encoder_output)
        decoder_input = torch.unsqueeze(torch.tensor([self.vocab.get_token_index('@start@', namespace='tokens')], dtype=torch.long), dim=0)

        predictions = []
        for _ in range(num_steps):
            decoder_input = {'target_tokens': {'tokens': decoder_input}}
            Y, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_input = Y.argmax(dim=2)
            pred = decoder_input.squeeze(dim=0).type(torch.int32).item()
            if pred == self.vocab.get_token_index('@end@', namespace='tokens'):
                print('end')
                break
            predictions.append(pred)
        return ' '.join([str(self.vocab.get_token_from_index(token, namespace='tokens')) for token in predictions])


    def training_step(self, batch, batch_idx):
        src, trg = batch['source_tokens'], batch['target_tokens']
        X, X_valid_len = src, get_text_field_mask(src).sum(axis=1)
        Y, Y_valid_len = trg,  get_text_field_mask(trg).sum(axis=1)

        Y_input = {'target_tokens': {'tokens': None}}
        Y_label = {'target_tokens': {'tokens': None}}


        Y_input['target_tokens']['tokens'], Y_label['target_tokens']['tokens'], Y_valid_len = Y['tokens']['tokens'][:, :-1], Y['tokens']['tokens'][:, 1:], Y_valid_len-1

        Y_hat, _ = self(X, Y_input, X_valid_len, Y_valid_len)

        l = self.loss(Y_hat, Y_label['target_tokens']['tokens'], Y_valid_len)
        logs = {'train_loss': l.sum().detach() / Y_valid_len.sum()}
        return {'loss': l.sum(), 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):

        batch_sampler = BucketBatchSampler(self.train_dataset,
                                           batch_size=self.hparams['batch_size'],
                                           sorting_keys=['source_tokens', 'target_tokens'],
                                           drop_last=True)
        train_data_loader = DataLoader(
            self.train_dataset, batch_sampler=batch_sampler, num_workers=self.hparams['num_workers'])
        return train_data_loader

    def validation_step(self, batch, batch_idx):
        src, trg = batch['source_tokens'], batch['target_tokens']
        X, X_valid_len = src, get_text_field_mask(src).sum(axis=1)
        Y, Y_valid_len = trg,  get_text_field_mask(trg).sum(axis=1)

        Y_input = {'target_tokens': {'tokens': None}}
        Y_label = {'target_tokens': {'tokens': None}}

        Y_input['target_tokens']['tokens'], Y_label['target_tokens']['tokens'], Y_valid_len = Y['tokens']['tokens'][:, :-1], Y['tokens']['tokens'][:, 1:], Y_valid_len-1

        Y_hat, _ = self(X, Y_input, X_valid_len, Y_valid_len)
        l = self.loss(Y_hat, Y_label['target_tokens']['tokens'], Y_valid_len)
        return {'val_loss': l.sum() / Y_valid_len.sum()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': loss}
        results = {'val_loss': loss, 'log': logs}
        return results

    def val_dataloader(self):
        batch_sampler = BucketBatchSampler(self.val_dataset,
                                           batch_size=self.hparams['batch_size'], 
                                           sorting_keys=['source_tokens', 'target_tokens'],
                                           drop_last=True)
        val_data_loader = DataLoader(
            self.val_dataset, batch_sampler=batch_sampler, num_workers=self.hparams['num_workers'])
        return val_data_loader

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--num_hiddens', default=64, type=int)
        parser.add_argument('--ffn_num_hiddens', default=64, type=int)
        parser.add_argument('--num_heads', default=2, type=int)
        parser.add_argument('--num_layers', default=2, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--embedding_dim', default=64)

        # data
        parser.add_argument(
            '--train_path', default=Path(root_dir) / 'data' / 'train_small.tsv', type=str)
        parser.add_argument(
            '--val_path', default=Path(root_dir) / 'data' / 'val_small.tsv', type=str)
        parser.add_argument(
            '--test_path', default=Path(root_dir) / 'data' / 'test_small.tsv', type=str)
        parser.add_argument(
            '--cache_path', default=Path(root_dir) / 'data' / 'cache-data', type=str)
        parser.add_argument('--max_len', default=50)
        parser.add_argument('--min_freq', default=15)

        return parser
