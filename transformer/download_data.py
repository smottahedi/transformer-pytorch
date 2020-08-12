"""download and process translation dataset"""


import argparse
from collections import defaultdict
from pathlib import Path
import random

ROOT = Path(".")
DATA = ROOT / 'data'

def read_sentences(filename, target_language, src_language):
    """
    Read sentences.csv and returns a dict containing sentence information.
    Parameters:
        filename (str): filename of 'sentence.csv'
        target_language (str): target language
        src_language (str): src language
    Returns:
        dict from sentence id (int) to Sentence information, where
        sentence information is a dict with 'sent_id', 'lang', and 'text' keys.
        dict only contains sentences in target_language or src_language."""

    sentences = {}
    for line in open(filename):
        sent_id, lang, text = line.rstrip().split('\t')
        if lang == src_language or lang == target_language:
            sent_id = int(sent_id)
            sentences[sent_id] = {'sent_id': sent_id, 'lang': lang, 'text': text}
    return sentences

def read_links(filename):
    """
    Read links.csv and returns a dict containing links information.
    Args:
        filename (str): filename of 'links.csv'
    Returns:
        dict from sentence id (int) of a sentence and a set of all its translation sentence ids."""

    links = defaultdict(set)
    for line in open(filename):
        sent_id, trans_id = line.rstrip().split('\t')
        links[int(sent_id)].add(int(trans_id))
    return links

def generate_translation_pairs(sentences, links, target_language, src_language):
    """
    Given sentences and links, generate a list of sentence pairs in target and source languages.
    Parameters:
        sentences: dict of sentence information (returned by read_sentences())
        links: dict of links information (returned by read_links())
        target_language (str): target language
        src_language (str): src language
    Returns:
        list of sentence pairs (sentence info 1, sentence info 2)
        where sentence info 1 is in target_language and sentence info 2 in src_language.
    """
    translations = []
    for sent_id, trans_ids in links.items():
        # Links in links.csv are reciprocal, meaning that if (id1, id2) is in the file,
        # (id2, id1) is also in the file. So we don't have to check both directions.
        if sent_id in sentences and sentences[sent_id]['lang'] == target_language:
            for trans_id in trans_ids:
                if trans_id in sentences and sentences[trans_id]['lang'] == src_language:
                    translations.append((sentences[sent_id]['text'], sentences[trans_id]['text']))
    return translations

def write_tsv(translations, train_ratio, val_ratio, test_ratio):
    """
    Write translations as TSV to stdout.
    Parameters:
        translations (list): list of sentence pairs returned by generate_translation_pairs()
    """
    N = len(translations)
    train_len = int(N * train_ratio)
    val_len = int(N * val_ratio)
    test_len = int(N * test_ratio)

    train = translations[:train_len]
    val = translations[train_len: train_len + val_len]
    test = translations[train_len + val_len:]

    with (DATA / 'train.tsv').open('w') as f:
        for sent1, sent2 in train:
            f.write("%s\t%s\n" % (sent2, sent1))

    with (DATA / 'val.tsv').open('w') as f:
        for sent1, sent2 in val:
            f.write("%s\t%s\n" % (sent2, sent1))


    with (DATA / 'test.tsv').open('w') as f:
        for sent1, sent2 in test:
            f.write("%s\t%s\n" % (sent2, sent1))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_language')
    parser.add_argument('-trg_language')
    parser.add_argument('-sentences_path', help='filename of sentences.csv')
    parser.add_argument('-links_path', help='filename of links.csv')
    parser.add_argument('-train_ratio', default=0.7)
    parser.add_argument('-val_ratio', default=0.15)
    parser.add_argument('-test_ratio', default=0.15)
    args = parser.parse_args()
    
    target_language, src_language = args.trg_language, args.src_language
    sentences = read_sentences(args.sentences_path, target_language, src_language)

    links = read_links(args.links_path)

    translations = generate_translation_pairs(sentences, links, target_language, src_language)

    train_ratio, val_ratio, test_ratio = args.train_ratio, args.val_ratio, args.test_ratio
    write_tsv(translations, train_ratio, val_ratio, test_ratio)


if __name__ == '__main__':
    main()