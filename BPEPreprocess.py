import argparse
import collections
import logging
import os
import sys
import re
import pickle
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
from seq2seq import utils
from seq2seq.data.dictionary import Dictionary
from io import StringIO



def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default=None, metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--vocab-src', default=None, type=str, help='path to dictionary')
    parser.add_argument('--vocab-trg', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')

    # Add BPE arguments
    parser.add_argument('--bpe-codes-src', default=None, type=str, help='path to source language BPE codes')
    parser.add_argument('--bpe-codes-tgt', default=None, type=str, help='path to target language BPE codes')
    parser.add_argument('--num-merge-operations', default=32000, type=int, help='number of BPE merge operations')
    return parser.parse_args()


def apply_bpe_to_data(bpe_codes_path, input_file, output_file):
    bpe = BPE(codecs.open(bpe_codes_path, encoding='utf-8'))
    with codecs.open(input_file, 'r', encoding='utf-8') as infile, \
            codecs.open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(bpe.process_line(line.rstrip()) + '\n')


def build_bpe_dictionary(tokenized_file):
    dictionary = Dictionary()
    with codecs.open(tokenized_file, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                dictionary.add_word(token)
    return dictionary


def word_tokenize(line):
    SPACE_NORMALIZER = re.compile("\s+")
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info(
                'Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
                    input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)

    # Train or load BPE codes for source language
    if not args.bpe_codes_src:
        bpe_out_src = StringIO()
        with open(args.train_prefix + '.' + args.source_lang, 'r', encoding='utf-8') as src_data:
            learn_bpe(src_data, bpe_out_src, num_symbols=args.num_merge_operations)
        args.bpe_codes_src = os.path.join(args.dest_dir, 'bpe.codes.' + args.source_lang)
        with open(args.bpe_codes_src, 'w', encoding='utf-8') as codes_file:
            bpe_out_src.seek(0)
            codes_file.write(bpe_out_src.read())

    # Train or load BPE codes for target language
    if not args.bpe_codes_tgt:
        bpe_out_tgt = StringIO()
        with open(args.train_prefix + '.' + args.target_lang, 'r', encoding='utf-8') as tgt_data:
            learn_bpe(tgt_data, bpe_out_tgt, num_symbols=args.num_merge_operations)
        args.bpe_codes_tgt = os.path.join(args.dest_dir, 'bpe.codes.' + args.target_lang)
        with open(args.bpe_codes_tgt, 'w', encoding='utf-8') as codes_file:
            bpe_out_tgt.seek(0)
            codes_file.write(bpe_out_tgt.read())

    # Apply BPE to preprocessed data
    for split in ['train', 'valid', 'test','tiny_train']:
        for lang in [args.source_lang, args.target_lang]:
            bpe_codes_path = args.bpe_codes_src if lang == args.source_lang else args.bpe_codes_tgt
            input_file = f'{args.train_prefix}.{lang}' if split == 'train' else f'{args.valid_prefix}.{lang}' \
                if split == 'valid' else f'{args.test_prefix}.{lang}' if split == 'test' else f'{args.tiny_train_prefix}.{lang}'
            output_file = os.path.join(args.dest_dir, f'{split}.bpe.{lang}')
            apply_bpe_to_data(bpe_codes_path, input_file, output_file)

    logging.info("Building dictionaries and binarizing datasets...")
    for lang in [args.source_lang, args.target_lang]:
        bpe_dict_path = os.path.join(args.dest_dir, f'dict.bpe.{lang}')
        logging.info(f"Building dictionary for language: {lang}")
        bpe_dict = build_bpe_dictionary(os.path.join(args.dest_dir, f'train.bpe.{lang}'))
        bpe_dict.save(bpe_dict_path)
        for split in ['train', 'valid', 'test', 'tiny_train']:
            input_file = os.path.join(args.dest_dir, f'{split}.bpe.{lang}')
            output_file = os.path.join(args.dest_dir, f'{split}.bin.{lang}')
            logging.info(f"Binarizing {input_file} and saving to {output_file}")
            make_binary_dataset(input_file, output_file, bpe_dict)


if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        utils.init_logging(args)
        logging.info('COMMAND: %s' % ' '.join(sys.argv))
        logging.info('Arguments: {}'.format(vars(args)))
    main(args)
