import argparse
import collections
import logging
import os
import re
import sys
import pickle
import codecs
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
from subword_nmt.get_vocab import get_vocab
from seq2seq import utils
from seq2seq.data.dictionary import Dictionary
from io import StringIO

def get_args():
    parser = argparse.ArgumentParser('Data pre-processing with BPE')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default=None, metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=1, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=1, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--vocab-src', default=None, type=str, help='path to dictionary')
    parser.add_argument('--vocab-trg', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')

    # Add BPE arguments
    parser.add_argument('--bpe-codes-src', default=None, type=str, help='path to source language BPE codes')
    parser.add_argument('--bpe-codes-tgt', default=None, type=str, help='path to target language BPE codes')
    parser.add_argument('--vocab-threshold', default=1, type=int, help='vocabulary threshold for BPE')
    parser.add_argument('--num-merge-operations', default=32000, type=int, help='number of BPE merge operations')
    return parser.parse_args()

def apply_bpe_to_data(bpe, input_file, output_file):
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


def apply_bpe_with_vocab_filter(bpe, input_file, output_file, vocab_path, threshold):
    # Load the vocabulary from the file
    vocab = {}
    with codecs.open(vocab_path, 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            word, freq = line.strip().split()
            vocab[word] = int(freq)

    # Apply BPE with vocabulary filter
    with codecs.open(input_file, 'r', encoding='utf-8') as infile, \
            codecs.open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            bpe_line = bpe.process_line(line.rstrip())
            filtered_line = []
            for word in bpe_line.split():
                if vocab.get(word, 0) >= threshold:
                    filtered_line.append(word)
                else:
                    filtered_line.append('@@UNKNOWN@@')  # Use your BPE implementation's symbol for unknown tokens
            outfile.write(' '.join(filtered_line) + '\n')


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

    # Step 1: Learn joint BPE codes for source and target languages
    logging.info("Learning joint BPE on source and target languages.")
    bpe_codes_path = os.path.join(args.dest_dir, 'bpe.codes')
    with open(args.train_prefix + '.' + args.source_lang, 'r', encoding='utf-8') as src_file, \
         open(args.train_prefix + '.' + args.target_lang, 'r', encoding='utf-8') as tgt_file, \
         open(bpe_codes_path, 'w', encoding='utf-8') as codes_file:
        learn_bpe(src_file, codes_file, args.num_merge_operations)
        learn_bpe(tgt_file, codes_file, args.num_merge_operations)

    # Step 2: Apply BPE to training data for each language
    bpe = BPE(codecs.open(bpe_codes_path, encoding='utf-8'))
    for lang in [args.source_lang, args.target_lang]:
        input_path = f'{args.train_prefix}.{lang}'
        output_path = os.path.join(args.dest_dir, f'train.bpe.{lang}')
        apply_bpe_to_data(bpe, input_path, output_path)

    # Step 3: Get vocabulary for each language from BPE data
    for lang in [args.source_lang, args.target_lang]:
        bpe_data_path = os.path.join(args.dest_dir, f'train.bpe.{lang}')
        vocab_path = os.path.join(args.dest_dir, f'vocab.{lang}')
        with codecs.open(bpe_data_path, encoding='utf-8') as infile, \
             open(vocab_path, 'w', encoding='utf-8') as outfile:
            get_vocab(infile, outfile)

    # Step 4: Re-apply BPE to all data with vocabulary filter
    for lang in [args.source_lang, args.target_lang]:
        vocab_path = os.path.join(args.dest_dir, f'vocab.{lang}')
        # Load vocabulary
        vocab = {}
        with codecs.open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                word, freq = line.strip().split()
                vocab[word] = int(freq)

        for split in ['train', 'valid', 'test', 'tiny_train']:
            input_file = f'{args.train_prefix}.{lang}' if split == 'train' else f'{args.valid_prefix}.{lang}' \
                if split == 'valid' else f'{args.test_prefix}.{lang}' if split == 'test' else f'{args.tiny_train_prefix}.{lang}'
            output_file = os.path.join(args.dest_dir, f'{split}.bpe.{lang}')
            vocab_path = os.path.join(args.dest_dir, f'vocab.{lang}')
            apply_bpe_with_vocab_filter(bpe, input_file, output_file, vocab_path, args.vocab_threshold)

    # Step 5: Build dictionaries and binarize datasets for both languages
    logging.info("Building dictionaries and binarizing datasets...")
    for lang in [args.source_lang, args.target_lang]:
        # Dictionary will be built for the training data and then used for all other splits
        #dictionary = Dictionary()
        dict_path = os.path.join(args.dest_dir, f'dict.{lang}')

        # Build and finalize the dictionary using only the training data
        bpe_train_data_path = os.path.join(args.dest_dir, f'train.bpe.{lang}')
        bpe_dict = build_bpe_dictionary(bpe_train_data_path)
        bpe_dict.finalize(threshold=args.threshold_src if lang == args.source_lang else args.threshold_tgt,
                          num_words=args.num_words_src if lang == args.source_lang else args.num_words_tgt)
        bpe_dict.save(dict_path)

        # Binarize all data splits using the finalized dictionary
        for split in ['train', 'valid', 'test', 'tiny_train']:
            bpe_data_path = os.path.join(args.dest_dir, f'{split}.bpe.{lang}')
            bin_data_path = os.path.join(args.dest_dir, f'{split}.bin.{lang}')
            logging.info(f"Binarizing {split} data for {lang} and saving to {bin_data_path}")
            make_binary_dataset(bpe_data_path, bin_data_path, bpe_dict)

if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        utils.init_logging(args)
    main(args)
