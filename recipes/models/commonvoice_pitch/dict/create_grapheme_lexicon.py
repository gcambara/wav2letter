''' FISHER CALLHOME LEXICON PROCESSOR
Adapt Kaldi's Lexicon to a Lexicon using graphemes.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Fisher Grapheme Lexicon creation.") 
parser.add_argument("--kaldi_lexicon", help="path to Kaldi's lexicon", default="./kaldi_lexicon.txt")
parser.add_argument("--output_lexicon", help="path to output lexicon", default="./grapheme_lexicon.txt")
parser.add_argument("--output_tokens", help="path to output tokens", default="./grapheme_tokens.txt")

args = parser.parse_args()

lexicon_file = args.kaldi_lexicon

lexicon_words = {}
special_words = ['[noise]', '[laughter]', '[sil]', '[oov]']
unknown_word = '<unk>'

with open(args.kaldi_lexicon) as f:
    lines = f.readlines()

for line in lines:
    for w in line.split():
        lexicon_words[w] = True

with open(args.output_lexicon, "w") as f:
    for w in lexicon_words.keys():
        f.write(w)
        f.write("\t")
        if w in special_words:
            f.write(w.replace('[', '').replace(']', ''))
        elif w == unknown_word:
            f.write('oov')
        else:
            f.write(" ".join(w))
        f.write(" |\n")
sys.stdout.write("Done !\n")

print("Generating tokens.txt for acoustic model training", flush=True)
with open(args.output_tokens, "w") as f_tokens:
    f_tokens.write("|\n")
    f_tokens.write("oov\n")
    for alphabet in range(ord("a"), ord("z") + 1):
        f_tokens.write(chr(alphabet) + "\n")
    # Add vowels with closed tilde
    f_tokens.write("á\n")
    f_tokens.write("é\n")
    f_tokens.write("í\n")
    f_tokens.write("ó\n")
    f_tokens.write("ú\n")
    f_tokens.write("ü\n")
    f_tokens.write("ñ\n")


