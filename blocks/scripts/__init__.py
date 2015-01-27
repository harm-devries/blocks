import sys
from collections import Counter

import dill


def continue_training(infile, rec_limit=None):
    if rec_limit:
        sys.setrecursionlimit(rec_limit)
    mainfileloop = dill.load(infile)
    mainfileloop.run()


def count_tokens(infile, outfile, token='word'):
    """Count words or characters.

    The output will be a file with words ordered by their count.

    Parameters
    ----------
    infile : file handle
        The file handle to read data from.
    outfile : file handle
        The file handle to write the word counts to.
    token : 'word' or 'character'
        Whether to count words or individual characters

    """
    if token not in ('word', 'character'):
        raise ValueError
    counter = Counter()
    for line in infile:
        if token == 'word':
            counter.update(line.split())
        else:
            counter.update(line.strip())
    infile.close()
    for token, count in counter.most_common():
        outfile.write("{} {}\n".format(count, token))
    outfile.close()


def create_vocabulary(infile, outfile, unk='<UNK>', eos=None, bos=None,
                      limit=None):
    vocabulary = {}
    index = 0
    for token in (unk, eos, bos):
        if token:
            vocabulary[token] = index
            index += 1
    for line in infile:
        if limit == index:
            break
        count, token = line.rstrip().split(maxsplit=1)
        vocabulary[token] = index
        index += 1
    infile.close()
    dill.dump(vocabulary, outfile)
    outfile.close()
