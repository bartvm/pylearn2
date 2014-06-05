from collections import Counter
import linecache
import logging

from clint.textui import progress
import numpy as np
import theano
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


log = logging.getLogger(__name__)


class NBest(DenseDesignMatrix):
    """
    A dataset based on n-best lists

    Parameters
    ----------
    nbest : filename
        Name of the nbest list in Moses format
    reference : filename
        The reference translation
    """
    def __init__(self, nbest_file=None, reference_file=None):
        if nbest_file is None or reference_file is None:
            raise ValueError
        # Count the number of sentences in the reference translation
        with open(reference_file) as f:
            for num_sentences, _ in enumerate(f):
                pass
            self.num_sentences = num_sentences + 1
        with open(nbest_file) as f:
            for num_nbest, _ in enumerate(f):
                pass
            self.num_nbest = num_nbest + 1
        self.mapping = np.zeros((self.num_sentences + 1,), dtype='uint16')
        X = []
        bleu_stats = []
        with progress.Bar(label="Reading n-best list ",
                          expected_size=self.num_nbest) as bar:
            with open(nbest_file) as f:
                for i, line in enumerate(f):
                    fields = [field.strip() for field in line.split('|||')]
                    sentence_index = int(fields[0])
                    hypothesis = fields[1].split()
                    reference = linecache.getline(reference_file,
                                                  sentence_index + 1).split()
                    raw_scores = fields[2]
                    scores = []
                    for raw_score in raw_scores.split():
                        try:
                            scores.append(float(raw_score))
                        except ValueError:
                            pass
                    X.append(scores)
                    bleu_stats.append(list(self.get_stats(hypothesis,
                                                          reference)))
                    self.mapping[sentence_index + 1] += 1
                    if i % 1000 == 0:
                        bar.show(i)
        assert sentence_index == self.num_sentences - 1
        self.mapping = np.cumsum(self.mapping)
        self.X = np.asarray(X, dtype=theano.config.floatX)
        self.bleu_stats = np.asarray(bleu_stats, dtype='uint32')
        self.y = np.zeros((self.num_nbest, 1), dtype=theano.config.floatX)
        super(NBest, self).__init__(X=self.X, y=self.y)

    def get(self, sources, indices):
        return (self.X[indices], self.y[indices])

    def rescore(self, indices):
        print "Rescoring..."
        best_stats = np.sum(self.bleu_stats[indices], axis=0)
        for i, stats in enumerate(self.bleu_stats):
            # stats are the stats per sentence
            # best_stats are the sum of current best
            self.y[i] = self.sentence_bleu(stats, best_stats)

    def get_stats(self, hypothesis, reference):
        yield len(hypothesis)
        yield len(reference)
        for n in xrange(1, 5):
            h_ngrams = Counter([tuple(hypothesis[i:i + n])
                                for i in xrange(len(hypothesis) + 1 - n)])
            r_ngrams = Counter([tuple(reference[i:i + n])
                                for i in xrange(len(reference) + 1 - n)])
            yield sum((h_ngrams & r_ngrams).values())
            yield max(len(hypothesis) + 1 - n, 0)

    def sentence_bleu(self, stats, best_stats=None, clipped_bp=True):
        """
        Calculates the BLEU+1 with (clipped) brevity penalty
        smoothing for two sentences

        stats : list of ints
            A list with the statistics that `bleu_stats` returns.
        best_stats : list of ints, optional
            If given, use this to calculate the relative score.
            Otherwise it uses BLEU+1
        """
        h_len, r_len = stats[:2]
        bp = 1 - (r_len + 1.) / h_len
        if clipped_bp:
            bp = min(0, bp)
        if best_stats is None:
            log_bleu = sum([np.log((1. + m) / (1. + w))
                            for m, w in zip(stats[2::2], stats[3::2])]) / 4.
        else:
            log_bleu = sum([np.log(float(y * (x + m)) / (x * (y + w)))
                            for x, y, m, w in zip(best_stats[2::2],
                                                  best_stats[3::2],
                                                  stats[2::2],
                                                  stats[3::2])]) / 4.
        return np.exp(bp + log_bleu)

    def bleu(self, stats):
        """
        Calculates the normal BLEU (BLEU4) score for a given sentence or corpus
        """
        h_len, r_len = stats[:2]
        log_bleu = sum([np.log(float(m) / w)
                        for m, w in zip(stats[2::2], stats[3::2])]) / 4.
        return np.exp(min(0, 1 - float(r_len) / h_len) + log_bleu)
