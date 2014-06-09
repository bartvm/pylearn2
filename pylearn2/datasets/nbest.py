from collections import Counter
import linecache
import logging
import os

from clint.textui import progress
import numpy as np
from sklearn.decomposition import PCA
import theano

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import py_integer_types
from pylearn2.utils.iteration import resolve_iterator_class


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
    word_normalize : bool
        If True, divides all features by the hypothesis length
    sphere : bool
        If True, spheres the data by subtracting the mean and
        dividing by the stdev
    pca : int
        If > 0 then perform whitened PCA and retain
        this number of components
    weighted_bleu : bool
        If True, weight the BLEU scores by the counts
        according to MERT
    max_margin : bool
        If True, the targets become maximum margin classifiers instead
    """
    def __init__(self, nbest_file=None, reference_file=None, sphere_y=False,
                 sphere=False, pca=0, weighted_bleu=False,
                 max_margin=False):
        self.max_margin = max_margin

        # Reading data from cache or text file
        root, ext = os.path.splitext(nbest_file)
        self.name = root
        if (os.path.isfile(root + '.bleu.npy') and
                os.path.isfile(root + '.features.npy') and
                os.path.isfile(root + '.mapping.npy')):
            log.info("Loading n-best from cache")
            self.X = np.load(root + '.features.npy')
            self.bleu_stats = np.load(root + '.bleu.npy')
            self.mapping = np.load(root + '.mapping.npy')
            self.num_nbest = len(self.X)
            self.num_sentences = len(self.mapping) - 1
        else:
            self.read_nbest(nbest_file, reference_file)

        # Processing features
        if pca:
            if isinstance(pca, NBest):
                self.X = pca.pca.transform(self.X)
            elif isinstance(pca, py_integer_types):
                log.info("Performing PCA...")
                self.pca = PCA(n_components=pca)
                self.pca.fit(self.X)
                self.X = self.pca.transform(self.X)
            else:
                raise ValueError("Expected NBest or True for pca")
        if sphere:
            if isinstance(sphere, NBest):
                self.X -= sphere.X_mean
                self.X /= sphere.X_std
            elif sphere is True:
                self.X_mean = self.X.mean(axis=0)
                self.X_std = self.X.std(axis=0)
                self.X -= self.X_mean
                self.X /= self.X_std
            else:
                raise ValueError("Expected NBest or True for sphere_y")

        # Calculating targets
        self.y = np.zeros((self.num_nbest, 1), dtype=theano.config.floatX)
        if not self.max_margin:
            log.info("Calculating targets..")
            if weighted_bleu:
                if isinstance(weighted_bleu, NBest):
                    best_stats = np.sum(
                        weighted_bleu.bleu_stats[weighted_bleu.mapping[:-1]],
                        axis=0
                    )
                elif weighted_bleu is True:
                    best_stats = np.sum(self.bleu_stats[self.mapping[:-1]],
                                        axis=0)
            for i, stats in enumerate(self.bleu_stats):
                if weighted_bleu:
                    self.y[i] = self.sentence_bleu(stats, best_stats)
                else:
                    self.y[i] = self.sentence_bleu(stats)
            if sphere_y:
                if isinstance(sphere_y, NBest):
                    self.y -= sphere_y.y_mean
                    self.y /= sphere_y.y_std
                elif sphere_y is True:
                    self.y_mean = self.y.mean()
                    self.y_std = self.y.std()
                    self.y -= self.y_mean
                    self.y /= self.y_std
                else:
                    raise ValueError("Expected NBest or True for sphere_y")

        # Printing info on best targets
        indices = []
        for i in range(self.num_sentences):
            indices.append(
                np.argmax(self.y[self.mapping[i]:self.mapping[i + 1]])
            )
        indices = self.mapping[:-1] + np.asarray(indices, dtype='uint32')
        stats = self.bleu_stats[indices].sum(axis=0)
        log.info("Optimal BLEU (" + root + "): " + str(100 * self.bleu(stats)))
        log.info("MERT BLEU (" + root + "): " + str(100 * self.bleu(
            self.bleu_stats[self.mapping[:-1]].sum(axis=0)
        )))

        super(NBest, self).__init__(X=self.X, y=self.y)
        self._iter_subset_class = resolve_iterator_class(
            'shuffled_sequential'
        )

    def read_nbest(self, nbest_file, reference_file):
        with open(reference_file) as f:
            for num_sentences, _ in enumerate(f):
                pass
            self.num_sentences = num_sentences + 1
        with open(nbest_file) as f:
            for num_nbest, _ in enumerate(f):
                pass
            self.num_nbest = num_nbest + 1
        self.mapping = np.zeros((self.num_sentences + 1,), dtype='uint32')
        X = []
        bleu_stats = []
        with progress.Bar(label="Reading n-best list ",
                          expected_size=self.num_nbest) as bar:
            with open(nbest_file) as f:
                first_line = None
                for i, line in enumerate(f):
                    fields = [field.strip() for field in line.split('|||')]
                    if first_line is None:
                        first_line = int(fields[0])
                    sentence_index = int(fields[0]) - first_line
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

        # Creating NumPy arrays
        self.mapping = np.cumsum(self.mapping).astype('uint32')
        self.X = np.asarray(X, dtype=theano.config.floatX)
        self.bleu_stats = np.asarray(bleu_stats, dtype='uint32')

        # Saving NumPy arrays to disk
        log.info("Caching features and BLEU scores as NPY files")
        root, ext = os.path.splitext(nbest_file)
        np.save(root + '.features.npy', self.X)
        np.save(root + '.bleu.npy', self.bleu_stats)
        np.save(root + '.mapping.npy', self.mapping)

    def get(self, sources, indices):
        if self.max_margin:
            raise ValueError
        else:
            return (self.X[indices], self.y[indices])

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
            bp = min(0., bp) / self.num_sentences
        if best_stats is None:
            log_bleu = sum([np.log((1. + m) / (1. + w))
                            for m, w in zip(stats[2::2], stats[3::2])]) / 4.
        else:
            log_bleu = sum([np.log(float(y * (x + m)) / (x * (y + w)))
                            for x, y, m, w in zip(best_stats[2::2],
                                                  best_stats[3::2],
                                                  stats[2::2],
                                                  stats[3::2])]) / 4.
        return np.exp(log_bleu)

    def bleu(self, stats):
        """
        Calculates the normal BLEU (BLEU4) score for a given sentence or corpus
        """
        h_len, r_len = stats[:2]
        log_bleu = sum([np.log(float(m) / w)
                        for m, w in zip(stats[2::2], stats[3::2])]) / 4.
        return np.exp(min(0, 1 - float(r_len) / h_len) + log_bleu)
