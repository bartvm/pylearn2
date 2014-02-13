"""
Class to read the PennTreebank and create training samples from missing values.
"""
__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = ["Bart van Merrienboer"]
__license__ = "3-clause BSD"
__maintainer__ = "Bart van Merrienboer"
__email__ = "vanmerb@iro"

import numpy as np
from pylearn2.datasets import dataset
from pylearn2.utils import serial, safe_zip
from pylearn2.utils.iteration import (
    resolve_iterator_class,
    FiniteDatasetIterator
)
from pylearn2.space import IndexSpace, CompositeSpace
import functools

class PennTreebank(dataset.Dataset):
    """
    Loads the Penn Treebank corpus.
    """
    def __init__(self, which_set, window_size,
                 num_target_words, shuffle=True, start=0, stop=None):
        path = "${PYLEARN2_DATA_PATH}/PennTreebankCorpus/"
        path = serial.preprocess(path)
        npz_data = np.load(path + 'penntree_char_and_word.npz')
        if which_set == 'train':
            self._raw_data = npz_data['train_words']
        elif which_set == 'valid':
            self._raw_data = npz_data['valid_words']
        elif which_set == 'test':
            self._raw_data = npz_data['test_words']
        else:
            raise ValueError("Dataset must be one of train, valid or test")
        del npz_data
        if stop is None:
            stop = len(self._raw_data)
        self._num_target_words = num_target_words
        self._raw_data = self._raw_data[start:stop]
        self._window_size = window_size
        self._X, self._y = self.get_data()
        self._X_space = IndexSpace(dim=self._window_size,
                                   max_labels=max(self._X) + 1)
        self._y_space = IndexSpace(dim=1, max_labels=self._num_target_words)
        self._data_specs = ((CompositeSpace((self._X_space, self._y_space))),
                            ('features', 'targets'))
        self._iter_mode = 'random_uniform'
        self._iter_batch_size = 50
        self._iter_num_batches = 100
        self.rng = np.random.RandomState(432)

    @functools.wraps(dataset.Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if data_specs is None:
            data_specs = self._data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            if (src == 'features' and
                    getattr(self, 'view_converter', None) is not None):
                conv_fn = (lambda batch, self=self, space=sp:
                           self.view_converter.get_formatted_batch(
                               batch,
                               space))
            else:
                conv_fn = None
            convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_mode'):
                mode = self._iter_mode
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(
            self,
            mode(
                self._X.shape[0] - self._window_size + 1,
                batch_size, num_batches, rng
            ), data_specs=data_specs,
            return_tuple=return_tuple,
            convert=convert)

    def get_shortlist(self, data, num_target_words):
        count = np.bincount(data)
        num_target_words = min(len(count), num_target_words)
        cutoff = np.sort(count)[-num_target_words]
        shortlist = np.argwhere(count >= cutoff)[:num_target_words]
        return shortlist.flatten()

    def has_targets(self):
        return True

    def get(self, indices):
        X_indices = np.repeat(indices, self._window_size) + \
            np.tile(np.arange(self._window_size), len(indices))
        y_indices = indices + self._window_size
        X_batch = self._X[X_indices].reshape((len(indices), self._window_size))
        y_batch = self._y[y_indices].reshape((len(indices), 1))
        return (X_batch, y_batch)

    def get_data(self):
        X = self._raw_data
        Y = X.copy()
        # All the words not in the shortlist get 0, the others are re-indexed
        shortlist = self.get_shortlist(X, self._num_target_words)
        replace = dict(zip(list(shortlist), range(1, len(shortlist) + 1)))
        mp = np.arange(max(Y) + 1)
        mp[replace.keys()] = replace.values()
        Y = mp[Y]
        Y[np.in1d(X, shortlist, invert=True)] = 0
        return (X, Y)

    def get_data_specs(self):
        return self._data_specs
