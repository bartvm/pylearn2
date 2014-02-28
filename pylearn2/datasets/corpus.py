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
import os

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
        self._raw_data = self._raw_data[start:stop].astype('int32')
        self._window_size = window_size
        self._X, self._y = self.get_data()
        self._X_space = IndexSpace(dim=self._window_size,
                                   max_labels=max(self._X) + 1)
        self._y_space = IndexSpace(dim=1, max_labels=self._num_target_words + 1)
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
        if isinstance(indices, slice):
            indices = np.arange(indices.start, indices.stop, indices.step)
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

class NRCNNJM(dataset.Dataset):
    """
    Loads the Penn Treebank corpus.
    """
    def __init__(self, which_set="train", src_context_size=11,
                 fname=None, target_context_size=3,
                 random_dropout=False,
                 num_target_words=-1, shuffle=False,
                 mmap_mode=None,
                 data_mode=0,
                 lang="ch",
                 size_type="small",
                 start=0, stop=None):

        #If iter_mode is 0,Just return one tuple otherwise
        #if it is 1 return two tuples for the data
        self.data_mode = data_mode

        path = "/data/lisatmp2/gulcehrc/nnjm/bolt/"
        path = serial.preprocess(path)
        path = os.path.join(path, lang)

        assert mmap_mode in [None, "r+", "r", "w+", "c"]

        npz_data = np.load(os.path.join(path, 'complete_%s.npz' % size_type), mmap_mode=mmap_mode)
        source_max_labels, target_max_labels, targets_max_labels = \
            np.max(npz_data["src_ctxt"]) + 1, np.max(npz_data["tgt_ctxt"]) + 1, np.max(npz_data["tgts"]) + 1
        source_ctxt = npz_data["src_ctxt"][start:stop]
        target_ctxt = npz_data["tgt_ctxt"][start:stop]
        targets = npz_data["tgts"][start:stop]
        self._num_examples = len(targets)

        del npz_data
        self._num_target_words = num_target_words

        if data_mode == 0:
            self._raw_data = np.append(source_ctxt, target_ctxt, axis=1).astype('int32')
            self._window_size = target_context_size + src_context_size
            self._X = self._raw_data
            self._y = targets.astype('int32')

            self._X_space = IndexSpace(dim=self._window_size,
                                       max_labels=max(source_max_labels, target_max_labels))
            self.data_len = self._X.shape[0]

            if num_target_words < 1:
                self._num_target_words = np.max(targets) + 1

            #self._y_space = VectorSpace(self._num_target_words + 1)
            self._y_space = IndexSpace(dim=1, max_labels=self._num_target_words)

            self._data_specs = ((CompositeSpace((self._X_space, self._y_space))),
                                ('features', 'targets'))
        else:
            self.source_ctxt = source_ctxt
            self.target_ctxt = target_ctxt
            self._raw_data = (self.source_ctxt, self.target_ctxt)
            self._X = self._raw_data
            self._y = np.atleast_2d(targets.astype('int32')).T
            self._source_ctxt_space = IndexSpace(dim=src_context_size,
                                                 max_labels=source_max_labels)
            self._target_ctxt_space = IndexSpace(dim=target_context_size,
                    max_labels=target_max_labels)

            self._y_space = IndexSpace(dim=1,
                                       max_labels=targets_max_labels)

            self.data_len = self.source_ctxt.shape[0]

            self._data_specs = (CompositeSpace((
                self._source_ctxt_space,
                self._target_ctxt_space,
                self._y_space)
            ), (
                'source_context',
                'target_context',
                'targets')
            )

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
                if self.data_mode == 0:
                    conv_fn = (lambda batch, self=self, space=sp:
                           self.view_converter.get_formatted_batch(
                               batch, space))
                else:
                    space_ = sp
                    sp_src = space_.components[0]
                    sp_tgt = space_.components[1]
                    conv_fn_src = (lambda batch, self=self, space=sp_src:
                           self.view_converter.get_formatted_batch(
                               batch, space_))
                    conv_fn_tgt = (lambda batch, self=self, space=sp_tgt:
                           self.view_converter.get_formatted_batch(
                               batch, space_))
                    convert.append(conv_fn_src)
                    convert.append(conv_fn_tgt)
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
            num_batches = self._num_examples // batch_size
            # num_batches = getattr(self, '_iter_num_batches', None)

        if rng is None and mode.stochastic:
            rng = self.rng

        return FiniteDatasetIterator(
                self,
                mode(self.data_len,
                     batch_size,
                     num_batches, rng),
                data_specs=data_specs,
                return_tuple=return_tuple,
                convert=convert)

    def has_targets(self):
        return True

    def get_data(self):
        if self.data_mode == 0:
            X = (self._X,)
            Y = (self._y,)
        else:
            X = (self.source_ctxt, self.target_ctxt)
            Y = (self._y,)
        return X + Y

    def get(self, indices):
        if isinstance(indices, slice):
            indices = np.arange(indices.start, indices.stop, indices.step)

        if self.data_mode == 0:
            X_batch = (self._X[indices],)
            y_batch = (self._y[indices],)
        else:
            X_batch = (self.source_ctxt[indices], self.target_ctxt[indices])
            y_batch = (self._y[indices],)

        return X_batch + y_batch

    def get_data_specs(self):
        return self._data_specs

class Corpus(dataset.Dataset):
    def __init__(self, window_size, batch_size=250, num_input_words=None,
                 num_target_words=None, shuffle=True, start=0, stop=None):
        if stop is None:
            stop = len(self._raw_data)
        self._num_target_words = num_target_words
        self._num_input_words = num_input_words
        self._raw_data = self._raw_data[start:stop].astype('int32')
        self._window_size = window_size
        self._X, self._y = self.get_data()
        if self._num_input_words is None:
            self._X_space = IndexSpace(dim=self._window_size,
                                       max_labels=max(self._X) + 1)
        else:
            self._X_space = IndexSpace(dim=self._window_size,
                                       max_labels=self._num_input_words)
        if self._num_target_words is None:
            self._y_space = IndexSpace(dim=1, max_labels=max(self._X) + 1)
        else:
            self._y_space = IndexSpace(dim=1, max_labels=self._num_target_words + 1)
        self._data_specs = ((CompositeSpace((self._X_space, self._y_space))),
                            ('features', 'targets'))
        self._iter_mode = 'random_uniform'
        self._iter_batch_size = batch_size
        self._iter_num_batches = len(self._X) // self._iter_batch_size
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
                self._X.shape[0] - self._window_size,
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
        if isinstance(indices, slice):
            indices = np.arange(indices.start, indices.stop, indices.step)
        X_indices = np.repeat(indices, self._window_size) + \
            np.tile(np.arange(self._window_size), len(indices))
        y_indices = indices + self._window_size
        X_batch = self._X[X_indices].reshape((len(indices), self._window_size))
        y_batch = self._y[y_indices].reshape((len(indices), 1))
        return (X_batch, y_batch)

    def get_data(self):
        X = self._raw_data
        Y = X.copy()
        if self._num_target_words is not None:
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

class Europarl(Corpus):
    def __init__(self, which_set, window_size, batch_size=250, num_input_words=None,
                 num_target_words=None, shuffle=True, start=0, stop=None):
        # 10,000 covers 98.1% of the training set
        path = "${PYLEARN2_DATA_PATH}/europarl-v3b/"
        path = serial.preprocess(path)
        self._vocab_size = 10000
        if which_set == 'train':
            self._raw_data = np.load(path + 'europarl.npy')
        elif which_set == 'valid':
            self._raw_data = np.load(path + 'devtest2006.npy')
        elif which_set == 'test':
            self._raw_data = np.load(path + 'devtest2006.npy')
        else:
            raise ValueError("Dataset must be one of train or valid")
        for sentence in self._raw_data:
            sentence[sentence >= self._vocab_size] = 1
        self._cum_examples = np.cumsum([len(sentence) for sentence in self._raw_data])
        self._cum_examples = np.insert(self._cum_examples, 0, 0)
        self._num_examples = self._cum_examples[-1]
        self._window_size = window_size
        self._X, self._y = self.get_data()
        self._X_space = IndexSpace(dim=self._window_size,
                                   max_labels=self._vocab_size)
        self._y_space = IndexSpace(dim=1, max_labels=self._vocab_size)
        self._data_specs = ((CompositeSpace((self._X_space, self._y_space))),
                            ('features', 'targets'))
        self._iter_mode = 'random_uniform'
        self._iter_batch_size = batch_size
        self._iter_num_batches = len(self._X) // self._iter_batch_size
        self.rng = np.random.RandomState(432)

    def get_data(self):
        X = self._raw_data
        Y = X.copy()
        return (X, Y)

    def get(self, indices):
        if isinstance(indices, slice):
            indices = np.arange(indices.start, indices.stop, indices.step)
        sentence_indices = np.array([np.argmax(index < self._cum_examples) - 1 for index in indices])
        word_indices = indices - self._cum_examples[sentence_indices]
        X_batch = np.empty((len(indices), self._window_size), dtype='int64')
        y_batch = np.empty((len(indices), 1), dtype='int64')
        for i, (word_index, sentence_index) in enumerate(zip(word_indices, sentence_indices)):
            if word_index < self._window_size:
                X_batch[i, :self._window_size - word_index] = np.zeros(self._window_size - word_index, dtype='int64')
                X_batch[i, self._window_size - word_index:] = self._X[sentence_index][:word_index]
                try:
                    y_batch[i] = self._y[sentence_index][word_index + self._window_size]
                except IndexError:
                    y_batch[i] = 0
            else:
                X_batch[i] = self._X[sentence_index][word_index - self._window_size:word_index]
                y_batch[i] = self._y[sentence_index][word_index]
        return (X_batch, y_batch)

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
                self._num_examples,
                batch_size, num_batches, rng
            ), data_specs=data_specs,
            return_tuple=return_tuple,
            convert=convert)
