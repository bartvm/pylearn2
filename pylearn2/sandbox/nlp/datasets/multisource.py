"""
Wrapper for loading HDF files
"""
import tables
import functools

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.cache import LocalDatasetCache
from pylearn2.space import CompositeSpace, IndexSpace, VectorSpace
from pylearn2.utils import string_utils
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

import scipy.sparse


class Multisource(Dataset):
    """
    Loads n-grams from a pytables file. Assumes the last column contains
    the targets, and the other columns the data.
    """
    def __init__(self, ngram_name, bow_name, mapping_name,
                 vocab_size, num_batches=None):
        """
        Parameters
        ----------
        name : string
            The file is assumed to be stored in 'name.hdf' and contain a
            node called 'name' with the data.
        vocab_size : int
            The size of the vocabulary used
        """
        if num_batches is not None:
            self._iter_num_batches = num_batches
        data = {}
        for name in [ngram_name, mapping_name]:
            remote_path = '${PYLEARN2_DATA_PATH}/joint_paper_hs/' + name + '.hdf'
            remote_path = string_utils.preprocess(remote_path)
            local_dataset_cache = LocalDatasetCache()
            local_path = local_dataset_cache.cache_file(remote_path)
            f = tables.openFile(local_path)
            node = f.get_node('/' + name)
            data[name] = node.read()
            f.close()
        remote_path = '${PYLEARN2_DATA_PATH}/joint_paper_hs/' + bow_name + '.hdf'
        remote_path = string_utils.preprocess(remote_path)
        local_dataset_cache = LocalDatasetCache()
        local_path = local_dataset_cache.cache_file(remote_path)
        f = tables.openFile(local_path)
        arrays = {}
        for part in ('data', 'indices', 'indptr', 'shape'):
            node = f.get_node("/%s_%s" % (bow_name, part))
            arrays[part] = node.read()
        data[bow_name] = scipy.sparse.csr_matrix(
            (arrays['data'], arrays['indices'], arrays['indptr']),
            shape=(arrays['shape'][0], arrays['shape'][1] - 1)
        )
        f.close()
        self.ngrams, self.bow, self.mapping = data[ngram_name], data[bow_name], data[mapping_name]

        self.num_examples = len(data[ngram_name])

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):

        mode = resolve_iterator_class(mode)

        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples,
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple)

    def get_data_specs():
        return ((CompositeSpace(IndexSpace(dim=6, max_labels=15000), VectorSpace(dim=15000, sparse=True)), IndexSpace(dim=1, max_labels=15000)), (('ngrams', 'source_sentence'), 'targets'))

    def get(self, sources, indices):
        rval = []
        for source in sources:
            if source == 'ngrams':
                rval.append(self.ngrams[indices, :6])
            if source == 'targets':
                rval.append(self.ngrams[indices, 6:])
            if source == 'source_sentence':
                rval.append(self.bow[self.mapping[indices]])
        return rval
