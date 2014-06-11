"""Plugins for the Train object."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import functools
import logging

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from theano import function

logger = logging.getLogger(__name__)


class TrainExtension(object):
    """
    An object called by pylearn2.train.Train at various
    points during learning.
    Useful for adding custom features to the basic learning
    procedure.

    This base class implements all callback methods as no-ops.
    To add a feature to the Train class, implement a subclass of this
    base class that overrides any subset of these no-op methods.
    """

    def on_save(self, model, dataset, algorithm):
        """
        Train calls this immediately before it saves the model.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained.

        dataset : pylearn2.datasets.Dataset
            The dataset object used for training.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
        """

    def on_monitor(self, model, dataset, algorithm):
        """
        Train calls this immediately after each call to the Monitor
        (i.e., when training begins, and at the end of each epoch).

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained

        dataset : pylearn2.datasets.Dataset
            The dataset object being trained.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
        """

    def setup(self, model, dataset, algorithm):
        """
        Train calls this immediately upon instantiation,
        before any monitoring is done.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained.

        dataset : pylearn2.datasets.Dataset
            The dataset object being trained.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
        """


class BLEU(TrainExtension):
    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        super(BLEU, self).__init__(**kwargs)

    def setup(self, model, dataset, algorithm):
        self.epoch = 0
        X = model.get_input_space().make_theano_batch()
        self.score_func = function([X], model.fprop(X))
        plt.ion()

    def score(self, dataset, indices):
        stats = np.sum(dataset.bleu_stats[indices], axis=0)
        return dataset.bleu(stats)

    def get_best(self, dataset):
        indices = []
        for i in range(dataset.num_sentences):
            y = self.score_func(
                dataset.X[dataset.mapping[i]:dataset.mapping[i + 1]]
            )
            indices.append(np.argmax(y.flatten()) + dataset.mapping[i])
        return indices

    def plot(self, dataset):
        plt.clf()
        indices = np.random.choice(range(100000), 5000, False)
        plt.scatter(dataset.y[indices], self.score_func(dataset.X)[indices])
        plt.axis([0, 1, 0, 1])
        plt.title('Epoch %s' % self.epoch)
        plt.draw()

    def on_monitor(self, model, dataset, algorithm):
        for dataset in self.datasets:
            best_indices = self.get_best(dataset)
            if self.epoch % 1 == 0:
                print "        Dataset " + dataset.name + ":"
                print "            BLEU: " + str(
                    100 * self.score(dataset, best_indices)
                )
                print "            Average rank: " + str(
                    np.mean(best_indices - dataset.mapping[:-1])
                )
                print "            Pearson coefficient: " + str(
                    pearsonr(dataset.y, self.score_func(dataset.X))[0][0]
                )
        # self.plot(dataset)
        self.epoch += 1


class SharedSetter(TrainExtension):
    """
    Sets shared variables to take on the specified values after the
    specified amounts of epochs have taken place.

    epoch_updates = [ [i, x, y] ]

    means run x.set_value(cast(y))

    after i epochs have passed.

    Parameters
    ----------
    epoch_updates : WRITEME
    """

    def __init__(self, epoch_updates):
        self._count = 0
        self._epoch_to_updates = {}
        self._vars = set([])
        for update in epoch_updates:
            epoch, var, val = update
            self._vars.add(var)
            if epoch not in self._epoch_to_updates:
                self._epoch_to_updates[epoch] = []
            assert hasattr(var, 'get_value')
            assert var.name is not None
            self._epoch_to_updates[epoch].append((var,val))

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        # TODO: write more specific docstring
        if self._count == 0:
            monitor = model.monitor
            # TODO: make Monitor support input-less channels so this hack
            # isn't necessary
            hack = monitor.channels.values()[0]
            for var in self._vars:
                monitor.add_channel(name=var.name, val=var,
                                    ipt=hack.graph_input, dataset=hack.dataset)


        if self._count in self._epoch_to_updates:
            for update in self._epoch_to_updates[self._count]:
                var, val = update
                var.set_value(np.cast[var.dtype](val))
        self._count += 1

class ChannelSmoother(TrainExtension):
    """
    Makes a smoothed version of a monitoring channel by averaging together
    the k most recent values of that channel.
    This is a little bit dangerous because if other TrainExtensions depend
    on the channel being up to date they must appear after this one in the
    extensions list. A better long term solution would be to make the Monitor
    support this kind of channel directly instead of hacking it in.
    Note that the Monitor will print this channel as having a value of -1, and
    then the extension will print the right value.

    Parameters
    ----------
    channel_to_smooth : WRITEME
    channel_to_publish : WRITEME
    k : WRITEME
    """

    def __init__(self, channel_to_smooth, channel_to_publish, k=5):
        self.__dict__.update(locals())
        del self.self

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        # TODO: more specific docstring
        monitor = model.monitor
        channels = monitor.channels
        channel_to_smooth = channels[self.channel_to_smooth]
        ipt = channel_to_smooth.graph_input
        dataset = channel_to_smooth.dataset

        monitor.add_channel(name=self.channel_to_publish,
                ipt=ipt,
                val=-1.,
                dataset=dataset)

        self.in_ch = channel_to_smooth
        self.out_ch = channels[self.channel_to_publish]

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        # TODO: write more specific docstring
        val_record = self.in_ch.val_record

        start = max(0, len(val_record) - self.k + 1)
        values = val_record[start:]
        mean = sum(values) / float(len(values))

        self.out_ch.val_record[-1] = mean
        logger.info('\t{0}: {1}'.format(self.channel_to_publish, mean))
