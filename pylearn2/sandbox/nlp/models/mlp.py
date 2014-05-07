"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
import numpy as np
import theano.tensor as T
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print

from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.models import mlp
from pylearn2.models.mlp import Layer
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.utils import py_integer_types


class Softmax(mlp.Softmax):
    """
    An extension of the MLP's softmax layer which monitors
    the perplexity

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    num_target_classes : int, optional
        If set, it is assumed that the targets for this
        layer are labels represented using an IndexSpace.
        This allows for more efficient cost calculation
        and data transfer.

        Note that using more than 1 target_class might be
        slow on GPU because of the missing support for
        fancy indexing in Theano.

        Set the output space to a VectorSpace after training
        if needed using e.g. `softmax.output_space =
        VectorSpace(dim=softmax.output_space.max_labels)`
    """
    def __init__(self, n_classes, layer_name, irange=None,
                 istdev=None,
                 sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None,
                 no_affine=False,
                 max_col_norm=None, init_bias_target_marginals=None,
                 num_target_classes=None):

        super(mlp.Softmax, self).__init__()

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)

        if num_target_classes is None:
            self.output_space = VectorSpace(n_classes)
        else:
            assert isinstance(num_target_classes, int)
            self.output_space = IndexSpace(max_labels=n_classes,
                                           dim=num_target_classes)
        if not no_affine:
            self.b = sharedX(np.zeros((n_classes,)), name='softmax_b')
            if init_bias_target_marginals:
                marginals = init_bias_target_marginals.y.mean(axis=0)
                assert marginals.ndim == 1
                b = pseudoinverse_softmax_numpy(marginals).astype(self.b.dtype)
                assert b.ndim == 1
                assert b.dtype == self.b.dtype
                self.b.set_value(b)
        else:
            assert init_bias_target_marginals is None

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            b = self.b

            Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        if isinstance(self.output_space, IndexSpace):
            if self.output_space.dim == 1:
                rval = rval.argmax(axis=1).dimshuffle(0, 'x')
            else:
                rval = rval.argsort(axis=0)[:, :self.output_space.dim]

        return rval

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(self.output_space, IndexSpace):
            op = owner.inputs[0].owner.inputs[0].owner.op
            owner = owner.inputs[0].owner.inputs[0].owner
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z, = owner.inputs
        assert z.ndim == 2
        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        if isinstance(self.output_space, IndexSpace):
            if self.output_space.dim == 1:
                # This prevents fancy indexing, which would require
                # GPU -> Host transfers
                # log_prob_of = \
                #     log_prob.flatten()[Y.flatten() +
                #                        T.arange(log_prob.shape[0]) *
                #                        self.output_space.max_labels]
                log_prob_of = log_prob[T.arange(log_prob.shape[0]), Y.flatten()]
            else:
                log_prob_of = log_prob[T.arange(log_prob.shape[0]),
                                       Y.T].sum(axis=0)
        else:
            log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()

        return - rval

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(self.output_space, IndexSpace):
            op = owner.inputs[0].owner.inputs[0].owner.op
            owner = owner.inputs[0].owner.inputs[0].owner
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z, = owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        if isinstance(self.output_space, IndexSpace):
            if self.output_space.dim == 1:
                # This prevents fancy indexing, which would require
                # GPU -> Host transfers
                log_prob_of = \
                    log_prob.flatten()[Y.flatten() +
                                       T.arange(log_prob.shape[0]) *
                                       log_prob.shape[1]]
            else:
                log_prob_of = log_prob[T.arange(log_prob.shape[0]),
                                       Y.T].T
        else:
            log_prob_of = (Y * log_prob)

        return -log_prob_of

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):

        rval = OrderedDict()

        if isinstance(self.output_space, VectorSpace):
            mx = state.max(axis=1)
            rval.update(OrderedDict([('mean_max_class', mx.mean()),
                                     ('max_max_class', mx.max()),
                                     ('min_max_class', mx.min())]))

        if target is not None:
            if isinstance(self.output_space, IndexSpace):
                misclass = T.neq(state, target).mean()
                misclass = T.cast(misclass, config.floatX)
            else:
                y_hat = T.argmax(state, axis=1)
                y = T.argmax(target, axis=1)
                misclass = T.neq(y, y_hat).mean()
                misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['ppl'] = 2 ** (rval['nll'] / T.log(2))

        return rval


class ProjectionLayer(Layer):
    """
    This layer can be used to project discrete labels into a continous space
    as done in e.g. language models. It takes labels as an input (IndexSpace)
    and maps them to their continous embeddings and concatenates them.
    """
    def __init__(self, dim, layer_name, irange=None, istdev=None):
        """
        Initializes a projection layer.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings. Note that this means that the
            output dimension is (dim * number of input labels)
        layer_name : string
            Layer name
        irange : numeric
           The range of the uniform distribution used to initialize the
           embeddings. Can't be used with istdev.
        istdev : numeric
            The standard deviation of the normal distribution used to
            initialize the embeddings. Can't be used with irange.
        """
        super(ProjectionLayer, self).__init__()
        self.dim = dim
        self.layer_name = layer_name
        if irange is None and istdev is None:
            raise ValueError("ProjectionLayer needs either irange or"
                             "istdev in order to intitalize the projections.")
        elif irange is not None and istdev is not None:
            raise ValueError("ProjectionLayer was passed both irange and "
                             "istdev but needs only one")
        else:
            self._irange = irange
            self._istdev = istdev

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if isinstance(space, IndexSpace):
            self.input_dim = space.dim
            self.input_space = space
        else:
            raise ValueError("ProjectionLayer needs an IndexSpace as input")
        self.output_space = VectorSpace(self.dim * self.input_dim)
        rng = self.mlp.rng
        if self._irange is not None:
            W = rng.uniform(-self._irange,
                            self._irange,
                            (space.max_labels, self.dim))
        else:
            W = rng.randn(space.max_labels, self.dim) * self._istdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        z = self.transformer.project(state_below)
        return z

    @wraps(Layer.get_params)
    def get_params(self):
        W, = self.transformer.get_params()
        assert W.name is not None
        params = [W]
        return params
