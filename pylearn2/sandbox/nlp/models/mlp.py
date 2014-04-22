import numpy as np
import theano.tensor as T
from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from theano.compat.python2x import OrderedDict


class HierarchicalSoftmax(Layer):
    """
    Hierarchical softmax layer
    """
    def __init__(self, dim, layer_name, num_classes, irange=None, istdev=None,
                 return_class_probs=False, supervised=None):
        """
        This layer expects one of two inputs: Either a VectorSpace, based on
        which the word class probabilities are calculated, after which the
        word probabilities for the class with the highest probability are
        calculated. It can also take a CompositeSpace consisting of a
        VectorSpace as well as an IndexSpace with 1 dimension, which determines
        the word probabilities to be calculated.

        Parameters
        ----------
        return_class_probs : bool, optional
            If set to True, this layer returns the class probabilities as well
        """
        super(HierarchicalSoftmax, self).__init__()
        self.__dict__.update(locals())
        del self.self

        if return_class_probs:
            self.output_space = CompositeSpace(VectorSpace(dim=num_classes),
                                               VectorSpace(dim=dim))
        else:
            self.output_space = VectorSpace(dim=dim)

        self.b_class = sharedX(np.zeros(num_classes))
        self.b_words = sharedX(np.zeros(num_classes, dim))

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)

        assert self.W_class.ndim == 2
        assert self.W_words.ndim == 3
        assert self.b_class.ndim == 1
        assert self.b_words.ndim == 2

        if self.supervised is None:
            raise ValueError("Unable to determine whether target classes will "
                             "be provided or not. Please pass the 'supervised'"
                             " parameter, or call the set_input_space method")
        elif self.supervised:
            hidden_state = state_below[0]
            target_classes = state_below[1]
        else:
            if isinstance(state_below, tuple):
                hidden_state = state_below[0]
            else:
                hidden_state = state_below

        # Create the Theano expression
        class_probs = T.nnet.softmax(T.dot(hidden_state,
                                           self.W_class) + self.b_class)
        if not self.supervised:
            target_classes = T.argmax(class_probs, axis=0)
        group_dot_op = T.nnet.GroupDot(self.num_classes)
        group_dot = group_dot_op(hidden_state, self.W_words,
                                 self.b_words, target_classes)
        word_probs = T.nnet.softmax(group_dot)

        if self.return_class_probs:
            return (class_probs, word_probs)
        else:
            return word_probs

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, CompositeSpace):
            assert self.supervised is None or self.supervised is True
            self.supervised = True
            hidden_state_space, target_class_space = space.components
            if not isinstance(hidden_state_space, VectorSpace):
                hidden_state_space, target_class_space = \
                    target_class_space, hidden_state_space
            assert isinstance(hidden_state_space, VectorSpace)
            assert isinstance(target_class_space, IndexSpace)
        elif isinstance(space, VectorSpace):
            assert self.supervised is None or self.supervised is False
            self.supervised = False
            hidden_state_space = space
        else:
            raise TypeError("Expected CompositeSpace or VectorSpace, got " +
                            str(space) + " of type " + str(type(space)))

        # Create weight matrices
        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W_class = rng.uniform(-self.irange, self.irange,
                                  (self.input_dim, self.num_classes))
            W_words = rng.uniform(-self.irange, self.irange,
                                  (self.num_classes,
                                   self.input_dim, self.dim))
        elif self.istdev is not None:
            assert self.sparse_init is None
            W_class = rng.rand(-self.irange, self.irange,
                               (self.input_dim,
                                self.num_classes)) * self.istdev
            W_words = rng.randn(-self.irange, self.irange,
                                (self.num_classes, self.input_dim,
                                 self.dim)) * self.istdev

        self.W_class = sharedX(W_class, 'hierarchical_softmax_W_class')
        self.W_words = sharedX(W_words, 'hierarchical_softmax_W_words')

        self._params = [self.b_words, self.b_class,
                        self.W_words, self.W_class]


class CompositeLayer(Layer):
    """
    A Layer that runs several layers in parallel. Its default behavior
    is to pass the layer input Space to each of the components. Alternatively,
    it can take a CompositeSpace as an input and a mapping from inputs
    to layers i.e. providing each component layer with a subset of the
    inputs.

    Parameters
    ----------
    layer_name : string
        The name of this layer
    layers : tuple or list
        The component layers to run in parallel.
    inputs_to_components : None or dict mapping int to list of int
        Should be None unless the input space is a CompositeSpace
        If inputs_to_components[i] contains j, it means input i will
        be given as input to component j.
        If an input does not appear in the dictionary, it will be given
        to all components.
    """

    def __init__(self, layer_name, layers, inputs_to_layers=None):
        for layer in layers:
            assert isinstance(layer, Layer)
        if inputs_to_layers is None:
            self.inputs_to_layers = None
        else:
            if not isinstance(inputs_to_layers, dict):
                raise TypeError("CompositeLayer expected inputs_to_layers to "
                                "be dict, got " + str(type(inputs_to_layers)))
            self.inputs_to_layers = OrderedDict()
            for key in inputs_to_layers:
                assert isinstance(key, int)
                assert key >= 0
                value = inputs_to_layers[key]
                assert isinstance(value, list)
                assert all([isinstance(elem, int) for elem in value])
                assert min(value) >= 0
                assert max(value) < self.num_layers
                self.inputs_to_layers[key] = list(value)
        super(CompositeLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.num_layers = len(layers)

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if not isinstance(space, CompositeSpace):
            if self.inputs_to_layers is not None:
                raise ValueError("CompositeLayer received an inputs_to_layers "
                                 "mapping, but does not have a CompositeSpace "
                                 "as its input space, so there is nothing to "
                                 "map. Received " + str(space) + " as input "
                                 "space instead.")
            self.routing_needed = False
        else:
            if self.inputs_to_layers is None:
                self.routing_needed = False
            else:
                self.routing_needed = True
                if not max(self.inputs_to_layers) < space.num_layers:
                    raise ValueError("The inputs_to_layers mapping of "
                                     "CompositeSpace contains they key " +
                                     str(max(self.inputs_to_layers)) + " "
                                     "(0-based) but the input space only "
                                     "contains " + str(self.num_layers) + " "
                                     "layers.")
                # Invert the dictionary
                self.layers_to_inputs = OrderedDict()
                for i in xrange(self.num_layers):
                    inputs = []
                    for j in xrange(space.num_layers):
                        if i in self.inputs_to_layers[j]:
                            inputs.append(i)
                    if len(inputs) < space.num_layers:
                        self.layers_to_inputs[i] = inputs
        for i, layer in enumerate(self.layers):
            if self.routing_needed and i in self.layers_to_inputs:
                cur_space = space.restrict(self.layers_to_inputs[i])
            else:
                cur_space = space

            layer.set_input_space(cur_space)

        self.input_space = space
        self.output_space = CompositeSpace(tuple(layer.get_output_space()
                                                 for layer in self.layers))

    @wraps(Layer.get_params)
    def get_params(self):

        rval = []

        for layer in self.layers:
            rval = safe_union(layer.get_params(), rval)

        return rval

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        rvals = []
        state_below = list(state_below)
        for i, layer in enumerate(self.layers):
            if self.routing_needed and i in self.layers_to_inputs:
                cur_state_below = state_below[self.layers_to_inputs[i]]
            else:
                cur_state_below = state_below

            rvals.append(layer.fprop(cur_state_below))
        return tuple(rvals)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        return sum(layer.cost(Y_elem, Y_hat_elem)
                   for layer, Y_elem, Y_hat_elem in
                   safe_zip(self.layers, Y, Y_hat))

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):

        super(CompositeLayer, self).set_mlp(mlp)
        for layer in self.layers:
            layer.set_mlp(mlp)


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
