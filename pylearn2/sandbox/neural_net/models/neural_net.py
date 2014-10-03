import abc
from collections import defaultdict
from itertools import chain, izip
import logging

import numpy as np
import theano
from theano import tensor
from theano.compat.python2x import OrderedDict

from pylearn2.initialization import Constant
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace, Space, VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.data_specs import is_flat_specs

log = logging.getLogger(__name__)


class Layer(Model):
    """
    Abstract base class for neural network layers. All layers must inherit
    from this class and overwrite the required methods. Note that although
    abstract methods must be overwritten, some still provide implementations
    that can be used e.g.

    >>> super(LayerImplementation, self).get_params()

    These defaults can lead to wrong behaviour though when they are inherited
    unknowingly, so that's why they must still be called explicitly from the
    child class.

    Layers take as an input a simple Space or a non-nested CompositeSpace.
    They can output a Simple space or non-nested CompositeSpace as well.

    Parameters
    ----------
    name : str
        This name must be unique within the NeuralNet that this particular
        layer is a member of.

    Notes
    -----
    Using property method instead of decorator because it allows child classes
    to call the parent's setter method (http://bugs.python.org/issue14965)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.name = name

    def __repr__(self):
        """
        Replicates Python's native object string representation, but
        adds the name, to make it easier to identify layers.
        """
        return '<{}.{}(name={}) at {}>'.format(self.__class__.__module__,
                                               self.__class__.__name__,
                                               self.name, hex(id(self)))

    def get_layer_monitoring_channels(self, state, targets=None):
        return OrderedDict()

    def fprop(self, state_below, space, **kwargs):
        """
        This method applies model extensions before and after the _fprop method
        (applying dropout, aggregating weight decays, etc.)

        Parameters
        ----------
        See _fprop

        Notes
        -----
        TODO
            Dropout for recurrent layers might be different for input weights
            and recurrent weights (see http://arxiv.org/pdf/1312.4569.pdf).
            Separate out the input, output and recurrent weights i.e.
            Linear/Recurrent/Linear, so that their dropout can be
            controlled independently.

            Should the same be done for e.g. Tanh, Sigmoid, etc.?

        TODO
            Change Dropout parameters during runtime? Same for weight noise.
            Use model extension to set parameters in shared values. Use
            train extension to update these.
        """
        for extension in self.extensions:
            state_below, space, kwargs = \
                extension.pre_modify_fprop(self, state_below, space,
                                           **kwargs)
        state_below, space, kwargs = self._fprop(state_below,
                                                 space, **kwargs)
        for extension in self.extensions:
            state_below, space, kwargs = \
                extension.post_modify_fprop(self, state_below, space,
                                            **kwargs)
        return state_below, space, kwargs

    @abc.abstractmethod
    def _fprop(self, state_below, space, **kwargs):
        """
        Performs the actual forward-propogating part of the layer.

        Parameters
        ----------
        state_below : theano.gof.Variable
            The input from a previous layer or data source to modify.
        space : Space
            The space in which the input data lies.
        **kwargs : dict
            A dictionary of extra information. This can be used to pass
            information around from layer to layer. Each layer can modify the
            dictionary in place, and is expected to pass it on to the next
            layer. Should be used sparingly.

        Notes
        -----
        Note that space is passed here, and it is not necessarily the same as
        self.input_space. input_space and output_space are used by layers to
        initialize their parameters. However, they might still be able to deal
        with multiple kinds of inputs, and based on their output_mode (and the
        space passed to fprop) they can produce multiple types of output.

        TODO
            Consider dropping kwargs?
        """
        raise NotImplementedError

    def cost(self, Y, Y_hat):
        raise NotImplementedError

    def get_params(self):
        """
        All layers must conform to the convention that parameters are stored in
        self.params as a list. Weights and biases in particular are stored in
        self.weights and self.biases, with self.params taking the form

        >>> self.params = self.weights + self.biases + [other, params]
        """
        if hasattr(self, '_params'):
            return self._params
        else:
            raise AttributeError  # TODO Add informative error message?

    def set_params(self, params):
        """
        `params` *must* be a list of the form [weights] + [biases] + [other].

        Parameters
        ----------
        params : list of theano.compile.SharedVariable
            List of shared varaibles.

        Notes
        -----
        This should generally only be called by a layer directly, when
        re-initializing all the layer parameters. Otherwise one needs to be
        careful to make sure the changes are actually propogated.

        For example, weight matrices of layers which use transformers are
        actually stored on the transformer itself. Changing it here might
        not have effect on the weight matrix actually used. Moreover, nothing
        will train, because the weight matrix uses won't be passed to the
        training algorithm.
        """
        assert isinstance(params, list)
        assert all(isinstance(param, theano.compile.SharedVariable)
                   for param in params)
        self._params = params

    params = property(get_params, set_params)

    def get_param_vals(self):
        if hasattr(self, '_params'):
            return [param.get_value() for param in self.params]
        else:
            raise AttributeError  # TODO Add informative error message?

    def set_param_vals(self, param_vals):
        """
        Parameters
        ----------
        param_vals : list of ndarrays/Nones
            If None, it won't touch the parameter, otherwise it will set
            a new value
        """
        for param_val, param in safe_izip(param_vals, self.params):
            if param_val is not None:
                param.set_value(param_val)

    param_vals = property(get_param_vals, set_param_vals)

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        assert isinstance(weights, list)
        assert all(isinstance(weight, theano.compile.SharedVariable)
                   for weight in weights)
        self._weights = weights

    weights = property(get_weights, set_weights)

    def get_biases(self):
        return self._biases

    def set_biases(self, biases):
        assert isinstance(biases, list)
        assert all(isinstance(weight, theano.compile.SharedVariable)
                   for weight in biases)
        self._biases = biases

    biases = property(get_biases, set_biases)

# TODO Factor out into ModelExtension?
#     @abc.abstractmethod
#     def get_weight_decay(self, coeff):
#         return tensor.constant(0, dtype=config.floatX)
#
#     @abc.abstractmethod
#     def get_l1_weight_decay(self, coeff):
#         return tensor.constant(0, dtype=config.floatX)

    def get_input_space(self):
        return self._input_space

    def set_input_space(self, space):
        """
        Setting the input space resets its parameters. This method *must*
        set self.params, self.weights and self.biases. If the layer does not
        use weights or biases, it must set them to an empty list.
        """
        assert isinstance(space, Space)
        if hasattr(self, '_input_space'):
            log.warning("Changing the input space of {} is likely reset its "
                        "parameters".format(self))
        self._input_space = space

    input_space = abc.abstractproperty(get_input_space, set_input_space)

    def get_output_space(self):
        return self._output_space

    def set_output_space(self, space):
        """
        The output space is not necessarily what the fprop will actually
        produce (that depends on the output_mode and potentially on what
        space was given to the fprop). However, it should be representative
        of the data in the sense that layers which take this layer as an input
        know how to initialize their parameters.
        """
        assert isinstance(space, Space)
        self._output_space = space

    output_space = property(get_output_space, set_output_space)

    def get_rng(self):
        if hasattr(self, '_rng'):
            return self._rng
        else:
            log.warn("No RNG set for {}. Using default seed.".format(self))
            return np.random.RandomState(seed=[2014, 10, 1])

    def set_rng(self, rng):
        assert isinstance(rng, np.random.RandomState)
        self._rng = rng

    rng = property(get_rng, set_rng)


# class TransformerWeightDecay(Layer):
#     """
#     Mixin class that can be inherited to provide get_weight_decay and
#     get_l1_weight_decay methods for all Layers which use a transformer
#     class. Inherit this class first, else MRO will give problems!
#     """
#     def get_weight_decay(self, coeff):
#         assert isinstance(coeff, py_float_types)
#         W, = self.transformer.get_params()
#         return coeff * tensor.sqr(W).sum()
#
#     def get_l1_weight_decay(self, coeff):
#         assert isinstance(coeff, py_float_types)
#         W, = self.transformer.get_params()
#         return coeff * abs(W).sum()


class NeuralNet(Layer):
    """

    Parameters
    ----------
    name : str
        The name of the network
    layers : list of Layer instances
        The layers that are part of this network
    input_source : tuple, optional
        A tuple of input sources which corresponds in structure to the
        input space. Needs to be defined if this is not a nested layer i.e.
        if this network requires some sort of training data.

        To allow for nested NeuralNets which don't need any input sources,
        this is not set to 'features' by default as in MLP.
    input_space : Space, optional
        Can be a simple Space object or a non-nested CompositeSpace. It is
        optional, because a nested neural network layer will get its input
        space set by its parent instead. This will basically define in what
        space the data from Dataset instances should be requested.
    inputs : dict, optional
        Describes the structure of your network. If not given, it is assumed
        the structure of the network is a daisy chain without any input sources
        (i.e. it must be nested). If given, any layer not listed will be
        without an input (unless a parent NeuralNet will assign an input).

        A syntax of `nested_neural_net/child_layer` is support to
        provide/override the input of a layer which is a member of a nested
        NeuralNet. Input sources must be redefined by the top level
        NeuralNet.

        Values of layers furthermore support a syntax of layer_name/1 to refer
        to a single output of a single which has a CompositeSpace output
        (0-indexed).
    targets : dict, optional
        If this network is not nested, you will need to define the target
        source here. In MLP this defaulted to 'targets', but here we do not
        provide a default, because a nested NeuralNet might not need a
        target.

        If a layer has multiple outputs, the layer_name/# syntax is
        supported here as well.
    seed : int, array, None
        The seed that will be used for the RNG. Each of the member layers
        will also use this RNG.
    batch_size : int
        Given here in case convolutional layers need it hardcoded.

    Notes
    -----
    Current restriction is that layers cannot be singletons without
    inputs i.e. just having one layer without an inputs argument doesn't work.
    This doesn't seem to unreasonable: In this case you don't need to wrap
    the layer in a NeuralNet.
    """
    def __init__(self, name, layers, input_source=None, input_space=None,
                 inputs=None, targets=None, seed=None, batch_size=None,
                 **kwargs):
        super(NeuralNet, self).__init__(name, **kwargs)

        # Perform some basic validation on the `layers` argument
        assert isinstance(layers, list)
        assert layers
        assert all(isinstance(layer, Layer) for layer in layers)
        layer_names = [layer.name for layer in layers]
        assert len(layer_names) == len(set(layer_names))
        self.layers = layers
        self.layer_names = layer_names

        # Set up the RNGs
        if seed is None:
            seed = [2013, 1, 4]
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        # Check whether the input space and source match
        if input_space is None or input_source is None:
            assert input_space is None and input_source is None
        else:
            DataSpecsMapping((input_space, input_source))
            assert is_flat_specs((input_space, input_source))

        # Set some class attributes
        # TODO Check necessity
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        # Perform some validation on the inputs dictionary
        # TODO Check that each layer is/has an input i.e. not an orphan
        if inputs is None:
            # Only when no inputs are given, otherwise the case where
            # layers = [A, B] with inputs={'A': ['B']} causes a cycle already
            inputs = dict((layer.name, input_layer.name) for layer, input_layer
                          in izip(layers[1:], layers[:-1]))
        assert all(isinstance(value, list) for value in inputs.values())
        assert all(isinstance(key, basestring) for key in inputs.keys())
        assert all(isinstance(name, basestring)
                   for name in chain(*inputs.values()))
        assert len(set(inputs.keys())) == len(inputs.keys())
        # Check if keys refer to layer names
        assert (set([key.split('/')[0] for key in inputs.keys()]) <=
                set(layer_names))
        # TODO Check existence nested layers
        # Check if values refer to layer names or input sources
        # and whether all input sources are used
        non_layer_values = set([value.split('/')[0] for value in
                                chain(*inputs.values())]) - set(layer_names)
        if input_source is None:
            assert not non_layer_values
        else:
            if isinstance(input_source, basestring):
                assert non_layer_values == set([input_source])
            else:
                assert non_layer_values == set(input_source)
        # TODO Check correctness /# syntax (only for layers and only 1 '/')
        self.inputs = inputs

        if targets is not None:
            assert all(isinstance(value, list) for value in targets.values())
            assert all(isinstance(key, basestring) for key in targets.keys())
            assert all(isinstance(name, basestring)
                       for name in chain(*targets.values()))
            assert len(set(targets.keys())) == len(targets.keys())
            assert (set([key.split('/')[0] for key in targets.keys()]) <=
                    set(layer_names))
            assert not set([value.split('/')[0] for value in
                            chain(*targets.values())]) & set(layer_names)
            # TODO Check correctness /# syntax (only 1 '/')
        self.targets = targets

        # Topologically sort the layers so that we can set the input spaces
        topo_sorted_inputs = self.topological_sort(inputs)
        self.topo_sorted_layers = sorted(
            self.layers,
            key=lambda layer: topo_sorted_inputs.index(layer.name)
        )
        # Output layers have inputs, but no outputs to other layers
        self.output_layers = list(set([key.split('/')[0]
                                      for key in self.inputs.keys()]) -
                                  set([value.split('/')[0]
                                       for value
                                       in chain(*self.inputs.values())]))

        # If not provided, they must be set by a parent NeuralNet later
        if input_space is not None:
            self.input_source = input_source
            self.input_space = input_space

    def __getitem__(self, key):
        """
        Allows layers to be retrieved as follows:

        >>> neural_net = NeuralNet('nnet',
                                   [Linear('h', dim=10,
                                           weights_init=IsotropicGaussian())])
        >>> neural_net['h']
        <pylearn2.models.neural_net.Linear(name=h) at 0x7fa400b722d0>
        """
        return self.layers[self.layer_names.index(key)]

    def __iter__(self):
        """
        Allows a syntax of

        >>> [layer.input_space for for layer in neural_net]
        [VectorSpace(dim=10), ...]
        """
        for layer in self.layers:
            yield layer

    def __contains__(self, key):
        """
        Allows for:

        >>> 'h' in neural_net
        True
        >>> neural_net['h'] in neural_net
        True
        """
        if isinstance(key, basestring):
            return key in self.layer_names
        else:
            return key in self.layers

    def topological_sort(self, inputs):
        sorted_list = []
        edges = defaultdict(list)
        for key, values in inputs.iteritems():
            edges[key.split('/')[0]].extend([value.split('/')[0]
                                             for value in values])
        start_nodes = set(chain(*edges.values())) - set(edges.keys())

        while start_nodes:
            sorted_list.append(start_nodes.pop())
            for target_node, source_nodes in edges.iteritems():
                if sorted_list[-1] in source_nodes:
                    edges[target_node].remove(sorted_list[-1])
                    if not edges[target_node]:
                        start_nodes.add(target_node)
        if any(edges.values()):
            raise ValueError("The inputs given describe a cyclical graph.")
        return sorted_list

    def get_rng(self):
        return super(NeuralNet, self).get_rng()

    def set_rng(self, rng):
        for layer in self.layers:
            layer.rng = rng
        super(NeuralNet, self).set_rng(rng)

    rng = property(get_rng, set_rng)

    def _fprop(self, state_below, space, **kwargs):
        """

        Parameters
        ----------
        kwargs : dict
            TODO
                support for 'layer'/'until' kwarg, in which case it returns the
                output of this layer only.
            TODO
                support for `return_all` argument

        Notes
        -----
        """
        assert space == self.input_space  # TODO Does this need to be enforced?
        if not isinstance(space, CompositeSpace):
            space = CompositeSpace([space])  # To simplify code
            state_below = (state_below,)

        outputs = {}  # Of the form {'name': (state_below, space, kwargs)}
        for layer in self.topo_sorted_layers:
            layer_input_names = self.inputs[layer.name]
            layer_inputs = []
            # Collect the tensors and spaces from the input layers/sources
            for input in layer_input_names:
                if input in self:  # The input is from another layer
                    layer_inputs.append(outputs[input])
                else:  # The input is from a data source
                    layer_state_below = \
                        state_below[self.input_source.index(input)]
                    layer_space = \
                        space.components[self.input_source.index(input)]
                    layer_inputs.append((layer_state_below, layer_space, {}))
            if len(layer_inputs) > 1:
                layer_state_below = tuple(zip(*layer_inputs)[0])
                layer_space = CompositeSpace(zip(*layer_inputs)[1])
            else:
                layer_state_below = layer_inputs[0][0]
                layer_space = layer_inputs[0][1]
            outputs[layer.name] = layer.fprop(layer_state_below, layer_space,
                                              **kwargs)
        net_out = [(state, space, kwargs) for name, (state, space, kwargs)
                   in outputs.iteritems() if name in self.output_layers]
        if len(net_out) > 1:
            state = tuple(zip(*net_out)[0])
            space = CompositeSpace(zip(*net_out)[1])
        else:
            state = net_out[0][0]
            space = net_out[0][1]
        # TODO Combine kwargs and check for consistency
        return state, space, kwargs

#     def get_weight_decay(self):
#         # TODO
#         raise NotImplementedError
#
#     def get_l1_weight_decay(self):
#         # TODO
#         raise NotImplementedError

    def get_params(self):
        params = []
        # Maintain order of parameters
        for layer in self.layers:
            for param in layer.get_params():
                if param not in params:
                    params.append(param)
        return params

    def set_params(self, params):
        # TODO Some testing
        params_names = [param.name for param in self.params]
        for layer in self.layers:
            layer_params = layer.get_params()
            layer_params_names = [layer_param.name
                                  for layer_param in layer_params]
            new_layer_params = \
                [params[params_names.index(layer_param_name)]
                 for layer_param_name in layer_params_names]
            layer.set_params(new_layer_params)

    params = property(get_params, set_params)

    def get_input_space(self):
        return super(NeuralNet, self).get_input_space()

    def set_input_space(self, space):
        super(NeuralNet, self).set_input_space(space)

        for layer in self.topo_sorted_layers:
            for input_name in self.inputs.get(layer.name, []):
                input_space = []
                if input_name in self.layer_names:
                    input_space.append(self[input_name].output_space)
                else:
                    if isinstance(self.input_space, CompositeSpace):
                        input_space.append(self.input_space.components[
                            self.input_source.index(input_name)])
                    else:
                        input_space.append(self.input_space)

            if len(input_space) == 1:
                layer.input_space = input_space.pop()
            elif len(input_space) > 1:
                layer.input_space = CompositeSpace(input_space)

        if len(self.output_layers) == 1:
            self.output_space = self[self.output_layers[0]].output_space
        else:
            self.output_space = CompositeSpace([self[output_layer].output_space
                                                for output_layer
                                                in self.output_layers])

    input_space = property(get_input_space, set_input_space)


class Linear(Layer):  # TransformerWeightDecay
    def __init__(self, name, dim, weights_init, bias_init=None,
                 use_bias=True, **kwargs):
        super(Linear, self).__init__(name, **kwargs)

        assert isinstance(dim, py_integer_types)
        if use_bias and bias_init is None:
            bias_init = Constant(0)
        self.__dict__.update(locals())
        del self.self

    def _fprop(self, state_below, space, **kwargs):
        state_below = space.format_as(state_below,
                                      VectorSpace(dim=self.input_dim))
        return (self._linear_part(state_below), VectorSpace(dim=self.dim),
                kwargs)

    def get_input_space(self):
        return super(Linear, self).get_input_space()

    def set_input_space(self, space):
        # TODO Maybe need some sort of space.is_compatible(VectorSpace)?
        #      Now the error doesn't happen until `fprop` is called.
        #      Then again, an error could still happen if a different space
        #      is passed to fprop.
        self.input_dim = space.get_total_dimension()
        self.transformer = MatrixMul(sharedX(np.zeros((self.input_dim,
                                                       self.dim)),
                                             name=self.name + '_W'))
        self.weights = self.transformer.get_params()
        param_vals = [self.weights_init.initialize(self.rng,
                                                   (self.input_dim, self.dim))]
        if self.use_bias:
            self.biases = [sharedX(np.zeros((self.dim,)), self.name + '_b')]
            param_vals.append(self.bias_init.initialize(self.rng,
                                                        (self.dim,)))
        else:
            self.biases = []
        self.params = self.weights + self.biases
        self.param_vals = param_vals
        self.output_space = VectorSpace(dim=self.dim)
        super(Linear, self).set_input_space(space)

    input_space = property(get_input_space, set_input_space)

    def _linear_part(self, state_below):
        z = self.transformer.lmul(state_below)
        if self.use_bias:
            b, = self.biases
            z += b
        return z


class Activation(Linear, Layer):
    """
    This is a base class for layers which perform a linear (affine)
    transformation and then apply an activation function. If
    affine is set to True, it will use the Linear class's fprop. If not,
    it will just use the default methods from the Layer class.
    """
    def __init__(self, name, affine=True, dim=None, weights_init=None,
                 bias_init=None, use_bias=True, **kwargs):
        if affine:
            assert weights_init is not None and dim is not None
            Linear.__init__(self, name, dim, weights_init, bias_init,
                            **kwargs)
        self.affine = affine

    def _fprop(self, state_below, space, **kwargs):
        if self.affine:
            state_below, space, kwargs = Linear._fprop(self, state_below,
                                                       space, **kwargs)
        return state_below, space, kwargs

    def get_input_space(self):
        return super(Activation, self).get_input_space()

    def set_input_space(self, space):
        if self.affine:
            Linear.set_input_space(self, space)
        else:
            Layer.set_input_space(self, space)
            self.weights = []
            self.biases = []
            self.params = []
            self.output_space = space

    input_space = property(get_input_space, set_input_space)


class Tanh(Activation):
    def _fprop(self, state_below, space, **kwargs):
        state_below, space, kwargs = super(Tanh, self)._fprop(state_below,
                                                              space, **kwargs)
        return tensor.tanh(state_below), space, kwargs


class Sigmoid(Activation):
    def _fprop(self, state_below, space, **kwargs):
        state_below, space, kwargs = super(Tanh, self)._fprop(state_below,
                                                              space, **kwargs)
        return tensor.nnet.sigmoid(state_below), space, kwargs
