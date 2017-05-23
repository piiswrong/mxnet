# coding: utf-8
# pylint: disable= arguments-differ
"""Neural network layers."""

from ... import symbol, ndarray
from ...symbol import Symbol
from ...ndarray import NDArray
from .parameter import Parameter, ParameterDict


class Layer(object):
    """Base class for all neural network layers and models.

    Your models should subclass this class.

    Layers can also contain other Layers, allowing you to nest them in a tree
    structure. You can assign sublayers as regular attributes::
        from mxnet import nn
        class Net(nn.Layer):
            def __init__(self, prefix=None, params=None):
                super(Net, self).__init__(prefix=prefix, params=params)
                self.dense1 = nn.Dense(20, in_units=10, prefix=self.prefix+'dense1_')
                self.dense2 = nn.Dense(20, in_units=20, prefix=self.prefix+'dense2_')

            def forward(self, x):
                x = self.dense1(x)
                return self.dense2(x)

    Sublayers assigned this way will be registered and will have their status changed
    too when you call .train() etc.

    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. It will be prepended to the name of all Symbols and
        Parameters created by this layer. Prefix should be unique within one network
        to prevent name collisions.
    params : ParameterDict or None
        Manages Parameters of this Layer and sublayers. You can make two Layers share
        parameter by passing the same dictionary to them. For example::
            params = nn.ParameterDict(prefix='dense_')
            dense1 = nn.Dense(20, in_units=10, prefix='dense1_', params=params)
            dense2 = nn.Dense(20, in_units=10, prefix='dense2_', params=params)

        dense1 and dense2 now have shared weights.
    """
    _counter = {}

    def __init__(self, prefix=None, params=None):
        self._prefix = self._make_prefix(prefix)
        self._name = self.prefix[:-1] if self.prefix.endswith('_') else self.prefix
        if params is None:
            params = ParameterDict(prefix=self.prefix)
        self._params = params
        self._children = []

    def __setattr__(self, name, value):
        """Automatically register sublayers."""
        super(Layer, self).__setattr__(name, value)
        if isinstance(value, Layer):
            self.register_sublayer(value)

    def _alias(self):
        return self.__class__.__name__.lower()

    def _make_prefix(self, prefix=None):
        """Create prefix. By default use class name plus counter."""
        if prefix is None:
            count = Layer._counter.get(type(self), 0)
            Layer._counter[type(self)] = count + 1
            return '%s%d_'%(self._alias(), count)
        return prefix

    @property
    def params(self):
        """A ParameterDict managing this Layer's Parameters."""
        return self._params

    @property
    def prefix(self):
        """Prefix of this Layer."""
        return self._prefix

    @property
    def name(self):
        if self.prefix.endswith('_'):
            return self.prefix[:-1]
        return self.prefix

    def register_sublayer(self, layer):
        """Register layer as sublayer of self. Layers assigned to self as attributes
        will be registered automatically."""
        self._children.append(layer)
        self.params.merge(layer.params)

    def __call__(self, *args, **kwargs):
        """Call forward."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Defines the forward computation. Arguments can be either NDArray or Symbol."""
        raise NotImplementedError


class Sequential(Layer):
    """Stack Layers sequentially.

    Example::
        net = nn.Sequential()
        net.add(Dense(10, activation='relu'))
        net.add(Dense(20))
    """
    def __init__(self, params=None, **kwargs):
        super(Sequential, self).__init__(prefix='', params=params, **kwargs)

    def add(self, layer):
        """Add layer on top of the stack."""
        self.register_sublayer(layer)

    def forward(self, x):
        #pylint: disable=arguments-differ
        for layer in self._children:
            x = layer(x)
        return x


class SimpleLayer(Layer):
    """SimpleLayer is a Layer that supports forwarding with both `Symbol` and `NDArray`.
    A SimpleLayer cannot use API that's specific to `NDArray` API, e.g. `.asnumpy()`.

    SimpleLayer is mostly used by developers or advanced users as a base class.
    If you only want to use one of `Symbol` and `NDArray` API you should inherit
    Layer instead."""
    def __init__(self, prefix=None, params=None, **kwargs):
        super(SimpleLayer, self).__init__(prefix=prefix, params=params, **kwargs)
        self._reg_params = {}

    def __setattr__(self, name, value):
        """Registers parameters in addition to sublayers."""
        if isinstance(value, Parameter):
            self._reg_params[name] = value
        super(SimpleLayer, self).__setattr__(name, value)

    def forward(self, x, *args):
        if isinstance(x, NDArray):
            with x.context as ctx:
                params = {k: v.data(ctx) for k, v in self._reg_params.items()}
                return self.ndarray_forward(x, *args, **params)
        else:
            assert isinstance(x, Symbol), \
                "SimpleLayer requires the first argument to forward be either Symbol or NDArray"
            params = {k: v.var() for k, v in self._reg_params.items()}
            return self.symbol_forward(x, *args, **params)

    def ndarray_forward(self, x, *args, **kwargs):
        return self.simple_forward(ndarray, x, *args, **kwargs)

    def symbol_forward(self, x, *args, **kwargs):
        return self.simple_forward(symbol, x, *args, **kwargs)

    def simple_forward(self, F, x, *args, **kwargs):
        """Simple forward supports both `Symbol` and `NDArray` API.

        Parameters
        ----------
        F : {mxnet.ndarray, mxnet.symbol}
            Name space of operators. `F` will be set to `mx.sym` when x is `Symbol`
            instance and `mx.nd` when x is `NDArray` instance.
        x : NDArray or Symbol
            The first input tensor.
        *args : list of NDArray or list of Symbol
            Additional input tensors.
        **kwargs : dict of str to NDArray or dict of str to Symbol
            `Symbol` or `NDArray` value of registered Parameters.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class Dense(SimpleLayer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use flatten to convert it
    to rank 2 manually if necessary.

    Example::
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, in_uints=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # No need to specify the size of the input if you only want to
        # use the `Symbol` API:
        model = Sequential()
        model.add(Dense(32))

    Parameters
    ----------
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use
        (see help on Activation operator).
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
        (see mxnet.initializer).
    bias_initializer: Initializer for the bias vector
        (see mxnet.initializer).
    in_units : int
        Size of input data. No need to specify for `Symbol` API. But must be
        specified for every Dense layer if you want to use `NDArray` API.
    prefix : str or None
        See document of Layer.
    params : ParameterDict or None
        See document of Layer.

    Input shape
    -----------
    a 2D input with shape `(batch_size, in_units)`.

    Output shape
    ------------
    the output would have shape `(batch_size, units)`.
    """
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None,
                 in_units=0, prefix=None, params=None, **kwargs):
        super(Dense, self).__init__(prefix=prefix, params=params, **kwargs)
        self._units = units
        self._use_bias = use_bias
        self.weight = self.params.get('weight', shape=(units, in_units),
                                      init=kernel_initializer)
        if use_bias:
            self.bias = self.params.get('bias', shape=(units,),
                                        init=bias_initializer)
        if activation is not None:
            self.act = Activation(activation, prefix=self.prefix+activation+'_',
                                  params=self.params.subdict(activation+'_'))
        else:
            self.act = None

    def simple_forward(self, F, x, **kwargs):
        act = F.FullyConnected(data=x, num_hidden=self._units,
                               no_bias=not self._use_bias,
                               **kwargs)
        if self.act is not None:
            act = self.act(act)
        return act


class Activation(SimpleLayer):
    """Applies an activation function to input.

    Parameters
    ----------
    activation: name of activation function to use
        See: help on Activation operator

    Input shape
    -----------
    Arbitrary.

    Output shape
    ------------
    Same shape as input.
    """
    def __init__(self, activation, prefix=None, params=None, **kwargs):
        super(Activation, self).__init__(prefix=prefix, params=params, **kwargs)
        self._act_type = activation

    def simple_forward(self, F, x):
        return F.Activation(x, act_type=self._act_type)


class Dropout(SimpleLayer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Parameters
    ----------
    rate: float between 0 and 1. Fraction of the input units to drop.

    References
    ----------
    - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
        http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """
    def __init__(self, rate, prefix=None, params=None, **kwargs):
        super(Dropout, self).__init__(prefix=prefix, params=params, **kwargs)
        self._rate = rate

    def simple_forward(self, F, x):
        return F.Dropout(x, p=self._rate)


class BatchNorm(SimpleLayer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Parameters
    ----------
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    """
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-3, center=True, scale=True,
                 num_features=0, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 prefix=None, params=None, **kwargs):
        super(BatchNorm, self).__init__(prefix=prefix, params=params, **kwargs)
        assert axis == 1, \
            "Only support NC* layout, i.e. channel must be in the second dimension"
        self._kwargs = {'eps': epsilon, 'momentum': momentum, 'fix_gamma': not center}

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(num_features,), init=gamma_initializer)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(num_features,), init=beta_initializer)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(num_features,),
                                            init=running_mean_initializer)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(num_features,),
                                           init=running_variance_initializer)

    def simple_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.BatchNorm(x, gamma, beta, running_mean, running_var, **self._kwargs)
