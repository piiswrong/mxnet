# coding: utf-8
# pylint: disable=too-many-arguments, no-member, protected-access, too-many-locals
# pylint: disable=unused-argument
""" losses for training neural networks """
from __future__ import absolute_import

import json

from ... import symbol, ndarray, metric
from ...base import numeric_types


def _get_F(x):
    """Get function domain from tensor"""
    return symbol if isinstance(x, symbol.Symbol) else ndarray


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        the loss to be weighted.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, sample_weight should have
        shape (64, 1)

    Returns
    -------
    loss : Symbol
        weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


def _unpack_symbol(loss):
    assert isinstance(loss, symbol.Symbol)
    outputs = symbol.Group([i for i in loss if i.attr('__output__') == 'out'])
    extra_outputs = symbol.Group([i for i in loss if i.attr('__output__') == 'extra'])
    losses = symbol.Group([i for i in loss if i.attr('__output__') == 'loss'])
    return outputs, extra_outputs, losses


def custom_loss(loss, output, label, weight=None, sample_weight=None,
                extra_outputs=(), metrics=None, name='custom'):
    F = _get_F(loss)
    loss = _apply_weighting(F, loss, weight, sample_weight)
    if F is ndarray:
        return loss
    outputs = [F.stop_gradient(i, name=i.name+'_out', __output__='out')
               for i in output]
    extra_outputs = [F.stop_gradient(i, name=i.name+'_out', __output__='extra')
                     for i in extra_outputs]

    loss = F.make_loss(loss, name=name, __output__='loss')

    if metrics:
        metrics = metric.create(metrics)
        metrics.output_names = outputs.list_outputs()
        metrics.label_names = label.list_outputs()
        loss._set_attr(__metric__=json.dumps(metrics.get_config()))

    return symbol.Group(outputs + extra_outputs + [loss])


def l2_loss(output, label, weight=1., sample_weight=None,
            extra_outputs=(), metrics='mse', name='l2'):
    """Calculate the mean squared error between output and label:

    .. math::
    L = \\frac{1}{2}\\sum_i \\Vert {output}_i - {label}_i \\Vert^2.

    output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    F = _get_F(output)
    if weight is None:
        weight = 1.
    loss = F.square(output.reshape(shape=(-1,)) - label.reshape(shape=(-1,)))
    return custom_loss(loss, output, label, weight/2, sample_weight,
                       extra_outputs, metrics, name)


def l1_loss(output, label, weight=None, sample_weight=None,
            extra_outputs=(), metrics='mae', name='l1'):
    """Calculate the mean absolute error between output and label:

    .. math::
    L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    F = _get_F(output)
    loss = F.abs(output.reshape(shape=(-1,)) - label.reshape(shape=(-1,)))
    return custom_loss(loss, output, label, weight, sample_weight,
                       extra_outputs, metrics, name)


def softmax_cross_entropy_loss(output, label, sparse_label=True, axis=-1,
                               weight=None, sample_weight=None,
                               extra_outputs=(), metrics='acc', name='ce'):
    """Compute the softmax cross entropy loss.

    If sparse_label is True, label should contain integer category indicators:
    .. math::
    p = {softmax}({output})
    L = -\\sum_i {log}(p_{i,{label}_i})
    label's shape should be output's shape without the `axis` dimension. i.e. for
    output.shape = (1,2,3,4) and axis = 2, label.shape should be (1,2,4)

    If sparse_label is False, label should cantain probability distribution
    with the same shape as output:
    .. math::
    p = {softmax}({output})
    L = -\\sum_i \\sum_j {label}_j {log}(p_{ij})

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    sparse_label : bool, default True
        where label is sparse integer or probability distribution
    axis : int, default -1
        The axis to sum over when computing softmax and entropy
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    F = _get_F(output)
    prob = F.log_softmax(output)
    if sparse_label:
        loss = -F.pick(prob, label, axis=axis, keepdims=False)
    else:
        loss = -F.sum(prob*label, axis=axis, keepdims=False)
    return custom_loss(loss, prob, label, weight, sample_weight,
                       extra_outputs, metrics, name)
