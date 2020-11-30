
# coding: utf-8

import numpy as np
from nest import topology as tp

class Network(object):

    @staticmethod
    def vn_to_io(layer_vn, layer_io, kernel=1.0, weights=-5.0, g=[1.0, 0.18], alpha_const=[1.0, 1.0], rows=1, columns=1, multiplier=0.0):
        # vn to io
        #    inhibitory inputs from 1x1 nearby model VN cells with probability 1.0
        #    synaptic weights: 5.0
        configuration = {'connection_type': 'divergent', 'mask': {'grid': {}}}
        configuration['kernel'] = kernel
        configuration['sources'] = {'model': 'VN'}
        configuration['targets'] = {'model': 'IO'}
        configuration['mask']['grid']['rows'] = rows
        configuration['mask']['grid']['columns'] = columns
        weights = weights * np.multiply(g, alpha_const)
        configuration['weights'] = np.max(weights) * multiplier
        tp.ConnectLayers(layer_vn, layer_io, configuration)

    @staticmethod
    def vn_to_s(layer_vn, layer_s_vn, kernel=1.0, weights=1.0, rows=1, columns=1):
        # vn to spike_detector
        #        connections are confirmed
        configuration = {'connection_type': 'convergent', 'mask': {'grid': {}}}
        configuration['kernel'] = kernel
        configuration['weights'] = weights
        configuration['sources'] = {'model': 'VN'}
        configuration['targets'] = {'model': 'SD'}
        configuration['mask']['grid']['rows'] = rows
        configuration['mask']['grid']['columns'] = columns
        print (layer_s_vn)
        tp.ConnectLayers(layer_vn, layer_s_vn, configuration)
