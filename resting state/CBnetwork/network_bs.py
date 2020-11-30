# coding: utf-8
import numpy as np
from nest import topology as tp


class Network(object):
    @staticmethod
    def bs_to_pkj(layer_bs, layer_pkj, subCB, kernel=1.0, weights=-1.0, g=[0.7, 1.0], alpha_const=[1.0, 1.0], multiplier=1.5):
        # bs to pkj
        #    1024. / 16. = 64.0 is the length between basket cells
        #    inhibitory inputs from 1x3 nearby model basket cells with probability 1.0
        #    synaptic weights: 5.3
        #    connections are confirmed
        configuration = {'connection_type': 'divergent', 'mask': {'box': {}}}
        configuration['kernel'] = kernel
        configuration['sources'] = {'model': subCB+'_layer_bs'}
        configuration['targets'] = {'model': subCB+'_layer_pkj'}
        lower_left = [-0.1, -0.25, -0.5]
        upper_right = [0.1, 0.25, 0.5]
        configuration['mask']['box']['lower_left'] = lower_left
        configuration['mask']['box']['upper_right'] = upper_right
        weights = weights * np.multiply(g, alpha_const)
        configuration['weights'] = np.max(weights) * multiplier
        tp.ConnectLayers(layer_bs, layer_pkj, configuration)
