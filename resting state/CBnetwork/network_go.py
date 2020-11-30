# coding: utf-8

import numpy as np
from nest import topology as tp


class Network(object):

    @staticmethod
    def go_to_gr(layer_go, layer_gr, subCB, kernel=0.025, weights=-10.0, g=[0.18, 0.025, 0.028, 0.028], alpha_const=[1.0, 1.0, 0.43, 0.57], multiplier=2.0):
        # define connections
        #    golgi to granule
        #    1024. / 32. = 32. is the length between golgi cells
        #    inhibitory inputs from 9x9 nearby model golgi cells with probability 0.025
        #    synaptic weights: 10.0
        configuration = {'connection_type': 'divergent', 'mask': {'box': {}}}
        configuration['kernel'] = kernel
        configuration['sources'] = {'model': subCB+'_layer_go'}
        configuration['targets'] = {'model': subCB+'_layer_gr'}
        lower_left = [-0.15, -0.15, -0.5]
        upper_right =[0.15, 0.15, 0.5]
        configuration['mask']['box']['lower_left'] = lower_left
        configuration['mask']['box']['upper_right'] = upper_right
        weights = weights * np.multiply(g, alpha_const) * multiplier
        configuration['weights'] = np.max(weights)
        tp.ConnectLayers(layer_go, layer_gr, configuration)
