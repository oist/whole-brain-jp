# coding: utf-8

import numpy as np
from nest import topology as tp
import nest

class Network(object):

    @staticmethod
    def gr_to_go(layer_gr, layer_go, subCB, kernel=0.05, weights=0.00005, g=[45.5, 30.0, 30.0], alpha_const=[1.0, 0.33, 0.67],  mulplier=1.0):
        #nbcpu
        #    1024. / 32. = 32. is the length between granule clusters
        #    excitatory inputs from 7x7 nearby model granule clusters with probability 0.5
        #    synaptic weights: 0.00004
        configuration = {'connection_type': 'divergent', 'mask': {'box': {}}}
        configuration['kernel'] = kernel
        configuration['sources'] = {'model': subCB+'_layer_gr'}
        configuration['targets'] = {'model': subCB+'_layer_go'}
        lower_left = [-0.1, -0.1, -0.1]
        upper_right = [0.1, 0.1, 0.1]
        configuration['mask']['box']['lower_left'] = lower_left
        configuration['mask']['box']['upper_right'] = upper_right
        weights = weights * np.multiply(g, alpha_const)
        configuration['weights'] = np.max(weights) * mulplier
        tp.ConnectLayers(layer_gr, layer_go, configuration)

############################
#gr to pkj new function
#2019-8-9 sun
##############################

    @staticmethod
    def gr_to_pkj(layer_gr, layer_pkj, subCB):

        weights = 0.007
        g = [0.7, 1.0]
        alpha_const = [1.0, 1.0]
        multiplier = 1.0
        weight = np.max(weights * np.multiply(g, alpha_const)) * multiplier

        sigma_x = 0.1
        sigma_y = 0.1
        p_center = 1.
        delay = 1.0  # 1.0
        conndict = {'connection_type': 'divergent',
                    'mask': {'box': {'lower_left': [-2., -2., -2.],
                                     'upper_right': [2., 2., 2.], }},
                    'kernel': {
                        'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': weight,
                    'delays': delay}
        tp.ConnectLayers(layer_gr, layer_pkj, conndict)


    @staticmethod
    def gr_to_bs(layer_gr, layer_bs, subCB, kernel=1.0, weights=0.015):
        # granule to bs
        #    1024. / 32. = 32. is the length between granule clusters
        #    excitatory inputs from 32x9 nearby model granule clusters with probability 1.0
        #    synaptic weights: 0.003
        configuration = {'connection_type': 'divergent', 'mask': {'box': {}}}
        configuration['kernel'] = kernel
        configuration['weights'] = weights
        configuration['sources'] = {'model': subCB+'_layer_gr'}
        configuration['targets'] = {'model': subCB+'_layer_bs'}
        lower_left = [-0.5, -0.1, -0.5]
        upper_right = [0.5,  0.1, 0.5]
        configuration['mask']['box']['lower_left'] = lower_left
        configuration['mask']['box']['upper_right'] = upper_right
        tp.ConnectLayers(layer_gr, layer_bs, configuration)
