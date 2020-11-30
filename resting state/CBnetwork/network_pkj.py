# coding: utf-8

import numpy as np
from nest import topology as tp
import nest

class Network(object):
    @staticmethod
    def pkj_to_vn(layer_pkj, layer_vn, subCB):
        weights = -0.00035
        g = [50.0, 25.8, 30.0]
        alpha_const = [1.0, 1.0, 1.0]
        multiplier = 2.0
        weight = np.max(weights * np.multiply(g, alpha_const)) * multiplier

        sigma_x = 0.1
        sigma_y = 0.1
        p_center = 1.
        delay = 1.0  # 1.0
        conndict = {'connection_type': 'divergent',
                    'mask': {'box': {'lower_left': [-0.5, -0.5, -0.5],
                                     'upper_right': [0.5, 0.5, 0.5]}},
                    'kernel': {
                        'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': weight,
                    'delays': delay}
        tp.ConnectLayers(layer_pkj, layer_vn, conndict)

    @staticmethod
    def pkj_to_s(layer_pkj, layer_s_pkj, kernel=1.0, weights=1.0, rows=1, columns=1):
        # vn to spike_detector
        #        connections are confirmed
        configuration = {'connection_type': 'convergent', 'mask': {'grid': {}}}
        configuration['kernel'] = kernel
        configuration['weights'] = weights
        configuration['sources'] = {'model': 'PKJ'}
        configuration['targets'] = {'model': 'SD'}
        configuration['mask']['grid']['rows'] = rows
        configuration['mask']['grid']['columns'] = columns
        tp.ConnectLayers(layer_pkj, layer_s_pkj, configuration)
