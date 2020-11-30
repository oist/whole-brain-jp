# coding: utf-8

import numpy as np
from nest import topology as tp
import nest

class Network(object):

    @staticmethod
    def pons_to_gr(layer_pons, layer_gr, subCB):
        weights = 1.
        g = [0.18, 0.025, 0.028, 0.028]
        alpha_const = [1.0, 1.0, 0.43, 0.57]
        multiplier = 0.225
        weight = np.max(weights * np.multiply(g, alpha_const)) * multiplier

        sigma_x = 0.05
        sigma_y = 0.05
        p_center = 1.
        delay = 1.0  # 1.0
        conndict = {'connection_type': 'divergent',
                    'mask': {'box': {'lower_left': [-0.5, -0.5, -0.5],
                                     'upper_right': [0.5, 0.5, 0.5]}},
                    'kernel': {
                        'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': weight,
                    'delays': delay}
        tp.ConnectLayers(layer_pons, layer_gr, conndict)
