#!/usr/bin/env python
# -*- coding: utf-8 -*-
#please set all flags before run the simulation
##

import fetch_params
import ini_all
import nest_routine
import nest
import nest.topology as ntop
import numpy as np
import time


def WB_build():
    '''
    #### Build the whole-brain network                    ####
    #### based on parameters defined at baseSimParams.py. ####
    #### This will only build the network and provide     ####
    #### its layers and spike detectors                   ####
    #### for further simulation.                          ####
    #### build_it will "build" the regions specified by      ####
    #### the 'on':TRUE element of diccionary 'sim_model'  ####
    #### of baseSimParams.py                              ####
    '''
    ctx_S1_layers, ctx_M1_layers, th_layers, cb_layers_S1, cb_layers_M1, detectors = {},{},{},{},{},{}

    # 1) reads parameters
    sim_params = fetch_params.read_sim()

    for sim_model in sim_params['sim_model'].keys():
        if sim_params['sim_model'][sim_model]['on']:
            sim_regions=sim_params['sim_model'][sim_model]['regions']
            sim_model_on=sim_model
            print ('simulation model ', sim_model_on, ' will start')
    if sim_regions['S1']:
        ctx_S1_params = fetch_params.read_ctx()
    if sim_regions['M1']:
        ctx_M1_params = fetch_params.read_ctx_M1()
    if sim_regions['TH_S1'] or sim_regions['TH_M1']:
        th_params = fetch_params.read_th()
    if sim_regions['BG']:
        bg_params = fetch_params.read_bg()
    if sim_regions['CB_S1'] or sim_regions['CB_M1']:
        cb_params = fetch_params.read_cb()
    conn_params = fetch_params.read_conn()

    # 1.5) initialize nest
    nest_routine.initialize_nest(sim_params)

    sim_params['circle_center'] = nest_routine.get_channel_centers(sim_params, hex_center=[0, 0],
                                                                   ci=sim_params['channels_nb'],
                                                                   hex_radius=sim_params['hex_radius'])
    # 2) instantiates regions
    if sim_regions['S1']:
        ctx_S1_layers = ini_all.instantiate_ctx(ctx_S1_params, sim_params['scalefactor'], sim_params['initial_ignore'])
    if sim_regions['M1']:
        ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'],sim_params['initial_ignore'])
    if sim_regions['TH_S1'] or sim_regions['TH_M1']:
        th_layers = ini_all.instantiate_th(th_params, sim_params['scalefactor'],sim_params['initial_ignore'])
    if sim_regions['CB_S1']:
        cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'], sim_params)
    if sim_regions['CB_M1']:
        cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'], sim_params)

    if sim_regions['BG']:
        if sim_params['channels']: #if True
            bg_params['channels'] = True #for channels input tasks
        else:
            bg_params['channels'] = False #resting state
        bg_params['circle_center'] = nest_routine.get_channel_centers(sim_params, hex_center=[0, 0],
                                                                ci=sim_params['channels_nb'],
                                                                hex_radius=sim_params['hex_radius'])
        
        #sim_params['circle_center'] = bg_params['circle_center']            
        #### Basal ganglia inter-regional connection with S1 and M1 ######
        if sim_regions['S1']:
            if sim_regions['M1']:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': {'layers': ctx_S1_layers, 'params': ctx_S1_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
            else:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': None, #{'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': {'layers': ctx_S1_layers, 'params': ctx_S1_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
        else:
            if sim_regions['M1']:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': None, #{'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
            else:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': None, #{'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': None, #{'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])

    # 3) interconnect regions
    start_time = time.time()
    if sim_regions['S1'] and sim_regions['M1']:
        pass
        #_ = nest_routine.connect_region_ctx_cb(ctx_layers['S1_L5B_Pyr'], cb_layers_S1['CB_S1_layer_pons'], 'S1')
    if sim_regions['S1'] and sim_regions['CB_S1']:
        _ = nest_routine.connect_region_ctx_cb(ctx_S1_layers['S1_L5B_Pyr'], cb_layers_S1['CB_S1_layer_pons'], 'S1')
    if sim_regions['M1'] and sim_regions['CB_M1']:
        _ = nest_routine.connect_region_ctx_cb(ctx_M1_layers['M1_L5B_PT'], cb_layers_M1['CB_M1_layer_pons'], 'M1')
    if sim_regions['S1'] and sim_regions['TH_S1']:
        _ = nest_routine.connect_region_ctx_th(ctx_S1_layers, th_layers, 'S1')
    if sim_regions['M1'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_region_ctx_th(ctx_M1_layers, th_layers, 'M1')
    if sim_regions['TH_S1'] and sim_regions['S1']:
        _ = nest_routine.connect_region_th_ctx(th_layers, ctx_S1_layers, 'S1')
    if sim_regions['TH_M1'] and sim_regions['M1']:
        _ = nest_routine.connect_region_th_ctx(th_layers, ctx_M1_layers, 'M1')
    if sim_regions['CB_S1'] and sim_regions['S1']:
        _ = nest_routine.connect_region_cb_th(cb_layers_S1, th_layers, 'S1')
    if sim_regions['CB_M1'] and sim_regions['M1']:
        _ = nest_routine.connect_region_cb_th(cb_layers_M1, th_layers, 'M1')
    if sim_regions['BG'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_region_bg_th(bg_layers, th_layers)
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Interconnect_Regions_Time ' + str(time.time() - start_time) + '\n')

    # 2.5) detectors
    #detectors = {}
    if sim_regions['BG']:
        for layer_name in bg_layers.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(bg_layers[layer_name], layer_name,
                                                                      sim_params['initial_ignore'])
    if sim_regions['S1']:
        for layer_name in ctx_S1_layers.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(ctx_S1_layers[layer_name], layer_name, sim_params['initial_ignore'])
    if sim_regions['M1']:
        for layer_name in ctx_M1_layers.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(ctx_M1_layers[layer_name], layer_name, sim_params['initial_ignore'])
    if sim_regions['CB_S1']:
        for layer_name in cb_layers_S1.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_S1[layer_name], layer_name, sim_params['initial_ignore'])
    if sim_regions['CB_M1']:
        for layer_name in cb_layers_M1.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_M1[layer_name], layer_name, sim_params['initial_ignore'])
    if sim_regions['TH_S1']:
        for layer_name in th_layers['TH_S1_EZ'].keys():
            detectors['TH_S1_EZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_EZ'][layer_name], 'TH_S1_EZ_'+layer_name, sim_params['initial_ignore'])
        for layer_name in th_layers['TH_S1_IZ'].keys():
            detectors['TH_S1_IZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_IZ'][layer_name], 'TH_S1_IZ_'+layer_name, sim_params['initial_ignore'])
    if sim_regions['TH_M1']:
        for layer_name in th_layers['TH_M1_EZ'].keys():
            detectors['TH_M1_EZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_EZ'][layer_name], 'TH_M1_EZ_'+layer_name, sim_params['initial_ignore'])
        for layer_name in th_layers['TH_M1_IZ'].keys():
            detectors['TH_M1_IZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_IZ'][layer_name], 'TH_M1_IZ_'+layer_name, sim_params['initial_ignore'])

    return sim_params, ctx_S1_layers, ctx_M1_layers, th_layers, cb_layers_S1, cb_layers_M1, detectors



