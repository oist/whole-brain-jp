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


def main():
    # 1) reads parameters
    sim_params = fetch_params.read_sim()

    for sim_model in sim_params['sim_model'].keys():
        if sim_params['sim_model'][sim_model]['on']:
            sim_regions=sim_params['sim_model'][sim_model]['regions']
            sim_model_on=sim_model
            print ('simulation model ', sim_model_on, ' will start')
    if sim_regions['S1']:
        ctx_params = fetch_params.read_ctx()
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
    wb_layers = {}
    # 2) instantiates regions
    if sim_regions['S1']:
        ctx_layers = ini_all.instantiate_ctx(ctx_params, sim_params['scalefactor'], sim_params['initial_ignore'])
        wb_layers = dict(wb_layers, **ctx_layers)
    if sim_regions['M1']:
        ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'],sim_params['initial_ignore'])
        wb_layers = dict(wb_layers, **ctx_M1_layers)
    if sim_regions['TH_S1'] or sim_regions['TH_M1']:
        th_layers = ini_all.instantiate_th(th_params, sim_params['scalefactor'],sim_params['initial_ignore'])
        wb_layers = dict(wb_layers, **th_layers)
    if sim_regions['CB_S1']:
        cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'], sim_params)
        wb_layers = dict(wb_layers, **cb_layers_S1)
    if sim_regions['CB_M1']:
        cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'], sim_params)
        wb_layers = dict(wb_layers, **cb_layers_M1)

    if sim_regions['BG']:
        if sim_params['channels']:
            bg_params['channels'] = True 
        else:
            bg_params['channels'] = False 
        bg_params['circle_center'] = nest_routine.get_channel_centers(sim_params, hex_center=[0, 0],
                                                                ci=sim_params['channels_nb'],
                                                                hex_radius=sim_params['hex_radius'])
        #### Basal ganglia inter-regional connection with S1 and M1 ######
        if sim_regions['S1']:
            if sim_regions['M1']:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': {'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
            else:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': None, #{'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': {'layers': ctx_layers, 'params': ctx_params},
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
        wb_layers = dict(wb_layers, **bg_layers)

    # 3) interconnect regions
    start_time = time.time()
    if sim_regions['S1'] and sim_regions['M1']:
        pass
        # _ = nest_routine.connect_inter_regions('S1', 'M1')
    if sim_regions['S1'] and sim_regions['CB_S1']:
        _ = nest_routine.connect_inter_regions('S1', 'CB', conn_params, wb_layers)
    if sim_regions['M1'] and sim_regions['CB_M1']:
        _ = nest_routine.connect_inter_regions('M1', 'CB', conn_params, wb_layers)
    if sim_regions['S1'] and sim_regions['TH_S1']:
        _ = nest_routine.connect_inter_regions('S1', 'TH', conn_params, wb_layers)
        _ = nest_routine.connect_inter_regions('TH', 'S1', conn_params, wb_layers)
    if sim_regions['M1'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_inter_regions('M1', 'TH', conn_params, wb_layers)
        _ = nest_routine.connect_inter_regions('TH', 'M1', conn_params, wb_layers)
    if sim_regions['CB_S1'] and sim_regions['TH_S1']:
        _ = nest_routine.connect_inter_regions('CB', 'TH', conn_params, wb_layers)
    if sim_regions['CB_M1'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_inter_regions('CB', 'TH', conn_params, wb_layers)
    if sim_regions['BG'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_inter_regions('BG', 'TH', conn_params, wb_layers)
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Interconnect_Regions_Time ' + str(time.time() - start_time) + '\n')

    ### adding corrections of edge effect (this mitigation works partially) ####
    _ = nest_routine.reduce_weights_at_edges('M1_L5B_PT','CB_M1_layer_pons',wb_layers['M1_L5B_PT'],wb_layers['CB_M1_layer_pons'],margin=0.025,new_weight=0.0)
    _ = nest_routine.reduce_weights_at_edges('GPi_fake','TH_M1_IZ_thalamic_nucleus_TC',wb_layers['GPi_fake'],wb_layers['TH_M1_IZ']['thalamic_nucleus_TC'],margin=0.025,new_weight=-1940.)



    # 2.5) detectors
    detectors = {}
    if sim_regions['BG']:
        for layer_name in bg_layers.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(bg_layers[layer_name], layer_name,
                                                                      sim_params['initial_ignore'])
    if sim_regions['S1']:
        for layer_name in ctx_layers.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name, sim_params['initial_ignore'])
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
            detectors['TH_S1_EZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_EZ'][layer_name], 'TH_S1_EZ_'+layer_name, sim_params['initial_ignore'])
        for layer_name in th_layers['TH_S1_IZ'].keys():
            detectors['TH_S1_IZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_IZ'][layer_name], 'TH_S1_IZ_'+layer_name, sim_params['initial_ignore'])
    if sim_regions['TH_M1']:
        for layer_name in th_layers['TH_M1_EZ'].keys():
            detectors['TH_M1_EZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_EZ'][layer_name], 'TH_M1_EZ_'+layer_name, sim_params['initial_ignore'])
        for layer_name in th_layers['TH_M1_IZ'].keys():
            detectors['TH_M1_IZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_IZ'][layer_name], 'TH_M1_IZ_'+layer_name, sim_params['initial_ignore'])
    print (sim_model_on)
    if sim_model_on=='resting_state':
        simulation_time = sim_params['simDuration']+sim_params['initial_ignore']
        print('Simulation Started:')
        start_time = time.time()
        nest.Simulate(simulation_time)
        with open('./log/' + 'performance.txt', 'a') as file:
            file.write('Simulation_Elapse_Time ' + str(time.time() - start_time) + '\n')
        print ('Simulation Finish')

        if sim_regions['BG']:
            for layer_name in bg_layers.keys():
                rate = nest_routine.get_firing_rate_from_gdf_files(layer_name, detectors[layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(bg_layers[layer_name]))
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['S1']:
            for layer_name in ctx_layers.keys():
                rate = nest_routine.get_firing_rate_from_gdf_files(layer_name,detectors[layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(ctx_layers[layer_name]))
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['M1']:
            for layer_name in ctx_M1_layers.keys():
                rate = nest_routine.get_firing_rate_from_gdf_files(layer_name,detectors[layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(ctx_M1_layers[layer_name]))
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['CB_S1']:
            for layer_name in cb_layers_S1.keys():
                rate = nest_routine.get_firing_rate_from_gdf_files(layer_name,detectors[layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(cb_layers_S1[layer_name]))
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['CB_M1']:
            for layer_name in cb_layers_M1.keys():
                rate = nest_routine.get_firing_rate_from_gdf_files(layer_name, detectors[layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(cb_layers_M1[layer_name]))
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")

        if sim_regions['TH_S1']:
            for layer_name in th_layers['TH_S1_EZ'].keys():
                rate = nest_routine.get_firing_rate_from_gdf_files('TH_S1_EZ_' + layer_name, detectors['TH_S1_EZ_' + layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(th_layers['TH_S1_EZ'][layer_name]))
                print('Layer ' + 'TH_S1_EZ_' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['TH_S1']:
            for layer_name in th_layers['TH_S1_IZ'].keys():
                rate = nest_routine.get_firing_rate_from_gdf_files('TH_S1_IZ_' + layer_name, detectors['TH_S1_IZ_' + layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(th_layers['TH_S1_IZ'][layer_name]))
                print('Layer ' + 'TH_S1_IZ_' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['TH_M1']:
            for layer_name in th_layers['TH_M1_EZ'].keys():
                rate = nest_routine.get_firing_rate_from_gdf_files('TH_M1_EZ_' + layer_name, detectors['TH_M1_EZ_' + layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(th_layers['TH_M1_EZ'][layer_name]))
                print('Layer ' + 'TH_M1_EZ_' + layer_name + " fires at " + str(rate) + " Hz")
        if sim_regions['TH_M1']:
            for layer_name in th_layers['TH_M1_IZ'].keys():
                rate = nest_routine.get_firing_rate_from_gdf_files('TH_M1_IZ_' + layer_name, detectors['TH_M1_IZ_' + layer_name], sim_params['simDuration'],
                                               nest_routine.count_layer(th_layers['TH_M1_IZ'][layer_name]))
                print('Layer ' + 'TH_M1_IZ_' + layer_name + " fires at " + str(rate) + " Hz")


    else:
        print ('wrong model set')



if __name__ == '__main__':
    main()
