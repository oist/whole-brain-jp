#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## nest_routine.py
##
## This script defines creation, connection, and simulation routines using PyNest
##
## It is split in several parts, one for each brain region simulated:
## `CTX`, 'CTX_M2' `TH`, `BH`, `CERE`
##
## Functions should be sufffixed by their regions: `_ctx`, 'ctx_M2' `_th`, `_bg`, `_cb`

import nest.topology as ntop
from nest.lib.hl_api_info import SetStatus
import nest
import numpy as np
import math
import time
import collections
import os
import random

import nest.lib.hl_api_parallel_computing as hapc


###########
# General #
###########
#### global variable to be updated when initializing nest #######
pyrngs = []


# -------------------------------------------------------------------------------
# Create new neuron model
# -------------------------------------------------------------------------------

def copy_neuron_model(elements, neuron_info, new_model_name):
    configuration = {}
    # Membrane potential in mV
    configuration['V_m'] = 0.0
    # Leak reversal Potential (aka resting potential) in mV
    configuration['E_L'] = -70.0
    # Membrane Capacitance in pF
    configuration['C_m'] = 250.0
    # Refractory period in ms
    configuration['t_ref'] = float(neuron_info['absolute_refractory_period'])
    # Threshold Potential in mV
    configuration['V_th'] = float(neuron_info['spike_threshold'])
    # Reset Potential in mV
    configuration['V_reset'] = float(neuron_info['reset_value'])
    # Excitatory reversal Potential in mV
    configuration['E_ex'] = float(neuron_info['E_ex'])
    # Inhibitory reversal Potential in mV
    configuration['E_in'] = float(neuron_info['E_in'])
    # Leak Conductance in nS
    configuration['g_L'] = 250. / float(neuron_info['membrane_time_constant'])
    # Time constant of the excitatory synaptic exponential function in ms
    configuration['tau_syn_ex'] = float(neuron_info['tau_syn_ex'])
    # Time constant of the inhibitory synaptic exponential function in ms
    configuration['tau_syn_in'] = float(neuron_info['tau_syn_in'])
    # Constant Current in pA
    configuration['I_e'] = float(neuron_info['I_ex'])
    nest.CopyModel(elements, new_model_name, configuration)
    return new_model_name

# -------------------------------------------------------------------------------
# Nest initialization
# -------------------------------------------------------------------------------
def initialize_nest(sim_params):
    nest.set_verbosity("M_WARNING")
    nest.SetKernelStatus({"overwrite_files": sim_params[
        'overwrite_files']})  # should we erase previous traces when redoing a simulation?
    nest.SetKernelStatus({'local_num_threads': int(sim_params['nbcpu'])})
    nest.SetKernelStatus({"data_path": 'log'})
    if sim_params['dt'] != '0.1':
        nest.SetKernelStatus({'resolution': float(sim_params['dt'])})
    ####### adding changes to nest seeds for independent experiments ##########
    ### changing python seeds ####
    N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    global pyrngs
    pyrngs = [np.random.RandomState(s) for s in range(sim_params['msd'], sim_params['msd'] + N_vp)]
    ### global nest rng ###
    nest.SetKernelStatus({'grng_seed': sim_params['msd'] + N_vp})
    ### per process rng #####
    nest.SetKernelStatus({'rng_seeds': range(sim_params['msd'] + N_vp + 1, sim_params['msd'] + 2 * N_vp + 1)})


# -------------------------------------------------------------------------------
# Starts the Nest simulation, given the general parameters of `sim_params`
# -------------------------------------------------------------------------------
def run_simulation(sim_params):
    nest.ResetNetwork()
    nest.Simulate(sim_params['simDuration'] + sim_params['initial_ignore'])


# -------------------------------------------------------------------------------
# Instantiate a spike detector and connects it to the entire layer `layer_gid`
# -------------------------------------------------------------------------------
def layer_spike_detector(layer_gid, layer_name, ignore_time,
                         params={"withgid": True, "withtime": True, "to_file": True}):
    # def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    print('spike detector for ' + layer_name)
    params.update({'label': layer_name, "start": float(ignore_time)})
    # params.update({'label': layer_name})
    detector = nest.Create("spike_detector", params=params)
    nest.Connect(pre=nest.GetNodes(layer_gid)[0], post=detector)
    
    return detector


def col_spike_detector(layer_gids_inside, layer_name, ignore_time,
                       params={"withgid": True, "withtime": True, "to_file": True}):
    params.update({'label': layer_name, "start": float(ignore_time)})
    # Add detector for all neuron types
    detector = nest.Create("spike_detector", params=params)
    nest.Connect(pre=layer_gids_inside, post=detector)

    return detector


# -------------------------------------------------------------------------------
# Returns the average firing rate of a population
# It is relative to the simulation duration `simDuration` and the population size `n`
def average_fr(detector, simDuration, n):
    return nest.GetStatus(detector, 'n_events')[0] / (float(simDuration) * float(n) / 1000.)

def average_fr_pre(detector, n, start_time, end_time):
    circle_events = nest.GetStatus(detector, 'events')[0]
    circle_events_times = circle_events['times']
    print('#####################################')
    print(circle_events)
    print(circle_events_times)
    print(len(circle_events_times))
    circle_events_select = circle_events_times[
        np.logical_and(circle_events['times'] >= start_time, circle_events['times'] <= end_time)]
    return len(circle_events_select) / (float(end_time - start_time) * float(n) / 1000.)


# -------------------------------------------------------------------------------
# Returns the average firing rate of one topology layer from gdf files
#
# -------------------------------------------------------------------------------
def get_firing_rate_from_gdf_files(layer_name, layer_gid, simDuration, neuron_nb):
    if hapc.Rank() == 0:
        pop, pop_gid = {}, {}
        gdf_path = 'log/'
        onlyfiles = [f for f in os.listdir(gdf_path) if os.path.isfile(os.path.join(gdf_path, f))]
        is_first = True
        for i in onlyfiles:
            if layer_name == i[:len(layer_name)] and i[-4:] == '.gdf':
                if os.stat(gdf_path + i).st_size > 0:
                    try:
                        a = np.loadtxt(gdf_path + i)
                        if is_first:
                            if len(a) != 0:
                                pop[layer_name] = a
                                is_first = False
                        else:
                            if len(pop[layer_name].shape) == 1:
                                pop[layer_name] = np.expand_dims(pop[layer_name], axis=0)
                            if len(a.shape) == 1:
                                a = np.expand_dims(a, axis=0)
                            pop[layer_name] = np.concatenate((pop[layer_name], a), axis=0)
                    except:
                        pass
        if is_first:
            pop[layer_name] = np.array([])
        if len(pop[layer_name]) > 0:
            return len(pop[layer_name]) / (float(simDuration) * float(neuron_nb) / 1000.)
        else:
            return 0.


# -------------------------------------------------------------------------------
# Returns the number of neurons inside a layer
# -------------------------------------------------------------------------------
def count_layer(layer_gid):
    return len(nest.GetNodes(layer_gid)[0])


# -------------------------------------------------------------------------------
# Returns the positions of neurons inside a layer -sun-20180911
# -------------------------------------------------------------------------------
def get_position(layer_gid):
    return ntop.GetPosition(layer_gid)


# -------------------------------------------------------------------------------
# Returns the connections of neurons inside a layer -sun-20180912
# -------------------------------------------------------------------------------
def get_connection(gids):
    return nest.GetConnections(gids)

# -------------------------------------------------------------------------------
# Generator for efficient looping over local nodes
# Assumes nodes is a continous list of gids [1, 2, 3, ...], e.g., as
# returned by Create. Only works for nodes with proxies, i.e.,
# regular neurons.
# -------------------------------------------------------------------------------
def get_local_nodes(nodes):
    nvp = nest.GetKernelStatus('total_num_virtual_procs')  # step size
    i = 0
    print(len(nodes))
    while i < len(nodes):
        if nest.GetStatus([nodes[i]], 'local')[0]:
            yield nodes[i]
            i += nvp
        else:
            i += 1


def save_layers_position(layer_name, layer_gid, positions):
    gid_and_positions = np.column_stack((np.array(nest.GetNodes(layer_gid)[0]), positions))
    # if not os.path.exists('log/'+layer_name+'.txt'):
    np.savetxt('log/' + layer_name + '.txt', gid_and_positions, fmt='%1.3f')


# -------------------------------------------------------------------------------
# randomizing the membrane potential
#
# -------------------------------------------------------------------------------
def randomizing_mp(layer, Vth, Vrest):
    for neuron in get_local_nodes(nest.GetNodes(layer)[0]):
        nest.SetStatus([neuron], {"V_m": Vrest + (Vth - Vrest) * np.random.rand()})


#######
# CTX #
#######

######
# S1 #
######

def gen_neuron_postions_ctx(layer_dep, layer_thickness, nbneuron, S1_layer_size, scalefactor, pop_name):
    neuron_per_grid = math.pow((nbneuron / layer_thickness), 1.0 / 3)
    Sub_Region_Architecture = [0, 0, 0]
    Sub_Region_Architecture[0] = int(np.round(neuron_per_grid * S1_layer_size[0] * scalefactor[0]))
    Sub_Region_Architecture[1] = int(np.round(neuron_per_grid * S1_layer_size[1] * scalefactor[1]))
    Sub_Region_Architecture[2] = int(np.round(neuron_per_grid * layer_thickness))

    Neuron_pos_x = np.linspace(-0.5 * scalefactor[0], 0.5 * scalefactor[0], num=Sub_Region_Architecture[0],
                               endpoint=True)
    Neuron_pos_y = np.linspace(-0.5 * scalefactor[1], 0.5 * scalefactor[1], num=Sub_Region_Architecture[1],
                               endpoint=True)
    Neuron_pos_z = np.linspace(layer_dep, (layer_dep + layer_thickness), num=Sub_Region_Architecture[2], endpoint=True)

    Neuron_pos = []
    for i in range(Sub_Region_Architecture[0]):
        for j in range(Sub_Region_Architecture[1]):
            for k in range(Sub_Region_Architecture[2]):
                Neuron_pos.append([Neuron_pos_x[i], Neuron_pos_y[j], Neuron_pos_z[k]])
    return Neuron_pos


def create_layers_ctx(extent, center, positions, elements):
    newlayer = ntop.CreateLayer(
        {'extent': extent, 'center': center, 'positions': positions, 'elements': elements, 'edge_wrap': True})
    return newlayer


def add_poisson_generator(layer_gid, n_type, layer_name, ignore_time, region):  #
    ini_time = ignore_time - 150
    ini_time_ini = ignore_time - 300
    if region == 'M1':
        if n_type == "E":
            if layer_name in ['M1_L23_CC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 800.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
            elif layer_name in ['M1_L5A_CC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 900.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5A_CS']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 900.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5A_CT']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 900.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5B_CC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1050.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5B_CS']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1050.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5B_PT']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1050.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L6_CT']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1050.0, "start": float(ini_time)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
        if n_type == "I":
            if layer_name in ['M1_L1_SBC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1100.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L1_ENGC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1100.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L23_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1350.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L23_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 650.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L23_VIP']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1400.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5A_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1300.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5A_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 650.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5B_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1300.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L5B_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 700.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L6_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1350.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
            elif layer_name in ['M1_L6_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 700.0, "start": float(ini_time_ini)})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
    elif region == 'S1':
        if n_type == "E":
            if layer_name in ['S1_L2_Pyr']:
                PSG = nest.Create('poisson_generator', 1,
                                  params={'rate': 450.0, "start": float(ini_time)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L3_Pyr']:
                PSG = nest.Create('poisson_generator', 1,
                                  params={'rate': 400.0, "start": float(ini_time)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L4_Pyr']:
                PSG = nest.Create('poisson_generator', 1,
                                  params={'rate': 400.0, "start": float(ini_time)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L5A_Pyr']:
                PSG = nest.Create('poisson_generator', 1,
                                  params={'rate': 800.0, "start": float(ini_time)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L5B_Pyr']:
                PSG = nest.Create('poisson_generator', 1,
                                  params={'rate': 800.0, "start": float(ini_time)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L6_Pyr']:
                PSG = nest.Create('poisson_generator', 1,
                                  params={'rate': 900.0, "start": float(ini_time)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
        if n_type == "I":
            if layer_name in ['S1_L1_SBC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 600.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L1_ENGC']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 600.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L2_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 600.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L2_VIP']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1000.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L2_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 800.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L3_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 600.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L3_VIP']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1000.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L3_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 800.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L4_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 600.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L4_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 900.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L5A_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 700., "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L5A_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1350.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L5B_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 680., "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L5B_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1350.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L6_SST']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 680.5, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})
            elif layer_name in ['S1_L6_PV']:
                PSG = nest.Create('poisson_generator', 1, params={'rate': 1350.0, "start": float(
                    ini_time_ini)})  # , 'label': layer_name})
                nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 5.0, 'delay': 1.5})

    elif region == 'TH':  ##added by Carlos amigo san.
        if layer_name in [
            'TH_M1_IZ_thalamic_nucleus_TC']:  ## added to compensate the input from BG so TC gets resting state same as other TH populations
            PSG = nest.Create('poisson_generator', 1,
                              params={'rate': 210.0, "start": float(ini_time_ini)})  # higher rate than others.
            nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
        else:
            PSG = nest.Create('poisson_generator', 1, params={'rate': 20.0, "start": float(ini_time_ini)})
            nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
    else:
        print('please check the region name for add poisson generator')


# connect (intra regional connection?)
def connect_layers_ctx(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma'] / 1000.
    sigma_y = conn_dict['sigma'] / 1000.
    weight_distribution = conn_dict['weight_distribution']
    if weight_distribution == 'lognormal':
        conndict = {'connection_type': 'divergent',
                    'mask': {'box': {'lower_left': [-0.5, -0.5, -0.5],
                                     'upper_right': [0.5, 0.5, 0.5]}},
                    'kernel': {
                        'gaussian2D': {'p_center': conn_dict['p_center'], 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': {'lognormal': {'mu': conn_dict['weight'], 'sigma': 1.0}},
                    'delays': conn_dict['delay'],
                    'allow_autapses': False,
                    'allow_multapses': False}
    else:
        conndict = {'connection_type': 'divergent',
                    'mask': {'box': {'lower_left': [-0.5, -0.5, -0.5],
                                     'upper_right': [0.5, 0.5, 0.5]}},
                    'kernel': {
                        'gaussian2D': {'p_center': conn_dict['p_center'], 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': conn_dict['weight'],
                    'delays': conn_dict['delay'],
                    'allow_autapses': False,
                    'allow_multapses': False}
    if conn_dict['p_center'] != 0.0 and sigma_x != 0.0 and conn_dict['weight'] != 0.0:
        ntop.ConnectLayers(pre_SubSubRegion, post_SubSubRegion, conndict)



##### General function for M1, S1, M2 ########################################################################
def get_input_column_layers_ctx(ctx_layers, circle_center, radius_small, my_area):  # my_area is 'M1' or 'S1'
    if my_area == 'M1':
        gid_pos_L5B = np.loadtxt('./log/M1_L5B_PT.txt')  # ntop.GetPosition(gid_M1_L5B_PT)
        gid_pos_L5A = np.loadtxt('./log/M1_L5A_CS.txt')  # ntop.GetPosition(gid_M1_L5B_PT)
    if my_area == 'S1':
        gid_pos_L5B = np.loadtxt('./log/S1_L5B_Pyr.txt')  # ntop.GetPosition(gid_M1_L5B_PT)
        gid_pos_L5A = np.loadtxt('./log/S1_L5A_Pyr.txt')  # ntop.GetPosition(gid_M1_L5B_PT)

    print('gids and pos l5a ', len(gid_pos_L5A))
    print('gids and pos l5b ', len(gid_pos_L5B))

    circle_gids_L5B, circle_gids_L5A = [], []

    for i in np.arange(len(circle_center)):
        idx_L5B = np.where(
            np.linalg.norm([(gid_pos_L5B[:, 1] - circle_center[i][0]), (gid_pos_L5B[:, 2] - circle_center[i][1])],
                           axis=0) <= radius_small)[0]
        print('number of neurons in channel ', str(i), 'for ', my_area, ' L5B: ', str(len(idx_L5B)))
        circle_gids_L5B.append([[int(x[0]), x[1:].tolist()] for x in gid_pos_L5B[idx_L5B, :]])

        idx_L5A = np.where(
            np.linalg.norm([(gid_pos_L5A[:, 1] - circle_center[i][0]), (gid_pos_L5A[:, 2] - circle_center[i][1])],
                           axis=0) <= radius_small)[0]
        print('number of neurons in channel ', str(i), 'for ', my_area, ' L5A: ', str(len(idx_L5A)))
        circle_gids_L5A.append([[int(x[0]), x[1:].tolist()] for x in gid_pos_L5A[idx_L5A, :]])

    return [circle_gids_L5B, circle_gids_L5A]  # circle_gids

######
# TH #
######

def create_layers_th(extent, center, positions, elements, neuron_info):
    newlayer = ntop.CreateLayer(
        {'extent': extent, 'center': center, 'positions': positions, 'elements': elements, 'edge_wrap': True})
    Neurons = nest.GetNodes(newlayer)
    SetStatus(Neurons[0], {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']),
                           "V_reset": float(neuron_info['reset_value']),
                           "t_ref": float(neuron_info['absolute_refractory_period'])})
    return newlayer


# connect (intra regional connection?)
def connect_layers_th(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma']
    sigma_y = conn_dict['sigma']
    conndict = {'connection_type': 'divergent',
                'mask': {'box': {'lower_left': [-0.5, -0.5, -0.5],
                                 'upper_right': [0.5, 0.5, 0.5]}},
                'kernel': {
                    'gaussian2D': {'p_center': conn_dict['p_center'], 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': conn_dict['weight'],
                'delays': conn_dict['delay'],
                'allow_autapses': False,
                'allow_multapses': False}
    if sigma_x != 0 and conn_dict['p_center'] != 0.:
        ntop.ConnectLayers(pre_SubSubRegion, post_SubSubRegion, conndict)


#CB
def create_layers_cb(rows, columns, subCB_name, elements, extent, center):
    print('Create a CB layer: ' + elements[0])
    if elements[0] in [subCB_name + '_layer_gr', subCB_name + '_layer_go', subCB_name + '_layer_vn',
                       subCB_name + '_layer_pkj', subCB_name + '_layer_bs', subCB_name + '_layer_pons']:
        pos_x = np.linspace(-extent[0] / 2., extent[0] / 2., num=int(rows), endpoint=True)
        pos_y = np.linspace(-extent[1] / 2., extent[1] / 2., num=int(columns), endpoint=True)
        pos_z = 0.0
        positions = np.zeros((int(rows * columns), 3))
        for i in range(int(rows)):
            for j in range(int(columns)):
                positions[int(i * columns + j)] = np.array([pos_x[i], pos_y[j], pos_z])
        if elements[1] > 1:
            positions_cluster = np.repeat(positions, elements[1], axis=0)
            layer = ntop.CreateLayer({'extent': [extent[0] + 2., extent[1] + 2., extent[2] + 2.], 'center': center,
                                      'positions': positions_cluster.tolist(), 'elements': elements[0],
                                      'edge_wrap': True})
            save_layers_position(elements[0], layer, positions_cluster)
        else:
            layer = ntop.CreateLayer({'extent': [extent[0] + 2., extent[1] + 2., extent[2] + 2.], 'center': center,
                                      'positions': positions.tolist(), 'elements': elements[0], 'edge_wrap': True})
            save_layers_position(elements[0], layer, positions)
    elif elements[0] == subCB_name + '_layer_io':
        pos_x = 0.0
        pos_y = 0.0
        pos_z = 0.0
        positions = [[pos_x, pos_y, pos_z]]
        layer = ntop.CreateLayer(
            {'extent': [extent[0] + 1., extent[1] + 1., extent[2] + 1.], 'center': center, 'positions': positions,
             'elements': elements[0]})
        save_layers_position(elements[0], layer, positions)
    return layer


def create_neurons_cb():
    import CBneurons
    CBneurons.create_neurons()


######
# BG #
######

AMPASynapseCounter_bg = 0  # initialize global counter variable for AMPA/NMDA colocalization in BG (unfortunate design choice, but required by nest fast connect procedure)


# -------------------------------------------------------------------------------
# Provides circular positions (check whether the position are within the layer dimensions beforehand !)
#
# nbCh: integer stating the number of channels to be created
# c: distance to the center (small distance means more channels in competition)
# r: radius of each channel (leading to larger overlap and thus broader competition)
# -------------------------------------------------------------------------------
# helper function that gives the channel center
def circular_center(nbCh, c, Ch=None):
    # equi-distant points on a circle
    if Ch == None:
        indices = np.arange(0, nbCh, dtype=float) + 0.5
    else:
        indices = np.array(Ch) + 0.5
    angles = (1. - indices / nbCh) * 2. * np.pi
    x, y = np.cos(angles) * c, np.sin(angles) * c
    return {'x': x, 'y': y}


def circular_positions(nbCh, c, r, sim_pts, Ch=None):  # circular positions only work in scale [1,1]
    # N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    # pyrngs = [np.random.RandomState(123456)] # for s in ([123456]*N_vp)]
    if Ch == None:
        Ch = range(nbCh)
    center_xy = circular_center(nbCh, c, Ch=Ch)
    xSim = []
    ySim = []
    for i in range(len(Ch)):
        angleSim = pyrngs[0].uniform(0., 2. * np.pi, int(sim_pts))
        rSim = pyrngs[0].uniform(0., r, int(sim_pts))
        xSim = xSim + (np.cos(angleSim) * rSim + center_xy['x'][i]).tolist()
        ySim = ySim + (np.sin(angleSim) * rSim + center_xy['y'][i]).tolist()
    return (xSim, ySim)

## Function hex_corner provides vertex coordinates of a hexagon, given a center, a radius (size) and the vertex id.
def hex_corner(center, size, i):
    angle_deg = 60 * i - 30
    angle_rad = np.pi / 180 * angle_deg
    return [center[0] + size * np.cos(angle_rad), center[1] + size * np.sin(angle_rad)]


# define the centers that will connect ctx to bg, and store them at bg_params['circle_center']
# centers must be within grid 2D dimensions.
def get_channel_centers(bg_params, hex_center=[0, 0], ci=6, hex_radius=0.240):
    center_aux = []
    if bg_params['channels']:
        if len(bg_params['circle_center']) == 0:  # must be done before bg instantiation.
            for i in np.arange(ci):
                x_y = hex_corner(hex_center, hex_radius,
                                 i)  # center, radius, vertex id # gives x,y of an hexagon vertexs.
                center_aux.append(x_y)
                # bg_params['circle_center'].append(x_y)
            np.savetxt('./log/centers.txt', center_aux)  # save the centers.
            print('generated centers: ', center_aux)
    return center_aux


################################################################################################################################
######### given columns mean firing rate, columns positions and current position, it infers (heuristically) the next positions 
########## attention !: points_to_reach and mean_fr should be aligned !!!! ####################
########### attention !: mean_fr of is sorted as channels R, C, L respectively (based on points_to_reach order -> centers[1:4])
#############################################################################################################################
def get_next_pos(points_to_reach, mean_fr, current_position=[0., 0.], delta_x=0.1):
    xt = current_position  # x,y components between -0.5 and 0.5 (receptive field, this need to be scaled to map physical positions)
    # normalize firing rates 
    a = mean_fr / sum(np.array(
        mean_fr))  # normalization between 3 rates # also can be used -> mean_fr/np.array([150.]) normalize against a maximum allowed.
    # calculate each of their delta_x contribution. x,y components.
    aa = a * delta_x
    xt1 = []
    for j, i in enumerate(points_to_reach):
        theta = math.degrees(math.atan((i[1] - xt[1]) / (i[0] - xt[0])))
        if theta < 0.:
            theta = 180. + theta
        if xt[1] > i[1] and (j == 0 or j == 2):  # only will do this if the L and R are left behind
            theta = 180. + theta
        xt_i = [aa[j] * (math.cos(math.radians(theta))), aa[j] * (math.sin(math.radians(theta)))]
        xt_i[0] = xt_i[0] + xt[0]  # shift
        xt_i[1] = xt_i[1] + xt[1]  # shift
        xt1.append(xt_i)
    xt1 = np.array(xt1)
    for k in np.arange(len(xt1)):
        xt1[k, 0] = xt1[k, 0] - xt[0]  # shift
        xt1[k, 1] = xt1[k, 1] - xt[1]  # shift

    xnew = [xt1[:, 0].sum() + xt[0], xt1[:, 1].sum() + xt[1]]
    
    return xnew


### function to define BG grid positions in 2D
### parameters: 
# nbCh: number of channels (always 1)
# sim_pts: number of points to generate
# a0, a1:  -x shift, distance from starting point (x axis)
# b0, b1:  -y shift, distance from starting point (y axis)
# -----------------------------------------------------
def grid_positions(nbCh, sim_pts, a0, a1, b0, b1):
    # N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    # pyrngs = [np.random.RandomState(123456)]# for s in ([123456]*N_vp)]
    n = int(sim_pts * nbCh)
    n_squared = np.ceil(np.sqrt(n))
    coord = [[x / n_squared * a1 - a0, y / n_squared * b1 - b0] for x in np.arange(0, n_squared, dtype=float) for y in
             np.arange(0, n_squared, dtype=float)]
    # too many points due to square root rounding? remove at random # same random numbers over multiple nodes
    if len(coord) > n:
        coord = np.array(coord)[np.sort(pyrngs[0].choice(range(len(coord)), size=n, replace=False))].tolist()
    aux_x = [coord[i][0] for i in range(len(coord))]
    aux_y = [coord[i][1] for i in range(len(coord))]
    return [aux_x, aux_y]


### function to get connections in format source,target,weigth and save as txt ######
def get_connections_to_file(source_name, target_name, source_layer, target_layer):
    np.savetxt('./log/' + source_name + '_to_' + target_name + '.txt', nest.GetStatus(
        nest.GetConnections(source=nest.GetNodes(source_layer)[0], target=nest.GetNodes(target_layer)[0]),
        keys={'source', 'target', 'weight'}))


# -------------------------------------------------------------------------------
# Establishes a topological layer and returns it
# bg_params: basal ganglia parameters
# nucleus: name of the nucleus to instantiate
# fake: numerical value - if 0, then a real population of iaf is instantiated
#                       - if fake > 0, then a Poisson generator population firing at `fake` Hz is instantiated
# force_pop_size: if defined, initialize only this number of neurons
#                 -> this is useful for the cortical connections, as some inputs will be derived from L5A and L5B layers
# -------------------------------------------------------------------------------
def create_layers_bg(bg_params, nucleus, fake=0, mirror_neurons=None, mirror_pos=None, scalefactor=[1, 1]):
    # N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    # pyrngs = [np.random.RandomState(123456)]# for s in ([123456]*N_vp)]
    # print('global pyrngs in create layers BG: ',pyrngs)

    # define extent and center for 2D layer
    my_extent = [1. * int(scalefactor[0]) + 1., 1. * int(scalefactor[1]) + 1.]
    my_center = [0.0, 0.0]

    if mirror_neurons is None:
        # normal case: full input layer is created
        if nucleus == 'GPi_fake':
            nucleus_tmp = nucleus[:3]
            pop_size = int(bg_params['nb' + nucleus_tmp])
        else:
            pop_size = int(bg_params['nb' + nucleus])
    else:
        # inputs come from existing ctx layer: only a fraction of poisson generators are created
        pop_size = int(bg_params['nb' + nucleus]) - len(mirror_neurons)

    print('population size for ' + nucleus + ': ' + str(pop_size))

    if nucleus == 'GPi_fake':
        # get xy position from real GPi and add z value
        positions_z = pyrngs[0].uniform(0., 0.5, pop_size).tolist()
        positions = np.loadtxt('./log/' + nucleus[:3] + '.txt')  # retrive positions x,y from GPi
        position_nD = [[positions[i][1], positions[i][2], positions_z[i]] for i in range(len(positions))]
        # define extent and center for 3D layer
        my_extent = my_extent + [1.]
        my_center = my_center + [0.]
        print('positions GPi_fake: ', position_nD[:10])
    else:
        if (nucleus == 'GPi' or nucleus == 'STN'):
            positions = grid_positions(1, pop_size, 0.4 * scalefactor[0], scalefactor[0] - 0.1, 0.4 * scalefactor[1],
                                       scalefactor[1] - 0.1)
        else:
            positions = grid_positions(1, pop_size, 0.5 * scalefactor[0], scalefactor[0], 0.5 * scalefactor[1],
                                       scalefactor[1])
        position_nD = [[positions[0][i], positions[1][i]] for i in range(len(positions[0]))]

    if mirror_neurons != None:
        mirror_neurons.sort(key=lambda x: x[0])  # sort by Gids and arrange related positions
        mirror_gids = [gids[0] for gids in mirror_neurons]
        mirror_pos = [pos[1] for pos in mirror_neurons]

        # add all positions together
        print('mirror neurons!   original position_nD: ', len(position_nD), '   ', position_nD[:3])
        print('mirror neurons!  mirror positions: ', len(mirror_pos), '  ', mirror_pos[:3])
        position_nD = position_nD + mirror_pos
        print('positions len for fake including mirrors: ', len(position_nD))

    if fake == 0:
        # fake == 0 is the normal case, where actual iaf neurons are instantiated
        if nucleus == 'GPi_fake':
            element = 'parrot_neuron'
        else:
            nest.SetDefaults('iaf_psc_alpha_multisynapse', bg_params['common_iaf'])
            nest.SetDefaults('iaf_psc_alpha_multisynapse', bg_params[nucleus + '_iaf'])
            nest.SetDefaults('iaf_psc_alpha_multisynapse', {"I_e": bg_params['Ie' + nucleus]})
            element = 'iaf_psc_alpha_multisynapse'
    else:
        # when fake > 0, parrot neurons instantiated
        element = 'parrot_neuron'
    layer_gid = ntop.CreateLayer(
        {'positions': position_nD, 'elements': element, 'extent': my_extent, 'center': my_center, 'edge_wrap': True})
    print(len(position_nD))
    save_layers_position(nucleus, layer_gid, np.array(position_nD))
    if fake > 0:
        # when fake > 0, parrot neurons are connected to poisson generators firing at `fake`Hz
        my_post = list(nest.GetNodes(layer_gid)[0])
        my_post.sort()

        if True:
            poisson = nest.Create('poisson_generator', 1)
            nest.SetStatus(poisson, {'rate': fake})
            poisson_string = poisson * pop_size
            nest.Connect(pre=poisson_string, post=my_post[0:pop_size], conn_spec={'rule': 'one_to_one'})
        
        if mirror_neurons != None:
            print(
                'special handling of ' + nucleus + ' input layer => the remaining neurons will be connected to the original ctx neurons')
            print('connecting mirror neurons of len: ', len(mirror_gids), ' to ', nucleus)
            nest.Connect(pre=mirror_gids, post=my_post[-len(mirror_gids):], conn_spec={'rule': 'one_to_one'},
                         syn_spec={'delay': 10.})  ## added delay !!!!

    return layer_gid


# -------------------------------------------------------------------------------
# Establishes a topological connection between two populations
# bg_params: basal ganglia parameters
# nType : a string 'ex' or 'in', defining whether it is excitatory or inhibitory
# bg_layers : dictionary of basal ganglia layers
# nameTgt, nameSrc : strings naming the populations, as defined in NUCLEI list
# projType : type of projections. For the moment: 'focused' (only channel-to-channel connection) and
#            'diffuse' (all-to-one with uniform distribution)
# redundancy, RedundancyType : contrains the inDegree - see function `connect` for details
# LCGDelays : shall we use the delays obtained by (Liénard, Cos, Girard, in prep) or not (default = True)
# gain : allows to amplify the weight normally deduced from LG14
# stochastic_delays: to enable stochasticity in the axonal delays
# spreads: a 2-item list specifying the radius of focused and diffuse projections
# -------------------------------------------------------------------------------
def connect_layers_bg(bg_params, nType, bg_layers, nameSrc, nameTgt, projType, redundancy, RedundancyType,
                      LCGDelays=True, gain=1., stochastic_delays=None, spreads=None, verbose=False, scalefactor=[1, 1]):
    def printv(text):
        if verbose:
            print(text)

    printv("\n* connecting " + nameSrc + " -> " + nameTgt + " with " + nType + " " + projType + " connection")

    recType = {'AMPA': 1, 'NMDA': 2, 'GABA': 3}

    if RedundancyType == 'inDegreeAbs':
        # inDegree is already provided in the right form
        inDegree = float(redundancy)
    elif RedundancyType == 'outDegreeAbs':
        #### fractional outDegree is expressed as a fraction of max axo-dendritic contacts
        inDegree = get_frac_bg(bg_params, 1. / redundancy, nameSrc, nameTgt, bg_params['count' + nameSrc],
                               bg_params['count' + nameTgt], verbose=verbose)
    elif RedundancyType == 'outDegreeCons':
        #### fractional outDegree is expressed as a ratio of min/max axo-dendritic contacts
        inDegree = get_frac_bg(bg_params, redundancy, nameSrc, nameTgt, bg_params['count' + nameSrc],
                               bg_params['count' + nameTgt], useMin=True, verbose=verbose)
    else:
        raise KeyError('`RedundancyType` should be one of `inDegreeAbs`, `outDegreeAbs`, or `outDegreeCons`.')

    # check if in degree acceptable (not larger than number of neurons in the source nucleus)
    if projType == 'focused' and inDegree > bg_params['nb' + nameSrc]:
        printv("/!\ WARNING: required 'in degree' (" + str(
            inDegree) + ") larger than number of neurons in individual source channels (" + str(
            bg_params['nb' + nameSrc]) + "), thus reduced to the latter value")
        inDegree = bg_params['nb' + nameSrc]
    if projType == 'diffuse' and inDegree > bg_params['nb' + nameSrc]:
        printv("/!\ WARNING: required 'in degree' (" + str(
            inDegree) + ") larger than number of neurons in the overall source population (" + str(
            bg_params['nb' + nameSrc]) + "), thus reduced to the latter value")
        inDegree = bg_params['nb' + nameSrc]

    if inDegree == 0.:
        printv("/!\ WARNING: non-existent connection strength, will skip")
        return

    global AMPASynapseCounter_bg

    # prepare receptor type lists:
    if nType == 'ex':
        lRecType = ['AMPA', 'NMDA']
        AMPASynapseCounter_bg = AMPASynapseCounter_bg + 1
        lbl = AMPASynapseCounter_bg  # needs to add NMDA later
    elif nType == 'AMPA':
        lRecType = ['AMPA']
        lbl = 0
    elif nType == 'NMDA':
        lRecType = ['NMDA']
        lbl = 0
    elif nType == 'in':
        lRecType = ['GABA']
        lbl = 0
    else:
        raise KeyError('Undefined connexion type: ' + nType)

    # compute the global weight of the connection, for each receptor type:
    W = computeW_bg(bg_params, lRecType, nameSrc, nameTgt, inDegree, gain, verbose=verbose)

    printv("  W=" + str(W) + " and inDegree=" + str(inDegree))

    # determine which transmission delay to use:
    if LCGDelays:
        delay = bg_params['tau'][nameSrc + '->' + nameTgt]
    else:
        delay = 1.

    if projType == 'focused':  # if projections focused, input come only from the same channel as tgtChannel
        mass_connect_bg(bg_params, bg_layers, nameSrc, nameTgt, lbl, inDegree, recType[lRecType[0]], W[lRecType[0]],
                        delay, spread=bg_params['spread_focused'], stochastic_delays=stochastic_delays, verbose=verbose)
    elif projType == 'diffuse':  # if projections diffused, input connections are shared among each possible input channel equally
        mass_connect_bg(bg_params, bg_layers, nameSrc, nameTgt, lbl, inDegree, recType[lRecType[0]], W[lRecType[0]],
                        delay, spread=bg_params['spread_diffuse'] * max(scalefactor),
                        stochastic_delays=stochastic_delays, verbose=verbose)

    if nType == 'ex':
        # mirror the AMPA connection with similarly connected NMDA connections
        src_idx = 0
        mass_mirror_bg(bg_params, nameSrc, nameTgt, nest.GetNodes(bg_layers[nameSrc])[src_idx], lbl, recType['NMDA'],
                       W['NMDA'], delay, stochastic_delays=stochastic_delays)

    return W


# ------------------------------------------------------------------------------
# Routine to perform the fast connection using nest built-in `connect` function
# - `bg_params` is basal ganglia parameters
# - `bg_layers` is the dictionary of basal ganglia layers
# - `sourceName` & `destName` are names of two different layers
# - `synapse_label` is used to tag connections and be able to find them quickly
#   with function `mass_mirror`, that adds NMDA on top of AMPA connections
# - `inDegree`, `receptor_type`, `weight`, `delay` are Nest connection params
# - `spread` is a parameter that affects the diffusion level of the connection
# ------------------------------------------------------------------------------
def mass_connect_bg(bg_params, bg_layers, sourceName, destName, synapse_label, inDegree, receptor_type, weight, delay,
                    spread, stochastic_delays=None, verbose=False):
    def printv(text):
        if verbose:
            print(text)

    # potential initialization of stochastic delays
    if stochastic_delays != None and delay > 0:
        printv('Using stochastic delays in mass-connect')
        low = delay * 0.5
        high = delay * 1.5
        sigma = delay * stochastic_delays
        delay = {'distribution': 'normal_clipped', 'low': low, 'high': high, 'mu': delay, 'sigma': sigma}

    ## set default synapse model with the chosen label
    nest.SetDefaults('static_synapse_lbl', {'synapse_label': synapse_label, 'receptor_type': receptor_type})

    # creation of the topological connection dict
    conndict = {'connection_type': 'convergent',
                'mask': {'circular': {'radius': spread}},
                'synapse_model': 'static_synapse_lbl', 'weights': weight, 'delays': delay,
                'allow_oversized_mask': True, 'allow_multapses': True}

    # The first call ensures that all neurons in `destName`
    # have at least `int(inDegree)` incoming connections
    integer_inDegree = np.floor(inDegree)
    if integer_inDegree > 0:
        printv('Adding ' + str(
            int(integer_inDegree * bg_params['nb' + destName])) + ' connections with rule `fixed_indegree`')
        integer_conndict = conndict.copy()
        integer_conndict.update({'number_of_connections': int(integer_inDegree)})
        ntop.ConnectLayers(bg_layers[sourceName], bg_layers[destName], integer_conndict)

    # The second call distributes the approximate number of remaining axonal
    # contacts at random (i.e. the remaining fractional part after the first step)
    # Why "approximate"? Because with pynest layers, there are only two ways to specify
    # the number of axons in a connection:
    #    1) with an integer, specified with respect to each source (alt. target) neurons
    #    2) as a probability
    # Here, we have a fractional part - not an integer number - so that leaves us option 2.
    # However, because the new axonal contacts are drawn at random, we will not have the
    # exact number of connections
    float_inDegree = inDegree - integer_inDegree
    remaining_connections = np.round(float_inDegree * bg_params['nb' + destName])
    if remaining_connections > 0:
        printv('Adding ' + str(remaining_connections) + ' remaining connections with rule `fixed_total_number`')
        float_conndict = conndict.copy()
        float_conndict.update({'kernel': 1. / (bg_params['nb' + sourceName] * float(remaining_connections))})
        ntop.ConnectLayers(bg_layers[sourceName], bg_layers[destName], float_conndict)


# ------------------------------------------------------------------------------
# Routine to duplicate a connection made with a specific receptor, with another
# receptor (typically to add NMDA connections to existing AMPA connections)
# - `source` & `synapse_label` should uniquely define the connections of
#   interest - typically, they are the same as in the call to `mass_connect`
# - `receptor_type`, `weight`, `delay` are Nest connection params
# ------------------------------------------------------------------------------
def mass_mirror_bg(bg_params, nameSrc, nameTgt, source, synapse_label, receptor_type, weight, delay, stochastic_delays,
                   verbose=False):
    def printv(text):
        if verbose:
            print(text)

    # find all AMPA connections for the given projection type
    printv('looking for AMPA connections to mirror with NMDA...\n')
    ampa_conns = nest.GetConnections(source=source, synapse_label=synapse_label)
    # in rare cases, there may be no connections, guard against that
    if ampa_conns:
        # extract just source and target GID lists, all other information is irrelevant here
        printv('found ' + str(len(ampa_conns)) + ' AMPA connections\n')
        if stochastic_delays != None and delay > 0:
            printv('Using stochastic delays in mass-miror')
            delay = np.array(nest.GetStatus(ampa_conns, keys=['delay'])).flatten()
        src, tgt, _, _, _ = zip(*ampa_conns)
        nest.Connect(src, tgt, 'one_to_one',
                     {'model': 'static_synapse_lbl',
                      'synapse_label': synapse_label, # tag with the same number (doesn't matter)
                      'receptor_type': receptor_type, 'weight': weight, 'delay':delay})

# -------------------------------------------------------------------------------
# Helper function to set a basal ganglia internal projection
# computes the inDegree as a fraction of maximal possible inDegree
# `FractionalOutDegree` is the outDegree, expressed as a fraction
# -------------------------------------------------------------------------------
def get_frac_bg(bg_params, FractionalOutDegree, nameSrc, nameTgt, cntSrc, cntTgt, useMin=False, verbose=False):
    if useMin == False:
        # 'FractionalOutDegree' is taken to be relative to the maximal number of axo-dendritic contacts
        inDegree = get_input_range_bg(bg_params, nameSrc, nameTgt, cntSrc, cntTgt, verbose=verbose)[
                       1] * FractionalOutDegree
    else:
        # 'FractionalOutDegree' is taken to be relative to the maximal number of axo-dendritic contacts and their minimal number
        r = get_input_range_bg(bg_params, nameSrc, nameTgt, cntSrc, cntTgt, verbose=verbose)
        inDegree = (r[1] - r[0]) * FractionalOutDegree + r[0]
    if verbose:
        print('\tConverting the fractional outDegree of ' + nameSrc + ' -> ' + nameTgt + ' from ' + str(
            FractionalOutDegree) + ' to inDegree neuron count: ' + str(
            round(inDegree, 2)) + ' (relative to minimal value possible? ' + str(useMin) + ')')
    return inDegree


# -------------------------------------------------------------------------------
# Helper function to set a basal ganglia internal projection
# computes the weight of a connection, based on LG14 parameters
# -------------------------------------------------------------------------------
def computeW_bg(bg_params, listRecType, nameSrc, nameTgt, inDegree, gain=1., verbose=False):
    recType = {'AMPA': 1, 'NMDA': 2, 'GABA': 3}
    nu = get_input_range_bg(bg_params, nameSrc, nameTgt, bg_params['count' + nameSrc], bg_params['count' + nameTgt],
                            verbose=verbose)[1]
    if verbose:
        print('\tCompare with the effective chosen inDegree   : ' + str(inDegree))

    # attenuation due to the distance from the receptors to the soma of tgt:
    LX = bg_params['lx'][nameTgt] * np.sqrt((4. * bg_params['Ri']) / (bg_params['dx'][nameTgt] * bg_params['Rm']))
    attenuation = np.cosh(LX * (1 - bg_params['distcontact'][nameSrc + '->' + nameTgt])) / np.cosh(LX)

    w = {}
    for r in listRecType:
        w[r] = nu / float(inDegree) * attenuation * bg_params['wPSP'][recType[r] - 1] * gain
    return w


# -------------------------------------------------------------------------------
# Helper function to set a basal ganglia internal projection
# returns the minimal & maximal numbers of distinct input neurons for one connection
# -------------------------------------------------------------------------------
def get_input_range_bg(bg_params, nameSrc, nameTgt, cntSrc, cntTgt, verbose=False):
    if nameSrc == 'CSN' or nameSrc == 'PTN':
        nu = bg_params['alpha'][nameSrc + '->' + nameTgt]
        nu0 = 0
        if verbose:
            print('\tMaximal number of distinct input neurons (nu): ' + str(nu))
            print('\tMinimal number of distinct input neurons     : unknown (set to 0)')
    else:
        nu = cntSrc / float(cntTgt) * bg_params['ProjPercent'][nameSrc + '->' + nameTgt] * bg_params['alpha'][
            nameSrc + '->' + nameTgt]
        nu0 = cntSrc / float(cntTgt) * bg_params['ProjPercent'][nameSrc + '->' + nameTgt]
        if verbose:
            print('\tMaximal number of distinct input neurons (nu): ' + str(nu))
            print('\tMinimal number of distinct input neurons     : ' + str(nu0))
    return [nu0, nu]



####################
# INTERCONNECTIONS #
####################

# -------------------------------------------------------------------------------
# If bgparams['channel'] is False it will
# identify randomly a pool of ctx neurons to project to bg
# (numb_neurons (bgparams['num_neurons']) are selected from l5a and l5b)
# if bgparams['channel'] (channels) is True it will select circular clusters with a
# of radius equal to radius_small for each center circle_center;
#
# -------------------------------------------------------------------------------

def identify_proj_neurons_ctx_bg_last(source_layer, params, numb_neurons, area_lbl, channels=False,
                                      channels_radius=0.16, circle_center=[]):

    if area_lbl == 'M1':
        L5A = area_lbl + '_' + 'L5A_CS'
        L5B = area_lbl + '_' + 'L5B_PT'
        my_area = 'M1'
        if channels:
            circles_l5b_l5a = get_input_column_layers_ctx(source_layer, circle_center, channels_radius,
                                                          'M1')  # A list with centers is sent as param
            my_PTN, my_CSN = [], []
            for i in circles_l5b_l5a[0]:  # l5b circles. iterate over circles
                for j in i:
                    my_PTN.append([j[0], j[1][:2]])  # use only x and y positions in PTN
            for i in circles_l5b_l5a[1]:  # l5a circles. iterate over circles
                for j in i:
                    my_CSN.append([j[0], j[1][:2]])  # use only x and y positions in CSN
        else:

            Neuron_pos_fileload = np.loadtxt('./log/' + L5A + '.txt')
            l5a_pos = Neuron_pos_fileload[:, 1:]
            Neuron_pos_fileload = np.loadtxt('./log/' + L5B + '.txt')
            l5b_pos = Neuron_pos_fileload[:, 1:]

            l5a_gids = nest.GetNodes(source_layer[L5A])[0]
            l5b_gids = nest.GetNodes(source_layer[L5B])[0]
            print('gids and pos l5a ', len(l5a_gids), len(l5a_pos))
            print('gids and pos l5b ', len(l5b_gids), len(l5b_pos))
            aux_l5a = np.arange(len(l5a_gids) - 10)
            aux_l5b = np.arange(len(l5b_gids) - 10)
            pyrngs[0].shuffle(aux_l5a)
            pyrngs[0].shuffle(aux_l5b)
            idx_l5a = aux_l5a[:numb_neurons]  # indexes of selected gids within safe range
            idx_l5b = aux_l5b[:numb_neurons]  # indexes of selected gids within safe range
            #### FIX for multiple nodes #######
            my_CSN = [[l5a_gids[i], l5a_pos[i, :2].tolist()] for i in idx_l5a]
            my_PTN = [[l5b_gids[j], l5b_pos[j, :2].tolist()] for j in idx_l5b]
            circles_l5b_l5a = []

        out = {'CSN': my_CSN, 'PTN': my_PTN, 'M1_CIR_L5A_L5B': circles_l5b_l5a}
        print('check lens of samples: ', len(out['CSN']), len(out['PTN']))  # ,
        print('check elements: ', out['CSN'][:3], '    ', out['PTN'][:3])
        return out

    if area_lbl == 'S1':
        L5A = area_lbl + '_' + 'L5A_Pyr'
        L5B = area_lbl + '_' + 'L5B_Pyr'
        my_area = 'S1'

        if channels:
            circles_l5b_l5a = get_input_column_layers_ctx(source_layer, circle_center, channels_radius,
                                                          'S1')  # A list with centers is sent as param
            my_PTN, my_CSN = [], []
            for i in circles_l5b_l5a[0]:  # l5b circles. iterate over circles
                for j in i:
                    my_PTN.append([j[0], j[1][:2]])  # use only x and y positions in PTN
            for i in circles_l5b_l5a[1]:  # l5a circles. iterate over circles
                for j in i:
                    my_CSN.append([j[0], j[1][:2]])  # use only x and y positions in CSN
        else:

            Neuron_pos_fileload = np.loadtxt('./log/' + L5A + '.txt')
            l5a_pos = Neuron_pos_fileload[:, 1:]
            Neuron_pos_fileload = np.loadtxt('./log/' + L5B + '.txt')
            l5b_pos = Neuron_pos_fileload[:, 1:]

            l5a_gids = nest.GetNodes(source_layer[L5A])[0]
            l5b_gids = nest.GetNodes(source_layer[L5B])[0]
            print('gids and pos l5a ', len(l5a_gids), len(l5a_pos))
            print('gids and pos l5b ', len(l5b_gids), len(l5b_pos))
            aux_l5a = np.arange(len(l5a_gids) - 10)
            aux_l5b = np.arange(len(l5b_gids) - 10)
            pyrngs[0].shuffle(aux_l5a)
            pyrngs[0].shuffle(aux_l5b)
            idx_l5a = aux_l5a[
                      :numb_neurons]  # random.sample(range(len(l5a_gids)-10), numb_neurons) #indexes of selected gids within safe range
            idx_l5b = aux_l5b[
                      :numb_neurons]  # random.sample(range(len(l5b_gids)-10), numb_neurons) #indexes of selected gids within safe range
            #### FIX for multiple nodes #######
            my_CSN = [[l5a_gids[i], l5a_pos[i, :2].tolist()] for i in idx_l5a]
            my_PTN = [[l5b_gids[j], l5b_pos[j, :2].tolist()] for j in idx_l5b]
            circles_l5b_l5a = []

        out = {'CSN': my_CSN, 'PTN': my_PTN, 'S1_CIR_L5A_L5B': circles_l5b_l5a}
        print('check lens of samples: ', len(out['CSN']), len(out['PTN']))  # ,
        print('check elements: ', out['CSN'][:3], '    ', out['PTN'][:3])
        return out

    if area_lbl == 'M2':
        print('ADD NEURONS TYPES FOR M2 !!!! ')


# -------------------------------------------------------------------------------
# Connect the ctx neurons of the chosen subset to the basal ganglia
# -------------------------------------------------------------------------------
def connect_ctx_bg(ctx_neurons_gid, bg_layer_gid):
    # import ipdb; ipdb.set_trace()
    nest.Connect(pre=ctx_neurons_gid, post=nest.GetNodes(bg_layer_gid)[0][-len(ctx_neurons_gid):],
                 conn_spec={'rule': 'one_to_one'})

def connect_GPi2d_GPi3d(GPi2d, GPi3d):  # connect GPi layer to fake GPi
    my_pre = nest.GetNodes(GPi2d)[0]
    my_post = nest.GetNodes(GPi3d)[0]
    nest.SetDefaults('static_synapse', {'receptor_type': 0})
    nest.Connect(pre=my_pre, post=my_post, conn_spec={'rule': 'one_to_one'})

# -------------------------------------------------------------------------------
# connect inter regions
#
# -------------------------------------------------------------------------------
def connect_inter_regions(pre_region_name, post_region_name, conn_params, wb_layers):
    region_params_list = conn_params[pre_region_name][post_region_name]
    for sub_region_conns in region_params_list:
        if pre_region_name == 'TH':
            ntop.ConnectLayers(wb_layers[sub_region_conns['pre'][0]][sub_region_conns['pre'][1]],
                               wb_layers[sub_region_conns['post']], sub_region_conns['conn_dict'])
        elif post_region_name == 'TH':
            ntop.ConnectLayers(wb_layers[sub_region_conns['pre']],
                               wb_layers[sub_region_conns['post'][0]][sub_region_conns['post'][1]],
                               sub_region_conns['conn_dict'])
        else:
            ntop.ConnectLayers(wb_layers[sub_region_conns['pre']], wb_layers[sub_region_conns['post']],
                               sub_region_conns['conn_dict'])


#### fix to mitigate edge effect #####

def reduce_weights_at_edges(pre_pos_file_name,post_pos_file_name,pre_pop,post_pop,margin=0.025,new_weight=0.):
### get target and source nodes, and positions of sources
    pre_l_nodes=nest.GetNodes(pre_pop)[0]
    post_l_nodes = nest.GetNodes(post_pop)[0]
    post_l_pos = np.loadtxt('./log/'+post_pos_file_name+'.txt')
    
    #### get the sources (targets) located in the edges 
    x,y = np.array([i[1] for i in post_l_pos]),np.array([i[2] for i in post_l_pos])
    margin = margin 
    idx = []
    for i,j in zip([x.min(),x.max()],[y.min(),y.max()]):
        idx_x = np.where(abs(x-i)<margin)[0]
        idx_y = np.where(abs(y-j)<margin)[0]
        idx.append(idx_x.flatten())
        idx.append(idx_y.flatten())
    edges_target = np.array(post_l_nodes)[np.array(idx).flatten()]

    #### Get the connections from edges and update their weight to zero.
    conns = nest.GetConnections(source=pre_l_nodes, target=list(edges_target))
    nest.SetStatus(conns, {'weight':new_weight})
    return



