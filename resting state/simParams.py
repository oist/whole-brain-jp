#!/usr/bin/env python 
## ID string of experiment:
#

## Reproducibility info:
#  platform = Local
#  git commit ID not available
#  Git status not available

simParams =\
{
    "channels": False,
    "channels_nb": 6,
    "channels_radius": 0.12,
    "circle_center": [],
    "dt": "0.1",
    "hex_radius": 0.24,
    "initial_ignore": 500.0,
    "macro_columns_nb": 7,
    "micro_columns_nb": 7,
    "msd": 123456,
    "nbcpu": 10,
    "nbnodes": 1,
    "overwrite_files": True,
    "scalefactor": [
        1.0,
        1.0
    ],
    "simDuration": 1500.0,
    "sim_model": {
        "resting_state": {
            "on": True,
            "regions": {
                "BG": True,
                "CB_M1": True,
                "CB_S1": True,
                "M1": True,
                "S1": True,
                "TH_M1": True,
                "TH_S1": True
            },
        },
    },
    "whichSim": "stim_all_model"
}