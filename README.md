# whole-brain-jp
whole-brain model by OIST, RIKEN and UEC, Japan.

<p align="center">
  <img width="300" src="https://github.com/oist/whole-brain-jp/blob/main/WB.png">
</p>



## Running the whole-brain:

Previous to a simulation, check that the folder ```log``` is empty. The ```gdf``` files containing the spikes of the different populations will be stored at ```log```.

This version can run on a desktop PC, we recommend at least using 10 threads for this simulation (the running time is 30 min. approximately for 1s of biological time). Please check the file ```simParams.py``` for changing the main simulation settings:

```dt```: time bin (we recommend 0.1 ms)

```nbcpu```: the number of threads (at least 10).

```simDuration```: 1500.0 ms (the simulation will add 500ms for initial settings, then it is needed at least 500ms for network stabilization. The last 1000.ms corresponds to resting state).

```regions```: to activate/deactivate regions.

To start a simulation run from command line:
```python stim_all_model.py```

For running on HPC clusters, please include the command above within a job (i.e SLURM), and adjust ```nbcpu``` accordingly (you may change to 1 if any issue on HPC, and manage the computational resources directly from the job file).

## Dependencies:
NEST versions (2.16 ~ 2.20)

Python version: 3.0 ~



## Notes:
If this work was used in your research, we ask kindly to refer our work. Thank you !.





