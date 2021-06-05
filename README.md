## Dependencies

Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases

You will find it convinient to place the simulation binaries in the root of this repo.

Note, to install on Linux or MacOS, you will need to execute the following command in a console:

```shell
chmod +x donkey_sim
```

Install master version of gym donkey car:

```shell
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

You might need following python packages as well.

Installation on a conda environment
```
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
```

## Instructions

There are couple of demo files (starting with a number in file name). Follow along the code and look out for TODO comments for places where tuning is required and obtaining verbose output for debugging.

There is also couple of associated jupyter notebooks illustrating concepts used in each demo.

Remember to update simulation binary path at the top of each demo.
```python
# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
# This assumes DonkeySim simulator is extracted at root of this repo.
exe_path = f".\\DonkeySimWin\\donkey_sim.exe"
```