### Important: Behaviour Clone Exersice contained in this repository requires Keras 1.2.2 and tensorflow 0.12.1 running on Python 3.5

You will find it convinient to use conda or virtual env to create a sandboxed python environment.

## Dependencies

Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases

You will find it convinient to place the simulation binaries in the root of this repo.

Note, to install on Linux or MacOS, you will need to execute the following command in a console:

```shell
chmod +x donkey_sim
```

Install master version of gym donkey car:

```shell
pip install git+https://github.com/AravindaDP/gym-donkeycar
```

Note: Above fork is modified to work with python 3.5

You might need following python packages as well. (Note these specific versions of keras and tensorflow is only available in python 3.5)

Installation on a conda environment
```
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install h5py
pip install keras==1.2.2
pip install tensorflow==0.12.1
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
exe_path = ".\\DonkeySimWin\\donkey_sim.exe"
```