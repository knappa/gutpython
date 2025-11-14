# gutpython
Python implementation of [GutLogo](https://github.com/GutLogo/GutLogo).

# Installation

1. Download the source from github. (You can download and extract the zip or use git. I suggest git.)
2. Open up a terminal and move to the gutpython directory.
3. Create a virtual environment using: 
    ```commandline
    python -m virtualenv venv
    ```
    Then enter the environment using:
    ```commandline
    source venv/bin/activate
    ```
    You should see your command prompt change. (In most unix environments, `(venv)` is prepended to the prompt.)

4. Install the gutpython to the virtual environment using `pip install -e .` To also install some optional dependencies (recommended), instead use `pip install -e .[all]`  

Once this is done, you should be able to instantiate the model using:
```python
import gutpython
model = gutpython.GutPython()
```
To advance time on step use `model.go()`. 

# Usage

In progress! In the meantime, inside the venv, run 
```shell
jupyter-notebook gutpython-demo.ipynb
```
where you will find some example usage and visualizations.

# Notes

We are making heavy use of `attrs` features; all of the model's fields are annotated with metadata. Right now, each field keep track of it's type, not in the sense of `int`/`float`/`bool`, but in the sense of whether the field is:
* A parameter: Values which describe the fundamental dynamics of the system and may vary between individuals.
* A control: Parameter-like values or functions of time which represent inputs to the system.
* A measurement: Observable properties of the system.
* An agent property: Self-explanatory; as the number of instances of each agent can vary, these have a internal representation which stores the properties in an array which is larger than the number of actual agents. You must use masks to select the valid values. e.g. `model.bifido_locations[model.bifido_mask,:]` gives an `Nx2` numpy array of actual locations of bifidobacteria, while `model.bifido_locations` will also include invalid data.
* A bookkeeping convenience: e.g. the masks referenced above. These are needed during model runs, but do not need to be saved.
* A molecular distribution: Molecular densities/amounts at each spatial patch.
