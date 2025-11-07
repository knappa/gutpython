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

4. Install the gutpython to the virtual environment using `pip install -e .`

Once this is done, you should be able to instantiate the model using:
```python
import gutpython
model = gutpython.GutPython()
```
To advance time on step use `model.go()`. 

# Usage

In progress! In the meantime, inside the venv, run 
```python
jupyter-notebook gutpython-demo.ipynb
```
where you will find some example usage and visualizations.