## Project 3 in FYS-STK4155

Project authors: Amund Solum Alsvik, Jørgen Vestly & Kristoffer DH Stalker  

This repository contains all source code and figures for our research, using PINNs for solving the 1D heat diffusion equation. 
To run the code, run the Jupyter notebooks, and the classes.py file.

<p align="center">
  <img src="Figures/heat_rod.gif" width="600">
</p>


## Installation

All required Python packages are listed in `requirements.txt`.  
All imports used in the project are handled via the files in the `Code/` directory.

### 1. Setting up the environment

All required Python packages are listed in `requirements.txt`.

After cloning the repository, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```
### Contents   

**Code**  
The Code folder contains:

- classes.py: Contains the PINN class, also contains heatmap test code for different activation functions and optimizers.  
- functions.py: contains all utilities related to the project, such as forward Euler algorithm and analytical solution.
- imports.py: contains all imports needed for the running the code.
- The following test files in jupyter notebooks, useful for benchmarking and as setup examples:  
  - tests.ipynb: contains, in principle all test code for the forward Euler algorithm, but also 3D animations for our solution.
  - tests_d.ipynb: contains test code for model selection of PINN, meaning finding the best set of hyperparameters yielding the best performance. GridSearchProsjekt3 contains the whole file for all different models. The best achieved was: MSE: 2.144e-08 | LR=0.001, epochs=5000, layers=4, nodes=100, act=gelu | time=43.42s. Use ctrl+F in this file to check this.
  - tests_helper.ipynb: contains additional testcode for doing nonlinear least squares, and various model comparisons.
  - classes.py: has test for different network configuration based on activation functions and optimizers.
 
**Figures**
The Figure folder contains all relevant plots with informative prefixes.

All required packages to run the code, are listed in requirements.txt

In the LLM folder, we have added links to our LLM chats (see LLM/links.md).
