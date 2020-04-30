# mantrap
Minimal interferring Interactive Risk-aware Planning for multimodal and time-evolving obstacle behaviour

## Description
Planning safe human-robot interaction is a necessary towards the widespread integration of autonomous systems in the 
society. However, while instinctive to humans, socially compliant navigation is still difficult to quantify due to the 
stochasticity in people’s behaviors. Previous approaches have either strongly simplified the multimodal and time-varying
behaviour of humans, applied hardly tractable methods lacking safety guarantees or were simply not computationally 
feasible. Therefore the goal of this work to develop a risk-aware planning methodology with special regards on 
minimizing the interaction between human and robot and taking account the actual multi-modality and time-evolving nature
of the humans behaviour, based on the Trajectron model (Ivanovic 19).  

## Installation
For installation clone the repository including it's submodules: 

```
git clone --recurse-submodules --remote-submodules https://github.com/simon-schaefer/mantrap.git
```

Next create a virtual environment for Python 3 and install all package requirements by running 

```
source ops/setup.bash
```

Afterwards install the NLP-solver [IPOPT](https://coin-or.github.io/Ipopt/) and it's python wrapper which is called 
[cyipopt](https://pypi.org/project/ipopt/):

```
# Download 
# Install Ipopt NLP solver. 
cd external/Ipopt
chmod u+x coinbrew
brew install bash  # update bash version (>= 4.0)

mkdir build
./coinbrew fetch Ipopt
./coinbrew build Ipopt --prefix=/path/to/build --test
./coinbrew install Ipopt

# Set PKG_CONFIG_PATH environment variable to IPOPT build directory
export PKG_CONFIG_PATH="path/to/mantrap/external/IPOPT/build/Ipopt/master"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/path/to/mantrap/mantrap/external/IPOPT/build/ThirdParty/Mumps/2.0"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/path/to/mantrap/mantrap/external/IPOPT/build/ThirdParty/Metis/2.0"

# Install cyipopt following https://pypi.org/project/ipopt/
# Download binary files from https://pypi.org/project/ipopt/#files
# Then install by running 
cd external/cyipopt
python setup.py install
```

In order to ensure a working Trajectron model the branch `online_with_torch` has to be checkout.

## Documentation 
For code documentation the [Sphinx](https://www.sphinx-doc.org/en/master/) engine has been used. For building the 
documentation locally setup the project and run `make html` in the documentation folder. Then open the documentation 
by opening the `index.html` file in the resulting documentation build directory. 

## Running in optimized mode
Running python in optimized mode let's skip all `assert` statements in order to save runtime.

```
python3 -O evaluation.py
```