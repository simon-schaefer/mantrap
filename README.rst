mantrap
=======

Minimal interferring Interactive Risk-aware Planning for multimodal and time-evolving obstacle behaviour

## Description
Planning safe human-robot interaction is a necessary towards the widespread integration of autonomous systems in the 
society. However, while instinctive to humans, socially compliant navigation is still difficult to quantify due to the 
stochasticity in people’s behaviors. Previous approaches have either strongly simplified the multimodal and time-varying
behaviour of humans, applied hardly tractable methods lacking safety guarantees or were simply not computationally 
feasible. Therefore the goal of this work to develop a risk-aware planning methodology with special regards on 
minimizing the interaction between human and robot and taking account the actual multi-modality and time-evolving nature
of the humans behaviour, based on the Trajectron model (Ivanovic 19).  

Installation
------------
For installation clone the repository including it's submodules: 

.. code-block:: bash

   git clone --recurse-submodules --remote-submodules https://github.com/simon-schaefer/mantrap.git

Next create a virtual environment for Python 3 and install all package requirements by running 

.. code-block:: bash

   source ops/setup.bash

Afterwards install the NLP-solver `IPOPT <https://coin-or.github.io/Ipopt/>`_ and it's python wrapper which is called
`cyipopt <https://pypi.org/project/ipopt/>`_:

.. code-block:: bash

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


In order to ensure a working Trajectron model the branch :code:`online_with_torch` has to be checkout.

Evaluation
----------
The evaluation of mantrap is grounded on real-world pedestrian behaviour datasets. While the  
`ETH Pedestrian datasets <https://icu.ee.ethz.ch/research/datsets.html>`_ and some custom scenarios already have
been integrated, other datasets can be easily added using the mantrap_evaluation dataset API; for more information
regarding this please read :code:`mantrap_evaluation/datasets/README`.

Documentation
-------------
For code documentation the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ engine has been used. For building the
documentation locally setup the project and run :code:`make html` in the documentation folder. Then open the
documentation by opening the :code:`index.html` file in the resulting documentation build directory.

Running in optimized mode
-------------------------
Running python in optimized mode let's skip all :code:`assert` statements and sets the logging level to warning
in order to save runtime.

.. code-block:: bash

   python3 -O evaluation.py
