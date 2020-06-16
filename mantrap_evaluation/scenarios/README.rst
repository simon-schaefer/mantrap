Mantrap Evaluation Dataset API
------------------------------
For a broad set of testing the mantrap project provides an API for integration of custom datasets. 

Currently the `ETH Pedestrian datasets <https://icu.ee.ethz.ch/research/datsets.html>`_ is supported for automatic
evaluation, however other datasets can be added easily with the mantrap evaluation API. To continue with the 
ETH Pedestrian dataset, download it first by following these steps: 

.. code-block:: bash

    cd mantrap_evaluation/datasets/eth
    bash download.bash

In order to add other scenarios for evaluation you have to add a function with the following structure and add
it to the dictionary of scenarios in :code:`__init__.py`:

.. code-block:: python

    def foo(
        env_type: mantrap.environment.base.GraphBasedEnvironment.__class__,
    ) -> Tuple[mantrap.environment.base.GraphBasedEnvironment, torch.Tensor, Union[Dict[str, torch.Tensor], None]]:


The function basically defines the initial state of the ego (robot) as well the initial state and state histories
of the ados (pedestrian) in the scene, then calls the :code:`create_environment()` method defined in :code:`api.py` to
create an environment, which builds the first return argument. The second argument is the ego state (position) for
the ego robot, the third the ground truth positions for the ados in the scene, i.e. how they would have moved if 
there wouldn't be a robot in the scene. Being based on a perfect behaviour prediction model and grounded on a 
perfect optimization the ado trajectories conditioned on the robot trajectory should approach this ground-truth 
trajectories. 

An easy to understand example of the usage of the API can be found in :code:`custom/haruki.py`.