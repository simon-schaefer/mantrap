# /bin/bash

# Install visualization package.
cp $PYTHON_INSTALL_DIR/setup_visualization.py $PYTHON_TARGET_DIR/setup.py
cd $PYTHON_PACKAGE_DIR
python $PYTHON_TARGET_DIR/setup.py develop
rm $PYTHON_TARGET_DIR/setup.py

# Make all python scripts in library executable.
chmod +x $PROJECT_PATH/ros/nodes/*.py

# Go back to work directory.
cd $WORKDIR
