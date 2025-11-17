#!/usr/bin/bash
python3 -m venv venv

. venv/bin/activate
which python

python3 --version
pip --version

pip install --upgrade pip
pip install tensorflow==2.15.1
pip install tensorflow[and-gpu]==2.15.1
pip install tensorflow[and-cuda]==2.15.1
pip install pandas # By default already installed, but need for poti for an unknow reason

pip freeze > reqs.txt

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
cat reqs.txt
