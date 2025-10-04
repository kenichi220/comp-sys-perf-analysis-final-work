#!/usr/bin/bash
python3 -m venv venv

. venv/bin/activate
which python3

python3 --version
pip --version

pip install --upgrade pip
pip install tensorflow==2.15.1
pip install tensorflow[and-gpu]==2.15.1
pip install tensorflow[and-cuda]==2.15.1

pip freeze > reqs.txt

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
