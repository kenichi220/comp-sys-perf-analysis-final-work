#!/usr/bin/bash
python3 -m venv venv

. venv/bin/activate
which python3

python3 --version
pip --version

pip install --upgrade pip
pip install tensorflow
pip install tensorflow[and-gpu]

pip freeze > reqs.txt

pip install -r reqs.txt

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
