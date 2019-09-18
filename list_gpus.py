import os
import sys
from contextlib import redirect_stderr, redirect_stdout

# Suppress excessive console output and then load Tensorflow and Keras
# NOTE: This will potentially swallow important or useful information about
#       problems with your tensorflow/keras installation, but it works.
#       (Tested on Linux)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
with open(os.devnull, "w") as null:
    with redirect_stderr(null), redirect_stdout(null):
        import tensorflow as tf
        from tensorflow.python.client import device_lib
        from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Get list of all devices
devices = device_lib.list_local_devices()

# Print GPUs only
print('\n\n\nList of available CUDNN GPUs:')
for device in devices:
    if device.device_type == 'GPU':
        print(device.physical_device_desc)
