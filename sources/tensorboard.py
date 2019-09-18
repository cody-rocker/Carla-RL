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
        from tensorflow.python.util import deprecation
        from keras.callbacks import Callback

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Own Tensorboard class giving ability to use single writer across multiple .fit() calls
# Allows us also to easily log additional data
# Dramatically decreases amount of data being saved into Tensorboard logs and write time (as appends to one file)
class TensorBoard(Callback):

    # Set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, log_dir):
        self.step = 1
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Saves logs with our step number (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(self.step, **logs)

    # Custom method for saving own (and also internal) metrics (can be called externally)
    def update_stats(self, step, **stats):
        self._write_logs(stats, step)

    # More or less the same writer as in Keras' Tensorboard callback
    # Physically writes to the log files
    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()
