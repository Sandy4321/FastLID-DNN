#mic.py

import numpy as np
import time

import sys
sys.path.append('..')

from common.audio import *

# Used to handle streaming audio input data
# NOTE: currently only accepts mono audio (i.e. num_channels = 1)
class MicrophoneHandler(object) :
    def __init__(self, num_channels):
        super(MicrophoneHandler, self).__init__()

        assert(num_channels == 1)

        # Set up audio buffer
        self.buf_size = int(0.100 * kSampleRate)    # 100 ms
        self.buf = np.zeros(self.buf_size, dtype=np.int8)
        self.buf_idx = 0

    # Receive data and send back a string indicating the language detected, if ready
    # Returns empty string if not ready to classify language
    def add_data(self, data):
        event = ""

        '''
        # Check if we will need to classify an event
        buffer_full = (self.buf_size - self.buf_idx) < len(data)
        if buffer_full:
            # Fill as much as we can, then reset index
            self.buf[self.buf_idx:] = data[:self.buf_size - self.buf_idx]
            self.buf_idx = 0

            # Get features (in the format our classifier expects)
            feature_vec = self.feature_manager.compute_features(self.buf)

            # Update our model as we go, if the classifier supports it
            if self.classifier.supports_partial_fit():
                self.classifier.partial_fit(feature_vec, label)

            # Classify the event now that we have a full buffer
            event = self._classify_event(feature_vec)

            # Clear buffer out
            self.buf[:] = 0

            # Wait until we are told again to start processing audio
            self.processing_audio = False
        else:
            # Fill 'er up!
            self.buf[self.buf_idx:self.buf_idx + len(data)] = data
            self.buf_idx += len(data)
        '''

        return event
