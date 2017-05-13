# demo.py

# IMPORTS
import sys
sys.path.append('..')

from mic import *

# common
from common.core import *
from common.gfxutil import *
from common.audio import *
from common.writer import *

# other
import numpy as np
import time


# MAIN WIDGET
class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        # Set up audio input
        self.writer = AudioWriter('data') # for debugging audio output
        self.mic_audio = Audio(1, self.writer.add_audio, self.process_mic_input)
        self.language_label = ""

        # info text
        self.info = topleft_label()
        self.add_widget(self.info)

        # Set up microphone input handling
        self.mic_handler = MicrophoneHandler(1)

    def on_key_down(self, keycode, modifiers):
        # play / pause toggle
        if keycode[1] == 'p':
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()

    def process_mic_input(self, data, num_channels):
        self.language_label = self.mic_handler.add_data(data)

    def on_update(self) :
        if len(self.language_label) > 0:
            self.info.text = 'Language: %s' % self.language_label
        else:
            self.info.text = 'No language detected yet'
        self.mic_audio.on_update()


# LET'S RUN THIS CODE!
run(MainWidget)
