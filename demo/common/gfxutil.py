#####################################################################
#
# gfxutil.py
#
# Copyright (c) 2015, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

from kivy.uix.label import Label
from kivy.core.window import Window


# return a Label object configured to look good and be positioned at
# the top-left of the screen
def topleft_label() :
    l = Label(text = "text", valign='top', font_size='50sp',
              pos=(Window.width * 0.5, Window.height * 0.4),
              text_size=(Window.width, Window.height))
    return l

