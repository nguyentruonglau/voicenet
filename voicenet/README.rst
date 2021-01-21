VoiceNet
--------

To setup, simply do::
	>>> pip install voicenet

To use, simply do::

    >>> from voicenet import VoiceNet
    >>> model = VoiceNet(input_shape=(40,80,1), classes=40)
    >>> model.summary()

Author::
	>>> Nguyen Truong Lau