# TangibleMIDI

## TLDR
**TangibleMIDI** uses hand landmarks, captured by Mediapipe, to dynamically control audio data. Breaking the constraint from physical musical instruments.

## Operation
Video demo can be access on [Youtube](https://youtu.be/5DegdKCca4c). There are three controllable musical features: volume, pitch, echo, controlled by thumb, index finger, and middle finger. Refer to the table below for specifics.

Before running the `TangibleMIDI.py` file, run the following:
```
python -m venv TangibleMIDI
source TangibleMIDI/bin/activate
brew install portaudio
pip install -r requirement.txt
python TangibleMIDI.py
```

| Ground | Volume | Pitch | Echo |
|:-------:|:-------:|:-------:|:-------:|
| ![Ground](misc/Ground.png) | ![Volume](misc/Volume.png) | ![Pitch](misc/Pitch.png) | ![Echo](misc/Echo.png) |
| Thumb touch palm | Index finger touch palm | Middle finger touch palm | Pinch thumb and index finger |
| Rainbow Color | Green | Red | Blue | 