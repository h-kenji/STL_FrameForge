import numpy as np

#The frame analysis can vary depending the use case. This case will only indicate a black frame as a bad frame, but you can change or add another bad frame classification within this function.


def analyze_frame(frame):
    return np.all(frame == 0)
