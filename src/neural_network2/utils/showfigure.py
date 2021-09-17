import numpy as np
import cv2
import matplotlib.pyplot as plt

num0 = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 0, 1],
     [1, 0, 1],
     [1, 1, 1]]
)
num1 = np.array(
    [[0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]
)
num2 = np.array(
    [[1, 1, 1],
     [0, 0, 1],
     [1, 1, 1],
     [1, 0, 0],
     [1, 1, 1]]
)
num3 = np.array(
    [[1, 1, 1],
     [0, 0, 1],
     [1, 1, 1],
     [0, 0, 1],
     [1, 1, 1]]
)
num4 = np.array(
    [[1, 0, 1],
     [1, 0, 1],
     [1, 1, 1],
     [0, 0, 1],
     [0, 0, 1]]
)
num5 = np.array(
    [[1, 1, 1],
     [1, 0, 0],
     [1, 1, 1],
     [0, 0, 1],
     [1, 1, 1]]
)
num6 = np.array(
    [[1, 1, 1],
     [1, 0, 0],
     [1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]
)
num7 = np.array(
    [[1, 1, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]
)
num8 = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]
)
num9 = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1],
     [0, 0, 1],
     [1, 1, 1]]
)
dot = np.array(
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [1, 0, 0]]
)

numbers = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9, dot]
margin = np.zeros(shape=(5, 1))


def showFigure(num):
    n1 = int(num / 10)  # ten
    n2 = int(num % 10)  # unit
    n3 = int(np.round((num - 10 * n1 - n2) * 10))

    img = np.hstack((numbers[n1], margin, numbers[n2], margin, numbers[-1], margin, numbers[n3]))
    img = np.pad(img, ((10, 10), (5, 5)))
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = (img > 0.5).astype(int)
    return img
