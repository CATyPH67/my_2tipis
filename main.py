import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from scipy.signal import butter, filtfilt


def create_harmonic(length, points_count, frequency):
    x = np.arange(0, length, length/points_count)
    y = np.sin(np.pi * 2 * frequency * x)

    return x, y


def create_digital(length, points_count, frequency):
    x, y = create_harmonic(length, points_count, frequency)

    for i in range((len(y))):
        if y[i] > 0:
            y[i] = 1
        else:
            y[i] = 0

    return x, y


def create_spectrum(y):
    yf = rfft(y)
    yf = yf[1: len(yf) // 2]
    return np.abs(yf)


def create_amplitude_modulation(length, points_count, frequency):
    x, yH = create_harmonic(length, points_count, frequency)
    x, yD = create_digital(length, points_count, 2)
    yA = []
    for i in range(len(x)):
        yA.append(yH[i] * yD[i])
    return x, yA


def create_frequency_modulation(length, points_count, frequency1, frequency2):
    x, yH1 = create_harmonic(length, points_count, frequency1)
    x, yH2 = create_harmonic(length, points_count, frequency2)
    x, yD = create_digital(length, points_count, 2)
    yF = []
    for i in range(len(x)):
        if yD[i] == 1:
            yF.append(yH1[i])
        else:
            yF.append(yH2[i])
    return x, yF


def create_phase_modulation(length, points_count, frequency):
    x, yH = create_harmonic(length, points_count, frequency)
    x, yD = create_digital(length, points_count, 2)
    yP = []
    for i in range(len(x)):
        if yD[i] == 1:
            yP.append(yH[i])
        else:
            yP.append(-yH[i])
    return x, yP


def synthesis_signal(spector):
    sintez = []
    for i in range(len(spector)):
        if np.abs(spector[i]) < 100:
            sintez.append(0)
        else:
            sintez.append(np.abs(spector[i]))

    y_synthesis = np.fft.ifft(np.array(sintez))
    return y_synthesis


def filter_signal(y):
    b, a = butter(5, 0.1)
    filtered_data = filtfilt(b, a, np.abs(y))

    a_y = list()
    for i in range(len(filtered_data)):
        if filtered_data[i] > 1:
            a_y.append(1)
        else:
            a_y.append(0)
    plt.plot(a_y)
    # plt.show()
    return a_y


if __name__ == '__main__':
    xA, yA = create_amplitude_modulation(1, 1000, 16)
    plt.subplot(1, 2, 1)
    plt.title("Амплитудная модуляция")
    plt.plot(xA, yA)

    yAS = create_spectrum(yA)
    plt.subplot(1, 2, 2)
    plt.title("Спектр амплитудной модуляция")
    plt.plot(yAS[1: 40])

    plt.show()

    xF, yF = create_frequency_modulation(1, 1000, 16, 8)
    plt.subplot(1, 2, 1)
    plt.title("Частотная модуляции")
    plt.plot(xF, yF)

    yFS = create_spectrum(yF)
    plt.subplot(1, 2, 2)
    plt.title("Спектр частотной модуляции")
    plt.plot(yFS[1: 40])

    plt.show()

    xP, yP = create_phase_modulation(1, 1000, 16)
    plt.subplot(1, 2, 1)
    plt.title("Фазовая модуляция")
    plt.plot(xP, yP)

    yPS = create_spectrum(yP)
    plt.subplot(1, 2, 2)
    plt.title("Спектр фазовой модуляции")
    plt.plot(yPS[1: 40])

    plt.show()

    y_synthesis = synthesis_signal(yAS)
    plt.subplot(1, 2, 1)
    plt.title("Синтезированный сигнал")
    plt.plot(y_synthesis)

    y_filtered = filter_signal(y_synthesis)
    plt.subplot(1, 2, 2)
    plt.title("Отфильтрованный сигнал")
    plt.plot(y_filtered)

    plt.show()
