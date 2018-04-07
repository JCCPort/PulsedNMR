import os

import matplotlib.pyplot as plt
import numpy as np
# from tkinter import filedialog, Tk
from pandas import read_csv, read_hdf
from scipy import fftpack, interpolate
from scipy.optimize import curve_fit

# from converter import DataConvert


# plt.style.use('scione')
e = 2.7182818
plt.switch_backend('QT5Agg')


def find_nearest(array, value):
    idx = (np.abs(array - value)).idxmin()
    return array[idx]


def read():
    data = read_csv('C:\\Users\Josh\IdeaProjects\PulsedNMR\RDAT\HMO_pulse_echo_RN=20.csv', names=['t', 'm'], engine='c')
    return data


def read2():
    data2 = read_hdf('C:\\Users\Josh\IdeaProjects\PulsedNMR\RDAT\HMO_pulse_echo_RN=20.h5', 'table', names=['t', 'm'],
                     engine='c')
    return data2


'''
Free induction decay (FID). This is the signal of M_x or M_y decaying after a pi/2 pulse.
'''


def T2_from_echo(M_xy, M0, tau):
    """
    This function extracts the spin-spin relaxition time from the height difference in initial magnetization
    and magnetization in the xy-plane that after a time two tau has passed.
    :param M_xy: Magnetization in the xy-plane.
    :param M0: Initial magnetization in z direction.
    :param tau: Time between the pi/2 and the pi pulse.
    :return: Spin-spin relaxation time.
    """
    return -2 * tau / (np.log(M_xy / M0))


def echo_as_T2(t, M0, T2, c, ph):
    """

    :param t:
    :param M0: Initial magnetization in z direction.
    :param T2: Spin-spin relaxation time.
    :param c: Intercept to compensate for DC-offset.
    :param ph: Phase difference.
    :return: Magnetization in the xy-plane.
    """
    return M0 * (np.exp(-((t - ph) / T2))) + c


os.chdir('C:\\Users\\Josh\\IdeaProjects\\PulsedNMR\\RDAT')
#  TODO: Split this into different functions for the different experiments.
for filename in os.listdir(os.getcwd()):
    dat = read_csv(filename, names=['t', 'm'],
                   engine='c')
    if 'RFID' in filename:
        maxi = np.max(dat['m'])
        try:
            smoothdat = interpolate.UnivariateSpline(dat['t'], dat['m'], k=5, s=200)
            grad1 = np.gradient(smoothdat(dat['t']))
            grad1_2 = np.gradient(grad1)
            grad2 = interpolate.UnivariateSpline(dat['t'], grad1_2, k=3, s=0)
            s = []
            max_pos = dat['t'][int(np.median(np.where(dat['m'] == find_nearest(dat['m'], maxi))[0]))]
            for p in range(0, len(grad2.roots())):
                f = find_nearest(dat['t'], grad2.roots()[p])
                if f > max_pos:
                    s.append(f)
            b = np.where(dat['t'] == s[0])[0][0]
        except ValueError:
            b = int(np.median(np.where(dat['m'] == maxi)[0]))
        mini = np.min(dat['m'][b:])
        mx = np.max(dat['m'][b:]) - mini
        max_loc = int(np.median(np.where(dat['m'] == find_nearest(dat['m'], mx + mini))))
        max_loc_time = dat['t'][max_loc]
        decay_con_amp = mx / e
        decay_con_amp_pos = int(
                np.median(np.where(dat['m'] == find_nearest(dat['m'], decay_con_amp + mini))))
        decay_con_amp_time = dat['t'][decay_con_amp_pos]
        decay_time = decay_con_amp_time - max_loc_time
        initial = np.array([mx, decay_time, mini, max_loc_time])
        popt, pcov = curve_fit(echo_as_T2, xdata=dat['t'][b:], ydata=dat['m'][b:], p0=initial, maxfev=10000,
                               method='trf')
        plt.title('{}'.format(filename))
        plt.plot(dat['t'], dat['m'], '+', ms=1.4, color='r')
        plt.plot(dat['t'][b:], echo_as_T2(dat['t'][b:], *popt), ls='--', lw=2, color='k')
        plt.axhline(mx + mini)
        plt.axhline(decay_con_amp + mini)
        plt.axvline(max_loc_time)
        plt.axvline(decay_con_amp_time)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
    elif 'RN' in filename:
        len2 = 2 * len(dat['m'])
        xs = np.linspace(np.min(dat['t']), np.max(dat['t']), len2)
        f = interpolate.Rbf(dat['t'], dat['m'], smooth=3, function='gaussian', epsilon=np.mean(np.diff(xs)) * 3)
        ys = f(xs)
        plt.plot(xs, -ys)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
        sample_rate = round(1 / np.mean(np.diff(dat['t'])), 11)
        length = len(xs)
        fo = fftpack.fft(-ys)
        freq2 = fftpack.fftfreq(length, 1 / sample_rate)
        halfln = int(length / 2)
        plt.title('{}'.format(filename))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.plot(dat['t'], dat['m'])
        plt.show()
        plt.title('{} Fourier Transformed'.format(filename))
        plt.plot(freq2[1:halfln], abs(fo[1:halfln]))
        plt.xlim(0, 2000)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
    elif 'DAT' in filename:
        sample_rate = round(1 / np.mean(np.diff(dat['t'])), 11)
        length = len(dat['t'])
        fo = fftpack.fft(dat['m'])
        freq4 = [1e6 * x * sample_rate / length for x in np.array(range(0, length))]
        halfln = int(length / 2)
        plt.title('{}'.format(filename))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.plot(dat['t'], dat['m'])
        plt.show()
        plt.title('{} Fourier Transformed'.format(filename))
        plt.plot(freq4[1:halfln], abs(fo[1:halfln]))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
