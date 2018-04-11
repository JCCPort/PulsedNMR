from tkinter import filedialog, Tk
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lmfit.models import GaussianModel, LinearModel
from pandas import read_csv, read_hdf
from scipy import fftpack, interpolate
from scipy.optimize import curve_fit

from range_selector import RangeTool

sns.set_style("whitegrid")
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
# from converter import DataConvert


# plt.style.use('scione')
e = 2.7182818


# plt.switch_backend('QT5Agg')

def find_nearest(array, value):
    idx = (np.abs(array - value)).idxmin()
    return array[idx]


def pick_dat(initdir='RDAT', title="Select file", cols=['t', 'm']):
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\Josh\IdeaProjects\PulsedNMR\{}".format(initdir),
                                               title=title)
    filename_parts = root.filename.split('/')[-1]
    if 'csv' in root.filename:
        data = read_csv(root.filename, names=cols, engine='c')
        return data, filename_parts
    elif 'h5' in root.filename:
        data = read_hdf(root.filename, 'table', names=cols, engine='c')
        return data, filename_parts
    else:
        print('Unexpected file type. Choose either a .csv or .h5 file.')


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
    # Old form: return M0 * (np.exp(-((t - ph) / T2))) + c
    return M0 * (np.exp(-(t / T2) + ph)) + c



def FID_Exponential_fit():
    dat, filename = pick_dat()
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
    popt, pcov = curve_fit(echo_as_T2, xdata=dat['t'][b:], ydata=dat['m'][b:], p0=initial, maxfev=30000,
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


def range_to_list():
    dat1, filename1 = pick_dat("RDAT_Test", "Select dataset to draw from")
    dat2 = read_csv("C:\\Users\\Josh\\IdeaProjects\\PulsedNMR\\Ranges\\{}".format(filename1),
                    names=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
    xrange = []
    yrange = []
    xranges = {}
    yranges = {}
    for o in range(0, len(dat2)):
        xrange.append((dat1['t'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
        yrange.append((dat1['m'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
    for o in range(0, len(xrange)):
        xranges[o] = xrange[o]
        yranges[o] = yrange[o]
    return xranges, yranges, xrange, yrange


# TODO: Compare this method with simply taking the maximum point of the peaks
def echo_fits():
    xrs, yrs, xr, yr = range_to_list()
    cents: List[float] = []
    cents_uncert: List[float] = []
    heights: List[float] = []
    heights_uncert: List[float] = []
    fig, ax = plt.subplots()
    for i in range(0, len(xrs)):
        mdl = GaussianModel(prefix='G_')
        lne = LinearModel(prefix='L_')
        params = mdl.guess(yrs[i], x=xrs[i])
        params += lne.guess(yrs[i], x=xrs[i])
        model = mdl + lne
        result = model.fit(yrs[i], params, x=xrs[i], method='leastsq')
        plt.plot(xrs[i], result.best_fit)
        plt.plot(xrs[i], yrs[i], 'x', ms=0.6, color='k')
        cent: float = result.params['G_center'].value
        amp: float = result.params['G_height'].value
        inter: float = result.params['L_intercept'].value
        grad: float = result.params['L_slope'].value
        height: float = amp + ((cent * grad) + inter)
        heights.append(height)
        cents.append(cent)
        cents_uncert.append(result.params['G_center'].stderr)
        partial_amp = 1
        partial_grad = cent
        partial_x = grad
        partial_inter = 1
        amp_term = partial_amp * result.params['G_height'].stderr
        grad_term = partial_grad * result.params['L_slope'].stderr
        x_term = partial_x * np.mean(np.diff(xrs[i]))
        inter_term = partial_inter * result.params['L_intercept'].stderr
        height_uncert = np.sqrt(amp_term ** 2 + grad_term ** 2 + x_term ** 2 + inter_term ** 2)
        heights_uncert.append(height_uncert)
    maxy = np.max(heights)
    miny = np.min(heights)
    bounds = [[maxy * 0.95, 0.01, 0, 0], [maxy * 2, 0.05, miny * 1, cents[0] * 0.0001]]
    initial = np.array([maxy * 1.3, 0.015, miny, cents[0] * 0.00005])
    popt, pcov = curve_fit(echo_as_T2, xdata=cents, ydata=heights, bounds=bounds, p0=initial, sigma=heights_uncert,
                           maxfev=30000,
                           method='dogbox')
    vals = np.linspace(0, np.max(cents), 1000)
    plt.plot(vals, echo_as_T2(vals, *popt), ls='-.', color='k', lw=1)
    plt.plot(cents, heights, 'x', ms=4, color='k')
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetization (A/m)")
    ax.grid(color="k", linestyle='--', alpha=0.4)
    plt.axhline(popt[0], color='k', ls='--', alpha=0.7, lw=1, zorder=1)
    plt.axhline(popt[0] / e, color='k', ls='--', alpha=0.7, lw=1, zorder=1)
    plt.text(0.9, 0.9, "T_1: {:.4f} s".format(popt[1]), horizontalalignment='center',
             verticalalignment="center",
             transform=ax.transAxes,
             bbox={'pad': 8, 'fc': 'w'}, fontsize=16)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def interped_fourier_transformer():
    dat, filename = pick_dat('RDAT')
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


def fourier_transformer():
    dat, filename = pick_dat('RDAT', 'Select data to be Fourier Transformed')
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
    fig, ax = plt.subplots()
    plt.title('{} Fourier Transformed'.format(filename))
    figure, = ax.plot(freq4[1:halfln], abs(fo[1:halfln]))
    Sel = RangeTool(freq4[1:halfln], abs(fo[1:halfln]), figure, ax, 'thing')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def fourier_curvefit():
    dat, filename = pick_dat('RDAT')
    sample_rate = round(1 / np.mean(np.diff(dat['t'])), 11)
    length = len(dat['t'])
    fo = fftpack.fft(dat['m'])
    freq4 = [1e6 * x * sample_rate / length for x in np.array(range(0, length))]
    halfln = int(length / 2)


def pick_ranges():
    dat, filename = pick_dat('RDAT', 'Select file to pick ranges in')
    fig, ax = plt.subplots()
    plt.title('{} Fourier Transformed'.format(filename))
    figure, = ax.plot(dat['t'], dat['m'])
    Sel = RangeTool(dat['t'], dat['m'], figure, ax, filename)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


# pick_ranges()
echo_fits()
