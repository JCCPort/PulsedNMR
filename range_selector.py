from math import isnan
from os import chdir

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


class RangeTool:
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, x, y, figure2, ax, key):
        self.ax = ax
        self.key = key
        self.figure2 = figure2
        self.lx = ax.axhline(color='k', ls='--', linewidth=1, zorder=1, alpha=0.8)  # the horiz line
        self.ly = ax.axvline(color='k', ls='--', linewidth=1, zorder=2, alpha=0.8)  # the vert line
        self.lowers = np.array([])
        self.uppers = np.array([])
        self.rects = []
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts "
        self.x = x
        self.y = y
        self.lenx = len(self.x)
        miny = np.min(self.y)
        maxy = np.max(self.y)
        self.ax.set_xlim(min(self.x), max(self.x))
        height = maxy - miny
        self.ax.set_ylim(miny - 0.1 * height, maxy + 0.1 * height)
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
        self.cid1 = figure2.figure.canvas.mpl_connect('key_press_event', self.rangeselect)
        self.cid2 = figure2.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.cid3 = figure2.figure.canvas.mpl_connect('key_press_event', self.rangeremove)
        self.cid4 = figure2.figure.canvas.mpl_connect('key_press_event', self.finishplot)
        # self.cid5 = figure2.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.Ranges = DataFrame(columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex', 'Displayed'])
        self.il = 0
        self.iu = 0
        self.t = 0

    def __call__(self, event):
        if event.inaxes != self.figure2.axes:
            return

    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        indx = min(np.searchsorted(self.x, [x])[0], self.lenx - 1)
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.figure2.figure.canvas.draw_idle()
        # print('x=%1.2f, y=%1.2f' % (x, y))

    # TODO: Merge the bounds, indices and mpl object into a single Pandas DataFrame that's row's can be deleted.
    # TODO: Add an object onto plots that lists the range selections?
    def rangeselect(self, event):
        x = event.xdata
        indx = min(np.searchsorted(self.x, [x])[0], self.lenx - 1)
        x = self.x[indx]
        if event.key == 'tab':
            self.Ranges.at[self.il, 'Lower Bound'] = x
            self.Ranges.at[self.il, 'LowerIndex'] = indx
            self.il += 1
        if event.key == 'shift':
            if self.il > self.iu:
                self.Ranges.at[self.iu, 'Upper Bound'] = x
                self.Ranges.at[self.iu, 'UpperIndex'] = indx
                self.iu += 1
            elif self.il <= self.iu:
                print('Upper range selected before lower range.')
        if self.il == self.iu and self.il > 0:
            try:
                if isnan(self.Ranges.at[self.il - 1, 'Displayed']):
                    self.rects.append(self.ax.axvspan(self.Ranges.at[self.il - 1, 'Lower Bound'], self.Ranges.at[
                        self.iu - 1, 'Upper Bound'], alpha=0.3, edgecolor='k', linestyle='--', lw=2))
                    self.Ranges.at[self.il - 1, 'Displayed'] = 1
            except ValueError:
                print('ValueError in creating visual range selection.')
                pass
        self.cid3 = self.figure2.figure.canvas.mpl_connect('key_press_event', self.rangeremove)

    # TODO: Get this running so that the aforementioned DataFrame rows can be removed with a click.
    # def onpress(self, event):
    #     for entry in self.rects:
    #             print(entry)
    #             print(self.rects)
    #             contains, attrd = entry.contains(event)
    #             if contains:
    #                 entry.remove()
    #                 self.il -= 1
    #                 self.iu -= 1
    #                 self.ax.relim()
    #                 print('clicked')

    def rangeremove(self, event):
        if event.key == 'delete' and self.il == self.iu and self.il > 0:
            if not self.Ranges.empty:
                self.figure2.figure.canvas.mpl_disconnect(self.cid1)
                try:
                    self.Ranges.at[self.il - 1, 'Displayed'] = float('NaN')
                    self.il -= 1
                    self.iu -= 1
                    self.Ranges.drop(self.Ranges.index[-1], inplace=True)
                    Polys = self.ax.get_children()
                    Polys[len(self.Ranges.index)].remove()
                except IndexError:
                    self.Ranges.at[self.il - 1, 'Displayed'] = float('NaN')
                    self.il -= 1
                    self.iu -= 1
                    self.Ranges.drop(self.Ranges.index[0], inplace=True)
                    Polys = self.ax.get_children()
                    Polys[0].remove()
                    if self.Ranges == 'Empty DataFrame':
                        print('Range list is empty')
                finally:
                    pass
                self.cid1 = self.figure2.figure.canvas.mpl_connect('key_press_event', self.rangeselect)

    def finishplot(self, event):
        self.Ranges.astype('float32')
        if event.key == 'enter':
            chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\Ranges')
            # Check this isn't writing headers.
            self.Ranges.to_csv('{}'.format(self.key), index=False, encoding='utf-8', columns=['Lower Bound',
                                                                                                  'LowerIndex',
                                                                                                  'Upper Bound',
                                                                                                  'UpperIndex'],
                               header=None)
            plt.close()
            chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR')
        elif event.key == 'escape':
            plt.close()
