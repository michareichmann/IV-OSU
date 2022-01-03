#!/usr/bin/env python
# --------------------------------------------------------
#       small scripts to plot iv data from Harris
# created on May 28th 2019 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------

from glob import glob
from numpy import genfromtxt, loadtxt, abs, min, max, log10
from plotting.save import *
from re import split as txtsplit
import h5py


class IV(object):

    def __init__(self, dut_name='II6-B2'):
        self.Draw = SaveDraw()
        self.DUTName = dut_name
        self.DataDir = join(BaseDir, 'data', dut_name)
        self.DataFile = join(self.DataDir, f'{dut_name}.hdf5')

        self.convert_data()
        self.Data = h5py.File(self.DataFile, 'r')
        self.Names = list(self.Data.keys())
        self.Units, self.Factors = array([self.find_unit(i) for i in range(len(self.Names))], object).T

    # -----------------------------
    # region CONVERT
    def convert_data(self):
        new_file = self.DataFile
        if not isfile(new_file):
            f = h5py.File(new_file, 'w')
            t = info('converting .iv data to hdf5 ... ', endl=False)
            for file_name in glob(join(self.DataDir, '*.iv')):
                g = f.create_group(basename(file_name).strip('.iv'))
                t_start = datetime.strptime(str(loadtxt(file_name, skiprows=3, max_rows=1, comments='sdf', dtype=str, delimiter='d')), '# Start Date, Time: %m/%d/%Y, %I:%M:%S %p')
                v = genfromtxt(file_name, comments='none', skip_header=5, max_rows=1)[3:]
                data = genfromtxt(file_name, usecols=arange(v.size) + 1).T * 1e15   # to fA
                g.create_dataset('current', data=data)
                g.create_dataset('voltage', data=v)
                g.create_dataset('t_start', data=array([time_stamp(t_start)]))
            add_to_info(t)
    # endregion CONVERT
    # -----------------------------

    # -----------------------------
    # region GET
    def get_title(self, i=0):
        return basename(self.Names[i]).replace('_', ' ').replace('-', ' ')

    def get_short_names(self):
        names = [name.replace('BIG_SIDE', 'BS').replace('SMALL_SIDE', 'SS').replace(self.DUTName, '', ).strip('_') for name in self.Names]
        return [' '.join(array(txtsplit('[-_]', n))[[0, 2]]) for n in names]

    def find_unit(self, i=0):
        x = sign(self.get_voltage(i)) * self.get_mean_current(i, 50, raw=True)
        low = min(x[x > 0])
        i = next((i for i, j in enumerate(10 ** arange(1.5, 8, 3)) if low < j), 2)
        return ['[fA]', '[pA]', '[nA]'][i], [1, 1e-3, 1e-6][i]

    def get_t0(self, i=0):
        return self.Data[self.Names[i]]['t_start'][0]

    def get_current(self, i=0):
        return array(self.Data[self.Names[i]]['current'])

    def get_diff(self, i=0, n_last=50):
        x, y = self.get_voltage(i), self.get_mean_current(i, n_last)
        return x[x > 0], abs(y[x > 0] + y[x < 0])
    
    def get_mean_diff(self, i=0, n_last=50):
        return mean_sigma(self.get_diff(i, n_last)[1])

    def get_mean_current(self, i=0, n_last=0, raw=False, log_modulus=False):
        factor = 1 if raw else self.find_unit(i)[1]
        current = mean(self.get_current(i)[:, -n_last:], axis=1) * factor
        return sign(current) * log10(1 + abs(current)) if log_modulus else current

    def get_mean_time(self, i=0):
        m, n = self.Data[self.Names[i]]['current'].shape
        return arange(m) * n + n / 2 + self.get_t0()

    def get_voltage(self, i=0):
        return array(self.Data[self.Names[i]]['voltage'])

    @staticmethod
    def t_args(rel_t=True, t0=None, log=True):
        return {'x_tit': 'Time [hh:mm]', 'y_tit': f'{"Log " if log else ""}Current [fA]', 't_ax_off': t0 if rel_t else 0}

    def v_args(self, i):
        return {'x_tit': 'Voltage [V]', 'y_tit': f'Current {self.Units[i]}'}
    # endregion GET
    # -----------------------------

    # -----------------------------
    # region DRAW
    def draw_expo(self, i=0, n=1, fit_start=50):
        y = self.get_current(i)[n] * self.Factors[i]
        g = self.Draw.graph(arange(y.size), y, x_tit='Time [s]', y_tit='Current {}'.format(self.Units[i]))
        if fit_start:
            g.Fit(self.Draw.make_f('fit', 'expo(0) + [2]'), 'qs', '', fit_start, y.size)

    def draw_expo_asym(self, i=0, n=1):
        y = self.get_current(i)[n]
        g = self.Draw.make_tgrapherrors(arange(y.size), y)
        fit = self.Draw.make_f('fit', 'expo(0) + [2]')
        f = self.Draw.make_tf1('a', lambda x: g.Fit(fit, 'qs', '', x, y.size).Parameter(2), 100, y.size - 100)
        self.Draw(f)

    def draw_time(self, i=0, rel_t=False, log_modulus=True, **dkw):
        """ plot raw data vs. time with option to use log modolus transformation for the current. """
        y, t0 = concatenate(self.get_current(i)), self.get_t0(i)
        y = sign(y) * log10(1 + abs(y)) if log_modulus else y
        return self.Draw.graph(arange(y.size) + t0, y, **prep_kw(dkw, title=self.get_title(i), **IV.t_args(rel_t, t0, log_modulus), markersize=.3, **Draw.mode(2)))

    def draw_mean_time(self, i=0, n_last=50, log_modulus=False, rel_t=False, **dkw):
        """ plot averaged time of the [n_last] data points vs. time with option to use log modolus transformation for the current. """
        y = self.get_mean_current(i, n_last, log_modulus=log_modulus)
        return self.Draw.graph(self.get_mean_time(i), y, **prep_kw(dkw, title=self.get_title(i), **IV.t_args(rel_t, self.get_t0(i), log_modulus), markersize=.5, **Draw.mode(2)))

    def draw_mean_times(self, n_last=50, **dkw):
        """ plot mean times for all data files. """
        graphs = [Draw.make_tgrapherrors(self.get_mean_time(i), self.get_mean_current(i, n_last, raw=True, log_modulus=True)) for i in range(len(self.Names))]
        mg = self.Draw.multigraph(graphs, f'Currents {self.DUTName}', self.get_short_names(), **prep_kw(dkw, **IV.t_args(True, 0, True), markersize=.7, **Draw.mode(2), color=True, grid=True))
        format_histo(mg, t_ax_off=self.get_t0())

    def draw(self, i=0, n_last=50, **dkw):
        """ plot mean currents of the last [n_last] data points vs voltage. """
        x, y = self.get_voltage(i), self.get_mean_current(i, n_last)
        y = sign(x) * y
        self.Draw.graph(x, y, **prep_kw(dkw, **self.v_args(i)), markersize=.7, grid=True, center_x=True, center_y=True, y_range=[min(y[y > 0]) / 2, max(abs(y)) * 2], logy=True,
                        title=self.get_title(i), **Draw.mode(2))

    def draw_same(self, i=0, n_last=50):
        """ plot mean currents of the last [n_last] data points vs voltage. flips negative to pos voltage """
        x, y = self.get_voltage(i), self.get_mean_current(i, n_last)
        data = array([[abs(x[cut]), sign(x[cut]) * y[cut]] for cut in [x < 0, x > 0]])
        graphs = [self.Draw.make_tgrapherrors(*idata, color=col) for idata, col in zip(data, [2, 1])]
        mg = self.Draw.multigraph(graphs, f'IV for {self.get_title(i)}', ['Negative Bias', 'Positive Bias'], markersize=.8, grid=True, w=2, lm=.08, bm=.2, logy=True, leg_left=True)
        v = concatenate(data[:, 1])
        format_histo(mg, tit_size=.05, lab_size=.045, y_range=[min(v[v > 0]) / 2, max(v) * 2], y_off=.8, y_tit='Current {}'.format(self.Units[i]), x_tit='Voltage [V]', center_x=True,
                     center_y=True)

    def draw_diff(self, i=0, n_last=50):
        x, y = self.get_diff(i, n_last)
        n = x.argmax() + 1
        graphs = [Draw.make_tgrapherrors(x[:n], y[:n], color=1), Draw.make_tgrapherrors(x[n:], y[n:], color=2)]
        self.Draw.multigraph(graphs, 'Current Difference', ['up', 'down'], x_tit='Voltage [V]', y_tit='Current Difference {}'.format(self.Units[i]), leg_left=True, w=2)
    # ---------------self.get_diff(i, n_last)--------------
    # endregion DRAW


def time_stamp(dt, off=None):
    t = float(dt.strftime('%s'))
    return t if off is None else t - (off if off > 1 else dt.utcoffset().seconds)


if __name__ == '__main__':

    from argparse import ArgumentParser
    main_parser = ArgumentParser()
    main_parser.add_argument('dut_name', nargs='?', default='II6-B2')
    args = main_parser.parse_args()

    z = IV(args.dut_name)
    save_last = z.Draw.save_last
