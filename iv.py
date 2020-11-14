#!/usr/bin/env python
# --------------------------------------------------------
#       small scripts to plot iv data from Harris
# created on May 28th 2019 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------

from draw import Draw, join, format_histo
from utils import *
from numpy import array, mean, arange, genfromtxt, loadtxt, sign, abs, min, max
import h5py
from os.path import basename


class IV(object):

    Margins = [.08, .12, .1, .08]

    def __init__(self, filename='II6-B5_BIG_SIDE_180330-NoSource.iv'):
        self.Dir = dirname(realpath(__file__))
        self.Draw = Draw(join(self.Dir, 'main.ini'))
        self.FileName = join(self.Dir, 'data', basename(filename).split('.')[0])
        self.Tit = basename(self.FileName).replace('_', ' ').replace('-', ' ')

        self.convert_data()
        self.Data = h5py.File('{}.hdf5'.format(self.FileName), 'r')
        self.Unit, self.F = self.find_unit()

    def find_unit(self):
        x = sign(self.get_voltage()) * self.get_mean_current(20, raw=True)
        low = min(x[x > 0])
        i = next((i for i, j in enumerate(10 ** arange(1.5, 8, 3)) if low < j), 2)
        return ['fA', 'pA', 'nA'][i], [1, 1e-3, 1e-6][i]

    def get_t0(self):
        return self.Data['t_start'][0]

    def get_current(self):
        return array(self.Data['current'])

    def get_mean_current(self, n_last=0, raw=False):
        return mean(self.get_current()[:, -n_last:], axis=1) * (1 if raw else self.F)
    
    def get_mean_time(self):
        m, n = self.Data['current'].shape
        return arange(m) * n + n / 2 + self.get_t0()

    def get_voltage(self):
        return array(self.Data['voltage'])

    def draw_expo_asym(self, n=1):
        y = self.get_current()[n]
        g = self.Draw.make_tgrapherrors(arange(y.size), y)
        fit = self.Draw.make_f('fit', 'expo(0) + [2]')
        f = self.Draw.make_tf1('a', lambda x: g.Fit(fit, 'qs', '', x, y.size).Parameter(2), 100, y.size - 100)
        self.Draw(f)

    def convert_data(self):
        new_file = '{}.hdf5'.format(self.FileName)
        if not file_exists(new_file):
            file_name = '{}.iv'.format(self.FileName)
            t_start = datetime.strptime(str(loadtxt(file_name, skiprows=3, max_rows=1, comments='sdf', dtype=str, delimiter='d')), '# Start Date, Time: %m/%d/%Y, %I:%M:%S %p')
            v = genfromtxt(file_name, comments='none', skip_header=5, max_rows=1)[3:]
            data = genfromtxt(file_name, usecols=arange(v.size) + 1).T * 1e15
            f = h5py.File(new_file, 'w')
            f.create_dataset('current', data=data)
            f.create_dataset('voltage', data=v)
            f.create_dataset('t_start', data=array([time_stamp(t_start)]))

    def draw_time(self, rel_t=False, log_modulus=True):
        y, t0 = concatenate(self.get_current()), self.get_t0()
        y = sign(y) * log10(1 + abs(y)) if log_modulus else y
        self.Draw.graph(arange(y.size) + t0, y, title=self.Tit, x_tit='Time [hh:mm]', y_tit='{}Current [fA]'.format('Log ' if log_modulus else ''), t_ax_off=t0 if rel_t else 0,
                        w=2, markersize=.3)

    def draw_mean_time(self, n_last=0, log_modulus=True, rel_t=False):
        y = self.get_mean_current(n_last)
        y = sign(y) * log10(1 + abs(y)) if log_modulus else y
        self.Draw.graph(self.get_mean_time(), y, title=self.Tit, x_tit='Time [hh:mm]', y_tit='{}Current [fA]'.format('Log ' if log_modulus else ''), w=2, markersize=.3,
                        t_ax_off=self.get_t0() if rel_t else 0)

    def draw(self, n_last=20, log_modulus=True):
        x, y = self.get_voltage(), self.get_mean_current(n_last)
        # y = sign(y) * log10(1 + abs(y)) if log_modulus else y
        # self.Draw.graph(self.get_voltage(), y, y_tit='{}Current [fA]'.format('Log ' if log_modulus else ''), x_tit='Voltage [V]', w=2, markersize=.6, grid=True)
        self.Draw.graph(x, sign(x) * y, y_tit='Current [fA]', x_tit='Voltage [V]', w=2, markersize=.6, grid=True, center_x=True, center_y=True, x_off=1.2, lm=.08, tit_size=.05, lab_size=.045,
                        y_range=[min(abs(y) / 2), max(abs(y)) * 2], logy=True, y_off=.8, bm=.2)

    def draw_same(self, n_last=20):
        x, y = self.get_voltage(), self.get_mean_current(n_last)
        data = array([[abs(x[cut]), sign(x[cut]) * y[cut]] for cut in [x < 0, x > 0]])
        graphs = [self.Draw.make_tgrapherrors(*idata, color=col) for idata, col in zip(data, [2, 1])]
        mg = self.Draw.multigraph(graphs, 'IV for {}'.format(self.Tit), ['Negative Bias', 'Positive Bias'], markersize=.6, grid=True, w=2, lm=.08, bm=.2, logy=True)
        v = concatenate(data[:, 1])
        format_histo(mg, tit_size=.05, lab_size=.045, y_range=[min(v[v > 0]) / 2, max(v) * 2], y_off=.8, y_tit='Current [{}]'.format(self.Unit), x_tit='Voltage [V]', center_x=True,
                     center_y=True)


if __name__ == '__main__':
    main_parser = ArgumentParser()
    main_parser.add_argument('filename', nargs='?', default='II6-B5_BIG_SIDE_180330-NoSource.iv')
    args = main_parser.parse_args()

    z = IV(args.filename)

