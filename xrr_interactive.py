#!/var/tmp/mpfluege-s/anaconda3/bin/python3

import numpy as np

import pyqtgraph as pg
## Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from pyqtgraph.Qt import QtCore, QtGui

import sys
sys.path.append('/home/mpfluege/python')
import xray_compounds as xc
import pint
u = pint.UnitRegistry()

import numba

from matrixmethod_numba import fields_positions_fields, fields_positions_positions

energy = 2000 * u.eV

n = np.array([1, 1-0.00010885192136989019+1.8106738274706911e-05j, 1-0.00020885192136989019+2.8106738274706911e-05j, 1])
rough = np.array([2.]*3)
thick = np.array([5, 20])  # nm
wl = energy.to(u.nm, 'sp').magnitude  # nm
ang_deg = np.linspace(0, 5, 10001)[1:]
ang = np.deg2rad(ang_deg)
positions = np.linspace(20, -30, 25001)

fr, ft, k_z, Z = fields_positions_fields(n, wl, ang, thick, rough)

app = QtGui.QApplication([])

## Define a top-level widget to hold everything
w = QtGui.QWidget()

## Create some widgets to be placed inside
field_plot = pg.PlotWidget()
xrr_plot = pg.PlotWidget()
theta_text_fmt = 'Theta: {:.3f}°'
theta_text = QtGui.QLabel(theta_text_fmt.format(ang_deg[-1]))

## Create a grid layout to manage the widgets size and position
layout = QtGui.QVBoxLayout()
w.setLayout(layout)

## Add widgets to the layout in their proper positions
layout.addWidget(theta_text)
layout.addWidget(xrr_plot,)
layout.addWidget(field_plot)

xrr_data = np.abs(fr[:,0])**2
xrr = xrr_plot.plot(ang_deg, xrr_data, pen=(0, 0, 0), name='Reflectivity')
trans_data = np.abs(ft[:,-1])**2
trans = xrr_plot.plot(ang_deg, trans_data, pen=(255, 0, 0), name='Transmitance')
xrr_plot.setLogMode(x=False, y=True)
theta_line = pg.InfiniteLine(ang_deg[-1])
theta_line.movable = True
xrr_plot.addItem(theta_line)
#xrr_plot.showGrid(x=True, y=True)
#xrr_plot.enableAutoRange('xy', False)
#xrr_plot.setYRange(1, 1e-6, padding=0)

# plot
abs = field_plot.plot(-positions, np.zeros_like(positions), pen=(0, 0, 0), name='abs')
real = field_plot.plot(-positions, np.zeros_like(positions), pen=(255, 0, 0), name='Re')
field_plot.enableAutoRange('xy', False)
field_plot.setYRange(-2, 2)
#field_plot.showGrid(x=True, y=True)
field_plot.addItem(pg.InfiniteLine(0, pen=(128, 128, 0)))
field_plot.addItem(pg.InfiniteLine(thick[:1].sum(), pen=(128, 128, 0)))
field_plot.addItem(pg.InfiniteLine(thick[:2].sum(), pen=(128, 128, 0)))
field_plot.addItem(pg.InfiniteLine(thick[:3].sum(), pen=(128, 128, 0)))


@numba.jit(cache=True)
def replot(pos):
    post, posr = fields_positions_positions(positions, fr[pos], ft[pos], k_z[pos], Z)
    abs.setData(-positions, np.abs(post + posr))
    real.setData(-positions, np.real(post + posr))
    theta_line.setValue(ang_deg[pos])


def line(l):
    val = l.value()
    pos = np.abs(ang_deg - val).argmin()
    theta_text.setText(theta_text_fmt.format(ang_deg[pos]))
    replot(pos)

theta_line.sigDragged.connect(line)

replot(-1)
theta_text.setText(theta_text_fmt.format(ang_deg[-1]))


## Display the widget as a new window
w.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
