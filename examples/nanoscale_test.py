import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import patches
from matplotlib.colors import LogNorm
from matplotlib.path import Path

from fieldplot import GetFlow3D

from math import sqrt, cos, sin, acos, pi, atan2
import quadpy

from numpy.testing import assert_allclose

EPS = 1e-8

def _rotate(m, v):
    return np.array(m).dot(np.array(v))

def rotateAroundX(v, angle):
    return _rotate(
        [[1, 0, 0],
        [0, cos(angle), sin(angle)],
        [0, -sin(angle), cos(angle)]], v)

def rotateAroundY(v, angle):
    return _rotate(
        [[cos(angle), 0, -sin(angle)],
        [0, 1, 0],
        [sin(angle), 0, cos(angle)]], v)

def rotateAroundZ(v, angle):
    return _rotate(
        [[cos(angle), sin(angle), 0],
        [-sin(angle), cos(angle), 0],
        [0, 0, 1]], v)

# oriented angle from a to b
# def angle2D(ax, ay, bx, by):
#     a = np.array([ax, ay, 0])
#     b = np.array([bx, by, 0])
#     a = a / np.linalg.norm(a)
#     b = b / np.linalg.norm(b)

#     angle = acos(min(1.0, a.dot(b)))
#     cross = np.cross(a, b)
#     if np.array([0, 0, 1]).dot(cross) < 0:
#         angle = -angle

#     return -angle

def angle2D(ax, ay, bx, by):
    a_angle = atan2(ay,ax)
    b_angle = atan2(by,bx)
    angle = b_angle-a_angle
    return -angle

import unittest
from random import random

class TestRotations(unittest.TestCase):

    def testRotateAroundXSimple(self):
        assert_allclose(rotateAroundX([1.0, 1.0, 1.0], pi / 2), np.array([1.0, 1.0, -1.0]))

    def testRotateAroundYSimple(self):
        assert_allclose(rotateAroundY([1.0, 1.0, 1.0], pi / 2), np.array([-1.0, 1.0, 1.0]))

    def testRotateAroundZSimple(self):
        assert_allclose(rotateAroundZ([1.0, 1.0, 1.0], pi / 2), np.array([1.0, -1.0, 1.0]))

    def testRotateZeroAngle(self):
        assert_allclose(rotateAroundX([1.0, 1.0, 1.0], 0.0), np.array([1.0, 1.0, 1.0]))

    def testRotateAroundSameAxie(self):
        assert_allclose(rotateAroundX([1.0, 0.0, 0.0], 1.0), np.array([1.0, 0.0, 0.0]))

    def testRotateZeroLengthVector(self):
        assert_allclose(rotateAroundX([0.0, 0.0, 0.0], 1.0), np.array([0.0, 0.0, 0.0]))

    def testAngle2DSimple(self):
        self.assertAlmostEqual(angle2D(1.3, 1.3, 0.4, 0.0), pi / 4)

    def testAngle2DNegativeAngle(self):
        self.assertAlmostEqual(angle2D(0.4, 0.0, 1.3, 1.3), -pi / 4)

    def testAngle2DZeroAngle(self):
        self.assertAlmostEqual(angle2D(1.3, 1.3, 0.4, 0.4), 0.0)

    def testRotationAroundXMakesThemEquals(self):
        for t in range(10):
            fromX = (random() - 0.5) * 10
            fromY = (random() - 0.5) * 10
            toX = (random() - 0.5) * 10
            toY = (random() - 0.5) * 10
            message = "fromX = %f, fromY = %f, toX = %f, toY = %f" % (fromX, fromY, toX, toY)

            angle = angle2D(fromX, fromY, toX, toY)
            new_v = rotateAroundX([0.0, fromX, fromY], angle)

            self.assertAlmostEqual(new_v[0], 0.0, msg=message)
            self.assertAlmostEqual(angle2D(new_v[1], new_v[2], toX, toY), 0.0, msg=message)

    def testRotationAroundYMakesThemEquals(self):
        for t in range(10):
            fromX = (random() - 0.5) * 10
            fromY = (random() - 0.5) * 10
            toX = (random() - 0.5) * 10
            toY = (random() - 0.5) * 10
            message = "fromX = %f, fromY = %f, toX = %f, toY = %f" % (fromX, fromY, toX, toY)
            
            angle = -angle2D(fromX, fromY, toX, toY)
            new_v = rotateAroundY([fromX, 0.0, fromY], angle)

            self.assertAlmostEqual(new_v[1], 0.0, msg=message)
            self.assertAlmostEqual(angle2D(new_v[0], new_v[2], toX, toY), 0.0, msg=message)

    def testRotationAroundZMakesThemEquals(self):
        for t in range(10):
            fromX = (random() - 0.5) * 10
            fromY = (random() - 0.5) * 10
            toX = (random() - 0.5) * 10
            toY = (random() - 0.5) * 10
            message = "fromX = %f, fromY = %f, toX = %f, toY = %f" % (fromX, fromY, toX, toY)

            angle = angle2D(fromX, fromY, toX, toY)
            new_v = rotateAroundZ([fromX, fromY, 0.0], angle)

            self.assertAlmostEqual(new_v[2], 0.0, msg=message)
            self.assertAlmostEqual(angle2D(new_v[0], new_v[1], toX, toY), 0.0, msg=message)


if __name__ == '__main__':
    unittest.main()

def fieldplot2(Eabs, coordX, coordZ, x, m, npts, factor, flow_total=0):
    field_to_plot='Eabs'
    WL_units='nm'
    outline_width = 1

    fig, ax = plt.subplots(1,1, figsize=(8,8))#, sharey=True, sharex=True)
    fig.tight_layout()

    label = r'$|E|$'

    Eabs_data = np.resize(Eabs, (npts, npts)).T

    # Define scale ticks
    min_tick = np.amin(Eabs_data[~np.isnan(Eabs_data)])
    #min_tick = 0.1
    max_tick = np.amax(Eabs_data[~np.isnan(Eabs_data)])
    #max_tick = 60
    scale_ticks = np.linspace(min_tick, max_tick, 5)
    #scale_ticks = np.power(10.0, np.linspace(np.log10(min_tick), np.log10(max_tick), 6))
    #scale_ticks = [0.1,0.3,1,3,10, max_tick]
    # Interpolation can be 'nearest', 'bilinear' or 'bicubic'
    ax.set_title(label)
    # build a rectangle in axes coords
    ax.annotate(' ', xy=(0.0, 1.1), xycoords='axes fraction',  # fontsize=10,
                horizontalalignment='left', verticalalignment='top')

    cax = ax.imshow(Eabs_data, interpolation='nearest', cmap=cm.jet,
                    origin='lower', vmin=min_tick, vmax=max_tick, extent=(min(coordX), max(coordX), min(coordZ), max(coordZ))
                    # ,norm = LogNorm()
                    )
    ax.axis("image")

    # Add colorbar
    cbar = fig.colorbar(cax, ticks=[a for a in scale_ticks], ax=ax)
    # vertically oriented colorbar
    if 'angle' in field_to_plot:
        cbar.ax.set_yticklabels(['%3.0f' % (a) for a in scale_ticks])
    else:
        cbar.ax.set_yticklabels(['%g' % (a) for a in scale_ticks])
    # pos = list(cbar.ax.get_position().bounds)
    #fig.text(pos[0] - 0.02, 0.925, '|E|/|E$_0$|', fontsize = 14)
    lp2 = -10.0
    lp1 = -1.0

    #if crossplane == 'XZ':
    ax.set_xlabel('Z, ' + WL_units, labelpad=lp1)
    ax.set_ylabel('X, ' + WL_units, labelpad=lp2)

    for r in x:
        s1 = patches.Arc((0, 0), 2.0 * r, 2.0 * r,  angle=0.0, zorder=1.8,
                         theta1=0.0, theta2=360.0, linewidth=outline_width, color='black')
        ax.add_patch(s1)
    #
    # (not crossplane == 'XY') and 
    if flow_total > 0:
        scanSP = np.linspace(-factor * x[-1], factor * x[-1], npts)
        min_SP = -factor * x[-1]
        step_SP = 2.0 * factor * x[-1] / (flow_total - 1)
        x0, y0, z0 = 0, 0, 0
        max_length = factor * x[-1] * 10
        max_angle = np.pi / 160
        rg = range(0, flow_total)
        for flow in rg:
            f = min_SP + flow*step_SP
            x0 = f
            z0 = min_SP
            
            flow_xSP, flow_ySP, flow_zSP = GetFlow3D(x0, y0, z0, max_length, max_angle, x, m, -1)
            
            #flow_z_plot = flow_zSP * WL / 2.0 / np.pi
            #flow_f_plot = flow_xSP * WL / 2.0 / np.pi
            flow_z_plot = flow_zSP
            flow_f_plot = flow_xSP

            verts = np.vstack((flow_z_plot, flow_f_plot)).transpose().tolist()
            codes = [Path.LINETO] * len(verts)
            codes[0] = Path.MOVETO
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=outline_width, edgecolor='white', zorder=1.9, alpha=0.7)
            ax.add_patch(patch)
            
    fig.subplots_adjust(hspace=0.3, wspace=-0.1)


def get_points():
    xs = quadpy.sphere.Lebedev("19").points

    
