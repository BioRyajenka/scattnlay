from math import sqrt, cos, sin, acos, pi, atan2
from matplotlib import cm
from matplotlib import patches
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from numba import complex128,float64,int64,jit
import matplotlib.pyplot as plt
import numpy as np
import quadpy

#custom module
from fieldplot import GetFlow3D
from nanoscale_test import fieldplot2, rotateAroundX, rotateAroundY, rotateAroundZ, angle2D
from scattnlay import fieldnlay
from scattnlay import scattnlay
import nanoscale_test
import scattnlay

#global script parameters
WL=455 #nm 				# wave length
core_r = 90.0			# partical radius

index = 4.6265+0.13845j	# refractive index
#index = sqrt(16)
#index = 4.639+0.078841j
#index = 21.4**0.5

npts = 151					# plot will be npts x npts
factor=1.1					# area of plot

x = np.ones((1), dtype = np.float64)
x[0] = core_r
nm = 1.0						# refractive index of host media
m = np.ones((1), dtype = np.complex128)
m[0] = index/nm


def get_field(coord):
    """Get amplitude of electric near-field.

    Args:
        coord (n x 3 array): coordinates (in WL units)
                             of points to evaluate the field.

    Returns:
        Amplitude vectors of electric field.
    """
    _, E, _ = fieldnlay(np.array([2.0*np.pi*x/WL]),
                        np.array([m]),
                        2.0*np.pi*coord/WL,
                        pl=-1)
    Ec = E[0, :, :]
    Eamp = np.absolute(Ec)
    return Eamp


def get_points(selector, r=1., quad_n=19):
    """Get 3D distribution of points.

    Args:
        selector (string): type of distribution
                           "quad" - Lebedev quadrature points
                           "meshXY", "meshXZ", "meshYZ" - uniform mesh
                r (float): "quad" only. A scale raduis.
             quad_n (int): "quad" only. Quadrature polynomial order.
                           Some of allowed values are 5, 13, 19, 131
                           
    Returns:
        array of points for fieldnlay. 
    
    """
    if selector=="quad":
        coord = quadpy.sphere.Lebedev(str(quad_n)).points * r
        return coord
    
    scan = np.linspace(-factor*x[-1], factor*x[-1], npts)
    zero = np.zeros(npts*npts, dtype = np.float64)

    coordA1, coordA2 = np.meshgrid(scan, scan)
    coordA1.resize(npts * npts)
    coordA2.resize(npts * npts)
    
    if selector=="meshXY":
        return np.vstack((coordA1, coordA2, zero)).transpose()
    if selector=="meshXZ":
        return np.vstack((coordA1, zero, coordA2)).transpose()
    if selector=="meshYZ":
        return np.vstack((zero, coordA1, coordA2)).transpose()
    
    raise ValueError('Unknown selector for get_points()')


def multi_hstack(stack_vec_list):
    """hstack multiple np.arrays

    Args:
       stack_vec_list: tuple of tuples to be stacked
    Return:
       tuple of stacked arrays
    """
    stacked = []
    for vec_list in stack_vec_list:
        stacked.append(np.hstack(vec_list))
    return stacked


def cart2sph(xyz):
    """Add columns with cartesian coordinats converted to spherical

    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def get_angles(coords):
    ptsnew = np.zeros(coords.shape)
    ptsnew[:,0] = np.arctan2(coords[:,1], coords[:,2])*np.sign(coords[:,2])
    ptsnew[:,1] = -np.arctan2(coords[:,0], coords[:,2])
    ptsnew[:,2] = -np.arctan2(coords[:,0], coords[:,1])
    return ptsnew


@jit(#complex128(complex128[:], complex128[:]),
     nopython=True, cache=True, nogil=True)
def isclose(a,b):
    atol = 1e-8
    rtol = 1e-5
    return np.absolute(a - b) <= (atol + rtol * np.absolute(b))

## Derivation of equation for the projection
# b - coord
# a - proj  #it is in XZ plane, so a2=0
# a1^2+a3^2 = 1  #vector norm
# a3 = sqrt(1-a1^2)
# a1*b1 + a3*b3 = 0   #dot product with a2=0
# a1*b1 + sqrt(1-a1^2)*b3 = 0
# a1*b1 = -sqrt(1-a1^2)*b3 
# a1^2 * b1^2 = (-1)^2* (1-a1^2)*b3^2 
# a1^2 * b1^2 - (1-a1^2)*b3^2 = 0
# a1^2 * (b1^2 + *b3^2) = b3^2
# a1^2   = b3^2/(b1^2 + *b3^2)
# a1  = sqrt(b3^2/(b1^2 + *b3^2))
@jit(#complex128(complex128[:], complex128[:]),
     nopython=True, cache=True, nogil=True)
def get_projections(coords, pol=0):
    """Get equivalent dipole vectors to project field on it.

    Args:
        coords (array of points): position of points to evaluate projections.
                       pol (int): polarization of the projection. 0 - along X, 1 - along Y.
    Return:
        Projection vectors constructed so that: 1) each is along the normal to coord vector
        2) it is in XZ plane for pol=0 or in YZ plane for pol=1    
    """
    prj = np.zeros((len(coords),3))
    b = coords
    prj[:,pol] = -np.sqrt(b[:,2]**2/(b[:,pol]**2 + b[:,2]**2))
    prj[:,2] = np.sqrt(1-prj[:,pol]**2)
    for i in range(len(coords)):
        if isclose(coords[i][pol],0.) and isclose(coords[i][2],0.):
            # print("zero")
            single = np.array([0.0, 0.0, 0.0])
            single[pol] = -1.
            prj[i] = single
        if isclose(angle_between(coords[i],prj[i]),pi/2) == False:        
            for j in range(2):
                prj[i][0] *=(-1)**j
                prj[i][2] *=(-1)**(j+1)            
                if isclose(angle_between(coords[i],prj[i]),pi/2) == True: break
        #Check the result
        if not isclose(angle_between(coords[i],prj[i]),pi/2):
            print("!!!!!! Projection problem !!!!!!!")
    return prj


#https://stackoverflow.com/a/13849249/4280547
@jit(#complex128(complex128[:], complex128[:]),
     nopython=True, cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

#https://stackoverflow.com/a/13849249/4280547
@jit(#complex128(complex128[:], complex128[:]),
     nopython=True, cache=True, nogil=True)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot = np.dot(v1_u, v2_u)
    if dot < -1: dot = -1
    if dot > 1: dot = 1
    return np.arccos(dot)


def visualization_test_of_projection_vectors(pol=0):
    """Plot projections vectors to visualy verify them """
    coords = get_points('quad', r=core_r*4./5., quad_n=5)
    Eamp = get_field(coords)
    prj = get_projections(coords, pol=pol)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]
    scale = 15.
    U = prj[:, 0]*scale
    V = prj[:, 1]*scale
    W = prj[:, 2]*scale
    # plot projections with origin at coord, and coords from zero
    U, V, W, X, Y, Z = multi_hstack(( (U,X), (V,Y), (W,Z),
                                     (X,X*0.), (Y,Y*0.), (Z,Z*0.)  ))

    ax.quiver(X, Y, Z, U, V, W)
    plt.show()


#visualization_test_of_projection_vectors(0)

# coords = get_points('meshXZ')
# Eamp = get_field(coords)
# Eabs = np.sqrt(Eamp[:, 0]**2 + Eamp[:, 1]**2 + Eamp[:, 2]**2)
# fieldplot2(Eabs, coords[:,0], coords[:,2], x, m, npts, factor)
# plt.show()
coords = get_points('quad', r=core_r*4./5., quad_n=131)
Eamp = get_field(coords)
for i in range(200):
    #coords = coords[:6]
    prj = get_projections(coords, pol=0)
print(len(coords))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=Eabs*10)
# X = coords[:,0]
# Y = coords[:,1]
# Z = coords[:,2]
# # Emax = np.max(Eamp)/15
# # U = Eamp[:, 0]/Emax
# # V = Eamp[:, 1]/Emax
# # W = Eamp[:, 2]/Emax
# scale = 15.
# U = prj[:, 0]*scale
# V = prj[:, 1]*scale
# W = prj[:, 2]*scale

# scale = 1.
# U, V, W, X, Y, Z = multi_hstack(( (U,X/scale), (V,Y/scale), (W,Z/scale),
#                                   (X,X*0.), (Y,Y*0.), (Z,Z*0.)  ))

# ax.quiver(X, Y, Z, U, V, W)
# plt.show()
