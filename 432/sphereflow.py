from sympy import symbols, diff, cos, sin, cot, eye, Rational, simplify, sqrt
from sympy import series
from sympy import lambdify, integrate, pi
from sympy.matrices import Matrix
import matplotlib.pyplot as plt
import numpy as np

theta = symbols('theta')
r = symbols('r', positive=True)
R, u0 = symbols('R, u_0', positive=True)
G = symbols('G', positive=True)
rbar, ubar = symbols('rbar, ubar', positive=True)


ur = u0*cos(theta)*(1-Rational(3, 2)*(R/r)+Rational(1, 2)*(R/r)**3)
ut = -u0*sin(theta)*(1-Rational(3, 4)*(R/r)-Rational(1, 4)*(R/r)**3)

# specify gradients
urr = diff(ur, r)
urt = diff(ur, theta)

utr = diff(ut, r)
utt = diff(ut, theta)

# specify gradient tensor
# see https://www.brown.edu/Departments/Engineering/Courses/En221/Notes/Polar_Coords/Polar_Coords.htm
I = eye(3)
grad = Matrix([[urr, urt/r-ut/r, 0],
               [utr, utt/r+ur/r, 0],
               [0, 0, cot(theta)*ut/r]])

# write in terms of rbar=r/R; ubar=u0/R
grad = grad.subs(r, rbar*R)
grad = grad.subs(u0, ubar*R)
grad = simplify(grad)

F = I + grad

# B is the Left Cauchy-Green deformation tensor
B = F*F.transpose()

# Now we'll get the stress tensor for an inompressible Neo-Hookean model
sig = G*(B - Rational(1, 3)*B.trace()*I)

# add the hydrostatic pressure, simga_all is valid for all rbar
p = Rational(3, 2)*ubar*G*cos(theta)/rbar**2
sig = sig - I*p


# sig_int is stress at the interface
sig_int = sig.subs(rbar, 1)

# now calculate the force acting outward from the sphere
Fn = sig_int*Matrix([1, 0, 0])
sig_r = Fn[0]
sig_t = Fn[1]

# now get component of the stress in the forward direction
sig_for = -sig_t*sin(theta) - sig_r*cos(theta)
sig_for = sig_for.simplify()

# now integrate the total forward stress
P_tot = integrate(2*pi*R**2*sin(theta)*sig_for, (theta, 0, pi))

# calculte the average stress
sig_avg = P_tot/(pi*R**2)

# calculate the normalized stress tensor
sig_norm = sig/sig_avg

# define normalized pressure
p_norm = p/sig_avg

# calculate the normalized principal stresses (this takes a while)
sig_p_norm = list(sig_norm.eigenvals().keys())

# now calculate the von mises stress
vonmises_norm = Rational(1,2)*sqrt(2)*sqrt((sig_p_norm[2]-sig_p_norm[1])**2+
                (sig_p_norm[2]-sig_p_norm[0])**2+
                (sig_p_norm[1]-sig_p_norm[0])**2)

# calculate the principal stretch ratios
lam_p = []
lam_p_lin =[] # linearized in ubar
lamsquared = list(B.eigenvals().keys())
for k in [0, 1, 2]:
    lam_p.append(sqrt(lamsquared[k]).simplify())
    lam_p_lin.append(series(lam_p[k], ubar, 0, 2).removeO())


# %% now make a bunch of plots - first set up the axes
plt.close('all')
fig, ax = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
for k in [0, 1, 2]:
    ax[k].set_xlabel(r'$\theta$ (deg.)')
    ax[k].set_xlim([0, 180])
    ax[k].set_xticks([0, 45, 90, 135, 180])

ax[0].set_ylabel(r'$\sigma_k^p/\sigma_{avg}$')
ax[0].set_title(r'principal stresses')
ax[1].set_ylabel(
    r'($\sigma_\theta$, -$\sigma_r$, $\tau_{max}$, $p$)/$\sigma_{avg}$')
ax[1].set_title(r'other stresses')
ax[2].set_ylabel(r'$\sigma_k^p/\sigma_{avg}$')
ax[2].set_title(r'principal stretches')

# create callable functions for the lots we want to make
fcn = {}
fcn['sig_r'] = lambdify([theta, ubar], (sig_r/sig_avg).simplify(), 'numpy')
fcn['sig_t'] = lambdify([theta, ubar], sig_t/sig_avg, 'numpy')
fcn['p_norm'] = lambdify([theta,rbar], p_norm, 'numpy')
fcn['vonmises_norm'] = lambdify([theta, rbar, ubar], vonmises_norm, 'numpy')
fcn['sig_p_norm'] = {}
fcn['lam_p'] = {}
for k in [0, 1, 2]:
    fcn['sig_p_norm'][k] = lambdify([theta, rbar, ubar], sig_p_norm[k], 'numpy')
    fcn['lam_p'][k] = lambdify([theta, ubar], lam_p[k].subs(rbar, 1), 'numpy')

# now plot the data
th = np.linspace(0, 180, 100)

# use th2 as x variable for plots that we mark with symbols
th2 = np.linspace(0, 180, 20)

color = {0: 'C0', 1: 'C1', 2: 'C2'}


def tau_max_single(theta, rbar, ubar):
    # calculates the max. shear stress from the principal stresses
    tau0 = abs(fcn['sig_p_norm'][1](theta, rbar, ubar)-
               fcn['sig_p_norm'][2](theta, rbar, ubar))
    tau1 = abs(fcn['sig_p_norm'][0](theta, rbar, ubar)-
               fcn['sig_p_norm'][2](theta, rbar, ubar))
    tau2 = abs(fcn['sig_p_norm'][0](theta, rbar, ubar)-
               fcn['sig_p_norm'][1](theta, rbar, ubar))
    return 0.5*max(tau0, tau1, tau2)


tau_max = np.vectorize(tau_max_single)


def plot_props(ubar, sym):
    ax[1].plot(th, -fcn['sig_t'](np.pi*th/180, ubar), 'C0' + sym,
               label=r'$\sigma_{\theta}$')
    ax[1].plot(th, -fcn['sig_r'](np.pi*th/180, ubar), 'C1' + sym,
               label=r'-$\sigma_{r}$')
    ax[1].plot(th2, tau_max(np.pi*th2/180, 1, ubar), 'C2+',
               label=r'$\tau_{max}$')

    for k in [0, 1, 2]:
        ax[0].plot(th, fcn['sig_p_norm'][k](np.pi*th/180, 1, ubar),
                   color[k]+sym, label='k='+str(k))
        # need to multiply by np.ones in case where the function is a constant
        ax[2].plot(th, np.ones(len(th))*fcn['lam_p'][k](np.pi*th/180, ubar),
                   color[k]+sym, label='k='+str(k))


# plot normlized tau_max, radial and tangential stress at r=R in low strain limit
plot_props(0, '-')

# now add the normallized pressure (note that this is independent of ubar)
ax[1].plot(th2, fcn['p_norm'](np.pi*th2/180, 1), 'C3+', label=r'$p$')

# add legends
ax[0].legend()
ax[1].legend(ncol=2)

fig.savefig('../figures/sphere_stress_plots.pdf')

# %%  Now make polar plots
plt.close('contour_plots')
numplots = 2

def convert_to_polar(xbar, ybar):
    rbar = np.sqrt(xbar**2+ybar**2)
    if ybar <= 0:
        theta = np.arctan(xbar/ybar)
    else:
        theta = np.pi+np.arctan(xbar/ybar)
    return rbar, theta

def tau_max_cart(xbar, ybar, ubar):
    rbar, theta = convert_to_polar(xbar, ybar) 
    if rbar < 0.5:
        return 0
    else:
        return tau_max(theta, rbar, ubar)
    
def p_cart(xbar, ybar):
    rbar, theta = convert_to_polar(xbar, ybar) 
    if rbar < 0.5:
        return 0
    else:
        return fcn['p_norm'](theta, rbar)

vtau_max_cart = np.vectorize(tau_max_cart)
vp_cart = np.vectorize(p_cart)

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)

xvals, yvals = np.meshgrid(x, y)
zvals = {}
zlevels = {}
titles = {}
ticks = {}

# this version plots linearlized versions of shear stress and pressure
# specify what we need for th tau_max plot
zvals[0] = 6*abs(vtau_max_cart(xvals, yvals, 0)) # normalization is GU/R
zlevels[0] = np.linspace(0, 1.5, 200)
titles[0] = r'$\frac{\left|\tau_{max}\right|R}{UG}$'
ticks[0] = np.linspace(0, 1.5, 6)

# now specify what we need for the pressure plot
zvals[1] = -6*vp_cart(xvals, yvals) # normalization is GU/R, neg. sign because particle is moving up
zlevels[1] = np.linspace(-1.5, 1.5, 200)
titles[1] = r'$\frac{pR}{UG}$'
ticks[1] = np.linspace(-1.5, 1.5, 11)


# set up the axes for the plots
fig, ax = plt.subplots(1, numplots, figsize=(numplots*4, 4), squeeze = False, 
                       constrained_layout='True', num = 'contour_plots')
for k in np.arange(numplots):
    ax[0,k].set_aspect(1)
    ax[0,k].set_xlabel('x/R')
    ax[0,k].set_ylabel('y/R')

contour = {}
cbar = {}
circle = {}
for k in np.arange(numplots):
    # now make the contour plot
    contour[k] = ax[0,k].contourf(xvals, yvals, zvals[k], levels=zlevels[k], 
                                cmap='jet')

    # add the colorbars
    cbar[k] = fig.colorbar(contour[k], ticks=ticks[k], 
                       fraction=0.05, pad=0.04, ax=ax[0,k])
    ax[0,k].set_title(titles[k], pad=20, fontsize=20)

    # now draw circles corresponding to location of the particle
    circle[k] = plt.Circle((0, 0), 1, facecolor='white', edgecolor='black')
    ax[0,k].add_artist(circle[k])
    
    # add arrow showing direction of particle motion
    ax[0,k].arrow(0, -0.5, 0, 0.8, width = 0.03, head_width = 0.2)

fig.savefig('../figures/polar_plot.png')
fig.show()


    
