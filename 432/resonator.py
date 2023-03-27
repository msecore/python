# import all the functions we need
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, I, exp, re, im, sqrt, atan, lambdify, pi

# specify the Magnitude and phase angle for the complex modulus
Gmag, phi = symbols(['|G^{*}|', 'phi'], real=True, positive=True)

# specify Gprime and G double prime
# need to define things so that these are real and positive
Gp, Gpp = symbols(["G'", "G''"], real=True, positive=True)

# construct the complex modulus
Gstar = Gp+I*Gpp

# C is the constant relating the torsional stiffness to G
C = symbols('C', real=True, positive=True)

# K is the torsionsal stiffness
K = C*Gstar

# Ib is the moment of inertia
Ib = symbols('I_m', real=True, positive=True)

# calculate the quality factor
Q = 2*np.pi/phi

# define real and imaginary components of w as wp and wpp, along with time, t
wp, wpp, t = symbols(["\\omega'", "\\omega''", 't'], real=True, positive=True)

# we have to different version of theta
theta = re(exp(I*(wp+I*wpp)*t)).simplify()

# now calculate the complex frequency and calculate theta['G'], which is 
# theta in terms of the detailed parameters in the problem
wp2 = re(sqrt(K/Ib)).subs(Gp**2+Gpp**2, Gmag**2).subs(atan(Gpp/Gp), phi)
wpp2 = im(sqrt(K/Ib)).subs(Gp**2+Gpp**2, Gmag**2).subs(atan(Gpp/Gp), phi)

# calculate the quality factor
Q = wp2/(4*pi*wpp2).simplify()

# now we create a bunch of plottable functions
fun = {}
fun['theta'] = lambdify([t, wp, wpp], theta, 'numpy')
fun['wp'] = lambdify([C, Ib, Gmag, phi], wp2, 'numpy')
fun['wpp'] = lambdify([C, Ib, Gmag, phi], wpp2, 'numpy')
fun['Q'] = lambdify(phi, Q, 'numpy')

# now specify the values of the parameters in our example
# start with mechanical properties of the fiber
Gstar_val = 1e9+1J*1e7
Gmag_val = abs(Gstar_val)
phi_val = np.angle(Gstar_val)

# geometry of the fiber
df_val = 1e-3  # diameter 
lf_val = 30e-2  # length

# properties of the suspended mass
rhob = 7850 # density in Kg/m^3
db_val = 0.01
lb_val = 0.1

Ib_val = np.pi*db_val**2*lb_val**3*rhob/48
C_val = np.pi*df_val**4/(32*lf_val)

# now calculate the real and imaginary components of the 
wp_val = fun['wp'](C_val, Ib_val, Gmag_val, phi_val)
wpp_val = fun['wpp'](C_val, Ib_val, Gmag_val, phi_val)
tau_val = 1/wpp_val
Q_val = fun['Q'](phi_val)

# now plot the time dependence of theta
# close any previous plts that may be open
plt.close('all')

# make the figure and the axes
fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True, 
                       num = 'viscoelastic')
ax.set_xlabel('$t$ (s)')
ax.set_ylabel(r'$\theta/\theta_0$')

# specify the time points to use
t_vals = np.linspace(0, 100, 1000)

# now calculate the theta values
theta_vals = fun['theta'](t_vals, wp_val, wpp_val)

ax.plot(t_vals, theta_vals, '-')

# add annotation showing quality fact1or
ax.text(15,0.95, r'$Q\approx16$', color = 'red')
ax.vlines(np.pi*2*Q_val/wp_val,-1, 1, color = 'red')
ax.hlines(np.sqrt(1/np.e), 0, 50, color = 'red')
ax.text(55, np.sqrt(1/np.e), r'$\sqrt{1/e}$', color = 'red',
        verticalalignment='center')

fig.show()
fig.savefig('../figures/torsional_theta_plot.svg')

# now make a figure showing just the elastic solution
fig, ax = plt.subplots(1,1, figsize=(4,1.5), constrained_layout = True,
                       num = 'elastic')
ax.set_xlabel ('$t$ (s)')
ax.set_ylabel('$\Theta$')

t_vals = np.linspace(0, 4*2*np.pi/wp_val, 100)
ax.plot(t_vals, np.cos(wp_val*t_vals))
fig.savefig('../figures/torsional_theta_plot_elastic.svg')
fig.show()

