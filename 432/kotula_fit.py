from sympy import symbols, simplify, oo
from sympy.solvers import solve

Gf = symbols('G_f')  # filler modulus
Gm = symbols('G_m')  # matrix modulus
G = symbols('G')  # composite modulus
[t, s] = symbols(['t', 's'], positive=True)  # exponents in Kotula model
phif = symbols('phi_f', positive=True)  # filler fraction
phifc = symbols('phi_f^c', positive=True)  # critical filler fraction
Ap = (1-phifc)/phifc

# specify function to solve to get G
F = ((1-phif)*(Gm**(1/s)-G**(1/s))/(Gm**(1/s)+Ap*G**(1/s)) +
     phif*(Gf**(1/t)-G**(1/t))/(Gf**(1/t)+Ap*G**(1/t)))

# now set Gm to zero
G2 = F.subs(Gm, 0)
G2solve = simplify(solve(G2, G)[0])

# set Gf to infinity
G1 = F.limit(Gf, oo)
G1solve = solve(G1, G)[0]
