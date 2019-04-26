import numpy as np
import random as rnd
from scipy.special import binom
from scipy.integrate import ode
import os.path

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-f", action="store", dest="outfile_file",
                  help="name of file to write results to")

parser.add_option("--ode",  action="store_true", dest="solve_ode", default=False,
                  help="should we numerically solve the ODE")

parser.add_option("--odem",  action="store_true", dest="solve_ode_multiple", default=False,
                  help="should we numerically solve the ODE with multiple reactions")

parser.add_option("--cme",  action="store_true", dest="solve_cme", default=False,
                  help="should we simulate trajectory via Monte Carlo")

parser.add_option("--cmem",  action="store_true", dest="solve_cme_multiple", default=False,
                  help="should we simulate trajectory via Monte Carlo")

(options, args) = parser.parse_args()


def write_line(file, t, a):
  np.savetxt(file, [np.concatenate(([t], a), axis=0)])
### Solve reaction system
#
#                                  k_for
# alpha_1 X_1 + ... + alpha_I X_I <======> beta_1 X_1 + ... + beta_I X_I
#                                  k_back
# with stoichiometric coefficients alpha_1,.., alpha_I, beta_1,..., beta_I
# and forward and backward rates k_for and k_back respectively.
# currently only one reaction is supported (R=1)
#

### Parameters
# stochiometric vectors
alpha = np.array([1, 1, 0])
beta  = np.array([0, 0, 1])

alphar = np.array([[0,0],
                   [1,0],
                   [2,1],
                   [1,0]])
betar  = np.array([[1,0],
                   [0,1],
                   [3,0],
                   [0,0]])

gamma = beta - alpha
gammar = betar-alphar
I = len(alpha) # number of species

# reaction rates
k_for  = 1.0e-1 # forward rate
k_back = 2.0e2 # backward rate


k_for_r  = np.array([1.0e3, 3.0, 1.0e-6, 1.0])
k_back_r = np.array([0.0, 0.0, 0.0, 0.0])


Rnum = len(k_for_r)

V = 1.0e2     # volume
T = 1.0e-2  # final time
max_steps = 10000000 # maximal allowed number of steps


# auxiliary variables
t = 0.0       # current time 
step = 0      #step counter

# initial state
c0  = np.array([1.0e2, 2.0e3, 1.56e3])   # vector of densities
eta = np.array(c0*V).astype(int) # convert densities to vector of particle numbers by multiplying with volume V

# initial state
c0r  = np.array([1.0e3, 1.0e3])   # vector of densities
etar = np.array(c0r*V).astype(int) # convert densities to vector of particle numbers by multiplying with volume V

Ir = len(c0r)




### Solve Chemcial Master Equation via Kinetic Monte Carlo (KMC)
if options.solve_cme:
  ## helper function to compute the transition rates used in kinetic Monte Carlo
  #
  #  B^alpha_V(eta) := V^(1-|alpha|) k_for eta!/(eta-alpha!)
  #  B^beta_V(eta)  := V^(1-|beta|) k_back eta!/(eta-beta!)
  #
  # The CME is given by the generator L_V^*
  # mu'(eta) = (L_V^* mu)(eta) := B^alpha(eta+gamma)mu(eta+gamma)- (B^alpha(eta) + B^beta(eta))mu(eta) + B^beta(eta-gamma)mu(eta-gamma) 
  #
  # (CAVEAT: We do not take care of the case that eta_i-gamma_i < 0)
  def compute_rates(eta):
    factor_forw  = np.zeros((I))
    factor_backw = np.zeros((I))
    for i in range(I):
      factor_forw[i]  = np.prod(range(eta[i] + gamma[i] - alpha[i] + 1, eta[i] + gamma[i] + 1))   # the last +1 is necessary as range(i0,iN) gives i0,..,iN-1
      factor_backw[i] = np.prod(range(eta[i] - gamma[i] - beta[i]  + 1, eta[i] - gamma[i] + 1))

    B_forw  = V**(1 - np.sum(alpha)) * k_for  * np.prod(factor_forw)
    B_backw = V**(1 - np.sum(beta))  * k_back * np.prod(factor_backw)

    return B_forw, B_backw


  print("Solving CME...")
  # open output file for writing
  with open(options.outfile_file, "w+") as f:
    # simulation loop (stop if final time is achieved or maximal number of steps is reached)
    while t<T and step < max_steps:
      rate_forw, rate_backw = compute_rates(eta) # compute rates for transition depending on current state
      tau    = rnd.expovariate(1.0)/(rate_forw+rate_backw) # sample waiting time in state via exponential distribution
      t      = t + tau   # add waiting time to process time
      p      = rnd.uniform(0.0, rate_forw + rate_backw) # sample which reaction (forward or backward) is chosen
      if p <= rate_forw:
        eta = eta + gamma # forward step
      else:
        eta = eta - gamma # backward step
      step = step + 1 # increase step counter
      write_line(f, t, eta/V) # write to output file
    f.close()



if options.solve_ode:
  ### Solve ODE system
  #
  #  c'(t) = R(c(t) := (k_for*c(t)^alpha - k_back*c(t)^beta )*gamma
  #

  # we change file name by appending '_ode' since we want to write into different file
  filename = os.path.splitext(options.outfile_file)
  ode_file = filename[0]+'_ode'+filename[1]


  # polynomial reaction function
  def Reac(t, c):
      calpha = np.power(c, alpha)
      cbeta  = np.power(c, beta) 
      rate = k_for * np.prod(calpha) - k_back * np.prod(cbeta)
      return rate*gamma

  # derivative of reaction function for Newton method
  def dReacdc(t, c):
      J = np.zeros((I,I)) # Jacobian
      for i in range(I): # loop over components over density vector
        alpham     = np.copy(alpha) # have to copy arrays, otherwise alpha and alpm point to the same array
        betam      = np.copy(beta)
        alpham[i] -= 1 # lower exponents
        betam[i]  -= 1
        calpham    = np.prod(np.power(c, alpham))
        cbetam     = np.prod(np.power(c, betam))
        dRdci      = (k_for* alpha[i] * calpham - k_back * beta[i] * cbetam) *gamma
        J[:,i]     = dRdci   
      return J

  # setup numerical scheme for ODE system
  ode_system = ode(Reac, dReacdc).set_integrator('vode', method='bdf', with_jacobian=True)
  # set intial conditions (t_0 = 0.0)
  ode_system.set_initial_value(c0, 0.0)

  # time step 
  dt = T*1.0e-3

  print("Solving ODE...")
  with open(ode_file, "w+") as f:
    while ode_system.successful() and ode_system.t < T:
      ode_system.integrate(ode_system.t+dt)
      write_line(f, ode_system.t, ode_system.y)
  f.close()




if options.solve_ode_multiple:
  ### Solve ODE system with multiple reaction
  #
  #  c'(t) = R(c(t) := sum_r=1^R (k_for_r*c(t)^alpha_r - k_back_r*c(t)^beta_r )*gamma_r
  #

  # we change file name by appending '_ode' since we want to write into different file
  filename = os.path.splitext(options.outfile_file)
  ode_file = filename[0]+'_ode'+filename[1]


  # polynomial reaction function
  def Rr(t, c):
    rate = np.zeros((Ir))
    for r in range(Rnum):
      calpha = np.power(c, alphar[r,:])
      cbeta  = np.power(c, betar[r,:]) 
      rate  += (k_for_r[r] * np.prod(calpha) - k_back_r[r] * np.prod(cbeta))*gammar[r]
    return rate

  # derivative of reaction function for Newton method
  def dRrdc(t, c):
      J = np.zeros((Ir,Ir)) # Jacobian
      for r in range(Rnum):
        for i in range(Ir): # loop over components over density vector
          alpham     = np.copy(alphar[r,:]) # have to copy arrays, otherwise alpha and alpm point to the same array
          betam      = np.copy(betar[r,:])
          alpham[i] -= 1 # lower exponents
          betam[i]  -= 1
          calpham    = np.prod(np.power(c, alpham))
          cbetam     = np.prod(np.power(c, betam))
          dRdci      = (k_for_r[r]* alphar[r,i] * calpham - k_back_r[r] * betar[r,i] * cbetam) *gammar[r]
          J[:,i]    += dRdci   
      return J

  # setup numerical scheme for ODE system
  ode_system_multi = ode(Rr, dRrdc).set_integrator('vode', method='bdf', with_jacobian=True)
  # set intial conditions (t_0 = 0.0)
  ode_system_multi.set_initial_value(c0r, 0.0)

  # time step 
  dt = T*1.0e-3

  print("Solving ODE with multiple reactions...")
  with open(ode_file, "w+") as f:
    while ode_system_multi.successful() and ode_system_multi.t < T:
      ode_system_multi.integrate(ode_system_multi.t+dt)
      write_line(f, ode_system_multi.t, ode_system_multi.y)
  f.close()



### Solve Chemcial Master Equation via Kinetic Monte Carlo (KMC)
if options.solve_cme_multiple:
  ## helper function to compute the transition rates used in kinetic Monte Carlo
  #
  #  B^alpha_V(eta) := V^(1-|alpha|) k_for eta!/(eta-alpha!)
  #  B^beta_V(eta)  := V^(1-|beta|) k_back eta!/(eta-beta!)
  #
  # The CME is given by the generator L_V^*
  # mu'(eta) = (L_V^* mu)(eta) := B^alpha(eta+gamma)mu(eta+gamma)- (B^alpha(eta) + B^beta(eta))mu(eta) + B^beta(eta-gamma)mu(eta-gamma) 
  #
  # (CAVEAT: We do not take care of the case that eta_i-gamma_i < 0)

  jump_vecs = np.zeros((2*Rnum, Ir), dtype=np.int64)
  for r in range(0,2*Rnum,2):
    jump_vecs[r,:] = gammar[r/2]
    jump_vecs[r+1,:] = -gammar[r/2]
  print(jump_vecs)

  def compute_rates_r(e):
    rates = np.zeros((2*Rnum))
    # print(e.dtype)
    # print(gammar)
    for r in range(0, Rnum):
      factor_forw  = np.zeros((Ir))
      factor_backw = np.zeros((Ir))   
      for i in range(Ir):
        factor_forw[i]  = np.prod(range(e[i] + gammar[r,i] - alphar[r,i] + 1, e[i] + gammar[r,i] + 1))   # the last +1 is necessary as range(i0,iN) gives i0,..,iN-1
        factor_backw[i] = np.prod(range(e[i] - gammar[r,i] - betar[r,i]  + 1, e[i] - gammar[r,i] + 1))
      B_forw  = V**(1 - np.sum(alphar[r,:])) * k_for_r[r]  * np.prod(factor_forw)
      B_backw = V**(1 - np.sum(betar[r,:]))  * k_back_r[r] * np.prod(factor_backw)
      rates[2*r]   = B_forw
      rates[2*r+1] = B_backw
    return rates


  print("Solving CME with multiple reactions...")
  # open output file for writing
  with open(options.outfile_file, "w+") as f:
    # simulation loop (stop if final time is achieved or maximal number of steps is reached)
    while t<T and step < max_steps:
      rates = compute_rates_r(etar) # compute rates for transition depending on current state
      # print(rates)
      rate_sum = np.sum(rates)
      tau    = rnd.expovariate(1.0)/(rate_sum) # sample waiting time in state via exponential distribution
      t      = t + tau   # add waiting time to process time
      idx    = np.random.choice(range(2*Rnum), p = rates/rate_sum)
      # print(idx)
      etar  = etar + jump_vecs[idx,:]
      step = step + 1 # increase step counter
      write_line(f, t, etar/V) # write to output file
    f.close()

