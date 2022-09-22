from axon_class import AxonProblem
from subprocess import Popen

problem = AxonProblem(
    Ri=0.8, Ro=1, muc=1, mua=1, Kc=100, Bt=-1.6, Bz=-1.6, Ka=0.1, tau=700
)

T = 3600
n_step = 200
lambdaZ = 1.2

# strain damage
gamma_bar = 0.0

# nocodazole
alpha_bar = 0.0

# cytochalasin D
beta_bar = 0.0

stretch = True
noco = True
cytod = False


problem.find_equilibrium(2 * 3600, 200)

tau_alpha = 20.0 * 60
tau_beta = 10.0 * 60


if noco:
    alpha_bar = 0.65
    problem.apply_noco(T, n_step, tau_alpha, alpha_bar)

if cytod:
    beta_bar = 0.9
    problem.apply_cytoD(T, n_step, tau_beta, beta_bar)

if stretch:
    if noco:
        gamma_bar = 0.1
    elif cytod:
        gamma_bar = 0.75
    else:
        gamma_bar = 0.75
    problem.apply_axial_stretch(
        T, T, n_step, lambdaZ, gamma_bar, alpha_bar, tau_alpha, beta_bar, tau_beta
    )
