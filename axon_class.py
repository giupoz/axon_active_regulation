import dolfin as df
import numpy as np
import ufl
import scipy.io
import os
import shutil


class AxonProblem:
    def __init__(
        self, nx=500, Ri=0.9, Ro=1, muc=1, mua=1, Kc=100, Ka=1, Bt=-1.3, Bz=-1.3, tau=1
    ):
        # check if output folder exists. If this is the case delete it.
        if os.path.exists("output") and os.path.isdir("output"):
            shutil.rmtree("output")

        self.nx = nx
        self.Ri = Ri
        self.Ro = Ro
        self.muc = df.Constant(muc)
        self.mua = df.Constant(mua)
        self.Kc = df.Constant(Kc)
        self.Ka = df.Constant(Ka)
        self.Bt = df.Constant(Bt)
        self.Bz = df.Constant(Bz)
        self.tau = df.Constant(tau)
        self.lambdaZ = df.Constant(1)

        self.lambda_Z_export = np.array([])
        self.lambda_a_Theta_export = np.array([])
        self.lambda_a_Z_export = np.array([])
        df.set_log_level(40)

        self.mesh = df.IntervalMesh(nx, 0, Ro)
        self.X = df.SpatialCoordinate(self.mesh)

        # Function Space:
        self.V = df.FunctionSpace(self.mesh, "CG", 1)
        self.DG = df.FunctionSpace(
            self.mesh, "DG", 0
        )

        # Trial functions:
        # u -> radial displacement
        # lat -> lambda_{a\Theta}
        # laz -> lambda_{az}
        
        self.u = df.Function(self.V)
        self.u.rename("u", "u")

        self.lat = df.Function(self.DG)
        self.laz = df.Function(self.DG)
        self.lat.rename("lambda_a_theta", "lambda_a_theta")
        self.laz.rename("lambda_a_z", "lambda_a_z")
        self.lat.vector()[:] = 1
        self.laz.vector()[:] = 1

        self.alpha = df.Function(self.DG)
        self.beta = df.Function(self.DG)
        self.alpha.rename("alpha", "alpha")
        self.beta.rename("beta", "beta")

        # Boundary conditions: r(0) = 0
        def center(x, on_boundary):
            tol = 1e-6
            return on_boundary and (df.near(x[0], 0, tol))

        self.bc = df.DirichletBC(self.V, df.Constant(0), center)

    def monitor(
        self,
        u,
        lat,
        laz,
        lambdaZ,
        output_file,
        export_data,
        label,
        t,
        alpha=None,
        beta=None,
    ):
        output_file.write(u, t)
        output_file.write(lat, t)
        output_file.write(laz, t)

        # Cauchy stress tensor
        W = df.TensorFunctionSpace(self.mesh, "DG", 0, shape=(3, 3))
        cauchy_proj = df.project(self.cauchy, W)
        cauchy_proj.rename("cauchy", "cauchy")

        # Calcolo media lat e laz
        select_cortex = df.conditional(self.X[0] > self.Ri, 1, 0)
        R = df.SpatialCoordinate(self.mesh)[0]
        Jgeo = 2 * df.DOLFIN_PI * R
        area = df.assemble(select_cortex * Jgeo * df.dx)
        lat_mean = df.assemble(select_cortex * lat * Jgeo * df.dx) / area
        laz_mean = df.assemble(select_cortex * laz * Jgeo * df.dx) / area

        # clacolo componenti di sforzo su Ri
        Vrr = df.as_tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        Vtt = df.as_tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        Vzz = df.as_tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        Trr = df.inner(cauchy_proj, Vrr)
        Ttt = df.inner(cauchy_proj, Vtt)
        Tzz = df.inner(cauchy_proj, Vzz)

        if alpha is not None:
            output_file.write(alpha, t)
        if beta is not None:
            output_file.write(beta, t)

        output_file.write(cauchy_proj, t)

        export_data["lambda_Z"] = np.append(export_data["lambda_Z"], float(lambdaZ))
        export_data["lambda_a_Theta_final"] = np.append(
            export_data["lambda_a_Theta_final"], lat([self.Ro])
        )
        export_data["lambda_a_Z_final"] = np.append(
            export_data["lambda_a_Z_final"], laz([self.Ro])
        )
        export_data["r"] = np.append(export_data["r"], u([self.Ro]) + self.Ro)
        export_data["lambda_a_Theta_mean"] = np.append(
            export_data["lambda_a_Theta_mean"], lat_mean
        )
        export_data["lambda_a_Z_mean"] = np.append(
            export_data["lambda_a_Z_mean"], laz_mean
        )
        export_data["Trr"] = np.append(
            export_data["Trr"], Trr([self.Ri + self.Ro / self.nx])
        )
        export_data["Ttt"] = np.append(
            export_data["Ttt"], Ttt([self.Ri + self.Ro / self.nx])
        )
        export_data["Tzz"] = np.append(
            export_data["Tzz"], Tzz([self.Ri + self.Ro / self.nx])
        )



        scipy.io.savemat("output/data_" + label + ".mat", export_data)

    def material_properties(self):
        # material properties of the axon: shear and bulk modulus

        properties = {
            "mu": df.conditional(self.X[0] > self.Ri, self.muc, self.mua),
            "K": df.conditional(self.X[0] > self.Ri, self.Kc, self.Ka),
        }

        return properties

    def kinematics_features(self):
        # kinematics features of the problem

        mat_prop = self.material_properties()
        mu = mat_prop["mu"]
        K = mat_prop["K"]

        r = self.X[0] + self.u
        R = self.X[0]

        Fbar = df.as_tensor([[r.dx(0), 0, 0], [0, r / R, 0], [0, 0, self.lambdaZ]])
        F = ufl.variable(Fbar)
        Fa = df.as_tensor(
            [[1 / (self.lat * self.laz), 0, 0], [0, self.lat, 0], [0, 0, self.laz]]
        )
        Fe = F * df.inv(Fa)
        I1e = df.tr(Fe.T * Fe)
        Je = df.det(Fe)
        Jgeo = 2 * df.DOLFIN_PI * R  # Integration in cylindrical coordinates


        Id = df.Identity(3)
        It = df.as_tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        Iz = df.as_tensor([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])

        M = (1 - self.alpha) * (1 - self.beta) * ( mu * (Fe.T * Fe) + (K * df.ln(Je) - mu) * Id )  
        Mt = df.inner(M, It)
        Mz = df.inner(M, Iz)

        kin_features = {
            "F": F,
            "Fe": Fe,
            "I1e": I1e,
            "Je": Je,
            "Jgeo": Jgeo,
            "Mt": Mt,
            "Mz": Mz,
        }

        return kin_features

    def strainenergy(self, mu, K, kin_features):
        # Strain energy density computation
        F = kin_features["F"]
        I1e = kin_features["I1e"]
        Je = kin_features["Je"]
        Jgeo = kin_features["Jgeo"]

        psi = mu / 2 * (I1e - 3 - 2 * df.ln(Je)) + K / 2 * df.ln(Je) ** 2
        energy = Jgeo * (1 - self.alpha) * (1 - self.beta) * psi * df.dx
        piola = ufl.diff((1 - self.alpha) * (1 - self.beta) * psi, F)
        self.cauchy = 1 / df.det(F) * piola * F.T

        return energy

    def time_increment_active_strain(self, Mt, Mz, dt):

        drive_t = df.conditional(
            self.X[0] > self.Ri, (1 - self.beta)**2 * self.Bt + Mt, df.Constant(0)
        )
        drive_z = df.conditional(
            self.X[0] > self.Ri, (1 - self.beta)**2 * self.Bz + Mz, df.Constant(0)
        )
        lat_new = df.project(
            self.lat + dt / (self.muc * self.tau) * drive_t * self.lat, self.DG
        )
        laz_new = df.project(
            self.laz + dt / (self.muc * self.tau) * drive_z * self.laz, self.DG
        )
        self.lat.assign(lat_new)

        # assign upper bound to lat
        latvals = (
            lat_new.vector().get_local()
        )
        latvals[latvals > 1.0] = 1.0
        lat_new.vector().set_local(
            latvals
        )
        self.lat.assign(lat_new)


        # assign upper bound to laz
        lazvals = (
            laz_new.vector().get_local()
        )
        lazvals[lazvals > 1.0] = 1.0
        laz_new.vector().set_local(
            lazvals
        )
        self.laz.assign(laz_new)

    def find_equilibrium(self, T, n_step):
        export_data = {
            "lambda_Z": np.array([]),
            "lambda_a_Theta_final": np.array([]),
            "lambda_a_Z_final": np.array([]),
            "r": np.array([]),
            "lambda_a_Theta_mean": np.array([]),
            "lambda_a_Z_mean": np.array([]),
            "Trr": np.array([]),
            "Ttt": np.array([]),
            "Tzz": np.array([]),
        }

        # instantiate the material properties of the body
        mat_properties = self.material_properties()
        mu = mat_properties["mu"]
        K = mat_properties["K"]

        # instantiate the kinematic features of the problem
        kin_features = self.kinematics_features()
        Fe = kin_features["Fe"]
        Je = kin_features["Je"]
        Mt = kin_features["Mt"]
        Mz = kin_features["Mz"]

        # compute the strain energy function
        energy = self.strainenergy(mu, K, kin_features)

        # variational problem
        FF = df.derivative(energy, self.u, df.TestFunction(self.V))
        Jacobian = df.derivative(FF, self.u, df.TrialFunction(self.V))

        # Time-Loop parameters initialization
        t = 0
        dt = T / n_step

        XDMF_options = {
            "flush_output": True,
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
        }
        output_file = df.XDMFFile("output/results_equilibrium.xdmf")
        output_file.parameters.update(XDMF_options)

        while t < T:
            df.solve(FF == 0, self.u, self.bc, J=Jacobian)

            self.monitor(
                self.u,
                self.lat,
                self.laz,
                self.lambdaZ,
                output_file,
                export_data,
                "equilibrium",
                t,
            )
            self.time_increment_active_strain(Mt, Mz, dt)

            t += dt

    def apply_noco(self, T, n_step, tau_alpha_value, alpha_bar_value):
        time = df.Constant(0)
        alpha_bar = df.Constant(alpha_bar_value)
        tau_alpha = df.Constant(tau_alpha_value)
        alphaexpr = df.conditional(
            self.X[0] > self.Ri, 0, alpha_bar * (1 - df.exp(-time / tau_alpha))
        )
        export_data = {
            "lambda_Z": np.array([]),
            "lambda_a_Theta_final": np.array([]),
            "lambda_a_Z_final": np.array([]),
            "r": np.array([]),
            "lambda_a_Theta_mean": np.array([]),
            "lambda_a_Z_mean": np.array([]),
            "Trr": np.array([]),
            "Ttt": np.array([]),
            "Tzz": np.array([]),
        }

        # instantiate the material properties of the body
        mat_properties = self.material_properties()
        mu = mat_properties["mu"]
        K = mat_properties["K"]

        # instantiate the kinematic features of the problem
        kin_features = self.kinematics_features()
        Fe = kin_features["Fe"]
        Je = kin_features["Je"]
        Mt = kin_features["Mt"]
        Mz = kin_features["Mz"]

        # compute the strain energy function
        energy = self.strainenergy(mu, K, kin_features)

        # variational problem
        FF = df.derivative(energy, self.u, df.TestFunction(self.V))
        Jacobian = df.derivative(FF, self.u, df.TrialFunction(self.V))

        # Time-Loop parameters initialization
        t = 0
        dt = T / n_step

        XDMF_options = {
            "flush_output": True,
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
        }
        output_file = df.XDMFFile("output/results_noco.xdmf")
        output_file.parameters.update(XDMF_options)

        while t < T:
            time.assign(t)
            self.alpha.assign(df.project(alphaexpr, self.DG))
            df.solve(FF == 0, self.u, self.bc, J=Jacobian)

            self.monitor(
                self.u,
                self.lat,
                self.laz,
                self.lambdaZ,
                output_file,
                export_data,
                "noco",
                t,
                alpha=self.alpha,
            )
            self.time_increment_active_strain(Mt, Mz, dt)

            t += dt

    def apply_axial_stretch(
        self,
        T_0,
        T,
        n_step,
        new_lambdaZ,
        gamma_bar_value,
        alpha_bar_value,
        tau_alpha_value,
        beta_bar_value,
        tau_beta_value,
    ):
        time = df.Constant(0)
        self.lambdaZ.assign(new_lambdaZ)

        # nocodazole
        alpha_bar = df.Constant(alpha_bar_value)
        tau_alpha = df.Constant(tau_alpha_value)
        gamma_bar = df.Constant(gamma_bar_value)
        alphaexpr = df.conditional(
            self.X[0] > self.Ri,
            0,
            gamma_bar + alpha_bar * (1 - df.exp(-(time + T_0) / tau_alpha)),
        )

        # cytochalasin D
        beta_bar = df.Constant(beta_bar_value)
        tau_beta = df.Constant(tau_beta_value)
        betaexpr = df.conditional(
            self.X[0] < self.Ri, 0, beta_bar * (1 - df.exp(-(time + T_0) / tau_beta))
        )

        export_data = {
            "lambda_Z": np.array([]),
            "lambda_a_Theta_final": np.array([]),
            "lambda_a_Z_final": np.array([]),
            "r": np.array([]),
            "lambda_a_Theta_mean": np.array([]),
            "lambda_a_Z_mean": np.array([]),
            "Trr": np.array([]),
            "Ttt": np.array([]),
            "Tzz": np.array([]),
        }

        # instantiate the material properties of the body
        mat_properties = self.material_properties()
        mu = mat_properties["mu"]
        K = mat_properties["K"]

        # instantiate the kinematic features of the problem
        kin_features = self.kinematics_features()
        Fe = kin_features["Fe"]
        Je = kin_features["Je"]
        Mt = kin_features["Mt"]
        Mz = kin_features["Mz"]

        # compute the strain energy function
        energy = self.strainenergy(mu, K, kin_features)

        # variational problem
        FF = df.derivative(energy, self.u, df.TestFunction(self.V))
        Jacobian = df.derivative(FF, self.u, df.TrialFunction(self.V))

        # Time-Loop parameters initialization
        t = 0
        dt = T / n_step

        XDMF_options = {
            "flush_output": True,
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
        }
        output_file = df.XDMFFile("output/results_axial_stretch.xdmf")
        output_file.parameters.update(XDMF_options)

        while t < T:
            time.assign(t)
            self.alpha.assign(df.project(alphaexpr, self.DG))
            self.beta.assign(df.project(betaexpr, self.DG))
            df.solve(FF == 0, self.u, self.bc, J=Jacobian)

            self.monitor(
                self.u,
                self.lat,
                self.laz,
                self.lambdaZ,
                output_file,
                export_data,
                "axial_stretch",
                t,
                alpha=self.alpha,
                beta=self.beta,
            )
            self.time_increment_active_strain(Mt, Mz, dt)

            t += dt

    def apply_cytoD(self, T, n_step, tau_beta_value, beta_bar_value):
        time = df.Constant(0)

        beta_bar = df.Constant(beta_bar_value)
        tau_beta = df.Constant(tau_beta_value)
        betaexpr = df.conditional(
            self.X[0] < self.Ri, 0, beta_bar * (1 - df.exp(-time / tau_beta))
        )

        export_data = {
            "lambda_Z": np.array([]),
            "lambda_a_Theta_final": np.array([]),
            "lambda_a_Z_final": np.array([]),
            "r": np.array([]),
            "lambda_a_Theta_mean": np.array([]),
            "lambda_a_Z_mean": np.array([]),
            "Trr": np.array([]),
            "Ttt": np.array([]),
            "Tzz": np.array([]),
        }

        # instantiate the material properties of the body
        mat_properties = self.material_properties()
        mu = mat_properties["mu"]
        K = mat_properties["K"]

        # instantiate the kinematic features of the problem
        kin_features = self.kinematics_features()
        Fe = kin_features["Fe"]
        Je = kin_features["Je"]
        Mt = kin_features["Mt"]
        Mz = kin_features["Mz"]

        # compute the strain energy function
        energy = self.strainenergy(mu, K, kin_features)

        # variational problem
        FF = df.derivative(energy, self.u, df.TestFunction(self.V))
        Jacobian = df.derivative(FF, self.u, df.TrialFunction(self.V))

        # Time-Loop parameters initialization
        t = 0
        dt = T / n_step

        XDMF_options = {
            "flush_output": True,
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
        }
        output_file = df.XDMFFile("output/results_cytoD.xdmf")
        output_file.parameters.update(XDMF_options)

        while t < T:
            time.assign(t)
            self.beta.assign(df.project(betaexpr, self.DG))
            df.solve(FF == 0, self.u, self.bc, J=Jacobian)

            self.monitor(
                self.u,
                self.lat,
                self.laz,
                self.lambdaZ,
                output_file,
                export_data,
                "cytoD",
                t,
                beta=self.beta,
            )
            self.time_increment_active_strain(Mt, Mz, dt)

            t += dt
