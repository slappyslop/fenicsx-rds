import params
import numpy as np
from dolfinx import mesh, fem
import ufl
from mpi4py import MPI
from dolfinx.io import VTKFile, XDMFFile
from basix.ufl import mixed_element
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import matplotlib.pyplot as plt


def on_equator(x):
    return np.isclose(x[2], 0.0, atol=1e-3)  

def initialize_function(functionSpace, initial):
    function = fem.Function(functionSpace)
    function.x.array[:] = initial
    return function

def initialize_peturbation(functionSpace, initial):
    function = fem.Function(functionSpace)
    if initial == 0:
        function.x.array[:] = 0.01 * (np.random.rand(len(function.x.array)) - 0.5)
    else:
        function.x.array[:] = initial + 0.01 * initial * (np.random.rand(len(function.x.array[:])) - 0.5)
    return function


with XDMFFile(MPI.COMM_WORLD, "out_gmsh/hemisphere.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="hemisphere")

V = fem.functionspace(domain, ("CG", 1))
W = fem.functionspace(domain, mixed_element([V.ufl_element(), V.ufl_element(), V.ufl_element(), V.ufl_element()]))
X0 = params.x1_star
Y0 = params.y1_star
X2_0 = 0 # change maybe
Y2_0 = 0  # change maybe



fdim = domain.topology.dim - 1  # 1 if surface mesh (dim=2)
domain.topology.create_connectivity(fdim, domain.topology.dim)
equator_facets = mesh.locate_entities_boundary(domain, fdim, on_equator)
# bc_value_u = fem.Function(V)
# bc_value_u.x.array[:] = X0

# bc_value_v = fem.Function(V)
# bc_value_v.x.array[:] = Y0

# bc_value_u2 = fem.Function(V)
# bc_value_u2.x.array[:] = X2_0

# bc_value_v2 = fem.Function(V)
# bc_value_v2.x.array[:] = Y2_0

bc_value_u = initialize_function(V, X0)
bc_value_v= initialize_function(V, Y0)
bc_value_u2 = initialize_function(V, X2_0)
bc_value_v2 = initialize_function(V, Y2_0)


dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, equator_facets)
dofs_v = fem.locate_dofs_topological((W.sub(1), V), fdim, equator_facets)
dofs_u2 = fem.locate_dofs_topological((W.sub(2), V), fdim, equator_facets)
dofs_v2 = fem.locate_dofs_topological((W.sub(3), V), fdim, equator_facets)

# Define DirichletBC objects
bc_u = fem.dirichletbc(bc_value_u, dofs_u, W.sub(0))
bc_v = fem.dirichletbc(bc_value_v, dofs_v, W.sub(1))
bc_u2 = fem.dirichletbc(bc_value_u2, dofs_u2, W.sub(2))
bc_v2 = fem.dirichletbc(bc_value_v2, dofs_v2, W.sub(3))


## Random peturbation
# u0 = fem.Function(V)
# v0 = fem.Function(V)
# u2_0 = fem.Function(V)
# v2_0 = fem.Function(V)
# u0.x.array[:] = X0 + 0.01 * X0 * (np.random.rand(len(u0.x.array)) - 0.5)
# v0.x.array[:] = Y0 + 0.01 * Y0 * (np.random.rand(len(v0.x.array)) - 0.5)
# if (X2_0 == 0):
#     u2_0.x.array[:] = 0.01 *(np.random.rand(len(u2_0.x.array)) - 0.5)
# else:
#     u2_0.x.array[:] = X2_0 + 0.01 * X2_0 *(np.random.rand(len(u2_0.x.array)) - 0.5)

# if (Y2_0 == 0):
#     v2_0.x.array[:] =  0.01 * (np.random.rand(len(v2_0.x.array)) - 0.5)
# else:
#     v2_0.x.array[:] = Y2_0 + 0.01 * Y2_0 * (np.random.rand(len(v2_0.x.array)) - 0.5)

u0 = initialize_peturbation(V, X0)
v0 = initialize_peturbation(V, Y0)
u2_0 = initialize_peturbation(V, X2_0)
v2_0 = initialize_peturbation(V, Y2_0)




# Combine into a mixed function
w0 = fem.Function(W)
w0.sub(0).interpolate(u0)
w0.sub(1).interpolate(v0)
w0.sub(2).interpolate(u2_0)
w0.sub(3).interpolate(v2_0)
w = fem.Function(W)
w.x.array[:] = w0.x.array
w  = initialize_function(W, w0.x.array)

# Trial, test functions
u, v, u2, v2 = ufl.split(w)
u_n, v_n, u2_n, v2_n = ufl.split(w0)
phi_u, phi_v, phi_u2, phi_v2 = ufl.TestFunctions(W)

f1 = (1/params.dt) *ufl.inner(u-u_n, phi_u) * ufl.dx + params.D_X1 * ufl.inner(ufl.grad(u), ufl.grad(phi_u)) * ufl.dx - params.f_1(u_n, v_n) *phi_u * ufl.dx
f2 =(1/params.dt) * ufl.inner(v-v_n, phi_v) * ufl.dx + params.D_Y1 * ufl.inner(ufl.grad(v), ufl.grad(phi_v)) * ufl.dx - params.g_1(u_n, v_n) *phi_v * ufl.dx
f3 = (1/params.dt) * ufl.inner(u2 - u2_n, phi_u2) * ufl.dx \
     + params.D_X2 * ufl.inner(ufl.grad(u2), ufl.grad(phi_u2)) * ufl.dx \
     - params.f_2(u_n, v_n, u2_n, v2_n) * phi_u2 * ufl.dx
 
f4 = (1/params.dt) * ufl.inner(v2 - v2_n, phi_v2) * ufl.dx \
     + params.D_Y2 * ufl.inner(ufl.grad(v2), ufl.grad(phi_v2)) * ufl.dx \
     - params.g_2(u_n, v_n, u2_n, v2_n) * phi_v2 * ufl.dx
F = f1 + f2 + f3 + f4

problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs = [bc_u, bc_v, bc_u2, bc_v2])
solver = dolfinx.nls.petsc.NewtonSolver(domain.comm, problem)
solver.rtol = 1e-6
 
vtk_u = VTKFile(domain.comm, "rewrite_hemisphere_b/bruss_u_implicit.pvd", "w")
vtk_v = VTKFile(domain.comm, "rewrite_hemisphere_b/bruss_v_implicit.pvd", "w")
vtk_u2 = VTKFile(domain.comm, "rewrite_hemisphere_b/gm_u2_implicit.pvd", "w")
vtk_v2 = VTKFile(domain.comm, "rewrite_hemisphere_b/gm_v2_implicit.pvd", "w")

u_out = fem.Function(V)
v_out = fem.Function(V)
u2_out = fem.Function(V)
v2_out = fem.Function(V)

t = 0
step = 0
L2normU, L2normV, L2normU2, L2normV2 = [], [], [], []

while t < params.T:
    t += params.dt
    step += 1
    

    n, converged = solver.solve(w)
    if not converged:
        print(f"Step {step}: Newton solver failed to converge")
        break


    # Split updated solution
    u_sol, v_sol, u2_sol, v2_sol = w.split()
    u_out.interpolate(u_sol)
    v_out.interpolate(v_sol)
    u2_out.interpolate(u2_sol)
    v2_out.interpolate(v2_sol)


    # Split previous solution for norms
    u_old, v_old, u2_old, v2_old = w0.split()

    # Compute L2 norms
    def compute_norm(new, old):
        diff = new - old
        return np.sqrt(fem.assemble_scalar(fem.form(diff * diff * ufl.dx))) / params.dt

    norm_u = compute_norm(u_sol, u_old)
    norm_v = compute_norm(v_sol, v_old)
    norm_u2 = compute_norm(u2_sol, u2_old)
    norm_v2 = compute_norm(v2_sol, v2_old)

    L2normU.append(norm_u)
    L2normV.append(norm_v)
    L2normU2.append(norm_u2)
    L2normV2.append(norm_v2)

   
    if step % 50 == 0:
        print(f"Step {step}, t = {t:.3f}, Newton iterations = {n}")
        vtk_u.write_function(u_out, t)
        vtk_v.write_function(v_out, t)
        vtk_u2.write_function(u2_out, t)
        vtk_v2.write_function(v2_out, t)

    # Update previous solution for next step
    w0.x.array[:] = w.x.array

def normalize(values):
    values = np.array(values)
    return (values - values.min()) / (values.max() - values.min() + 1e-10)

L2normU_normalized = normalize(L2normU)
L2normV_normalized = normalize(L2normV)
L2normU2_normalized = normalize(L2normU2)
L2normV2_normalized = normalize(L2normV2)

T_range = np.linspace(0, params.T, len(L2normU))

plt.figure(figsize=(12, 6))
plt.plot(T_range, L2normU_normalized, label="U1", marker='o')
plt.plot(T_range, L2normV_normalized, label="V1", marker='s')
plt.plot(T_range, L2normU2_normalized, label="U2", marker='^')
plt.plot(T_range, L2normV2_normalized, label="V2", marker='x')

plt.xlabel("Time", fontsize=14)
plt.ylabel("Normalized L2 Norm", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("norms_all_fields.png")
