import params
import numpy as np
from dolfinx import mesh, fem
import ufl
from mpi4py import MPI
from dolfinx.io import VTXWriter, XDMFFile
from basix.ufl import mixed_element
from ufl import div, grad, inner, dot
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import matplotlib.pyplot as plt


def on_equator(x):
    return np.isclose(x[2], 0.0, atol=1e-3)  

def initialize_function(functionSpace, initial):
    '''Creates a function on a functionspace with initial conditions'''
    function = fem.Function(functionSpace)
    function.x.array[:] = initial
    return function

def initialize_peturbation(functionSpace, initial):
    '''Creates a function with a small random peturbation around initial'''
    function = fem.Function(functionSpace)
    if initial == 0:
        function.x.array[:] = 0.001 * (np.random.rand(len(function.x.array)) - 0.5)
    else:
        function.x.array[:] = initial + 0.001 * initial * (np.random.rand(len(function.x.array[:])) - 0.5)
    return function

def compute_norm(new, old):
    diff = new - old
    return np.sqrt(fem.assemble_scalar(fem.form(diff * diff * ufl.dx))) / params.dt

def normalize(values):
    values = np.array(values)
    return (values - values.min()) / (values.max() - values.min() + 1e-10)

def compute_normals(mesh: dolfinx.mesh.Mesh):
    cells = mesh.topology.connectivity(2, 0).array
    coords = domain.geometry.x
    element_normals = np.zeros((cells.shape[0] // 3,3))
    for i in range(0, len(cells), 3):
        v0 = coords[cells[i]]
        v1 = coords[cells[i+1]]
        v2 = coords[cells[i+2]]
        n = np.cross(v1 - v0, v2-v0)
        n /= np.linalg.norm(n) + 1e-14
        element_normals[i//3] = n

    vertex_normals = np.zeros_like(coords)
    counts = np.zeros((coords.shape[0], 1))
    for i in range(0,len(cells),3):
        for j in range(3):
            v = cells[i + j]
            vertex_normals[v] += element_normals[i//3]
            counts[v] += 1
    vertex_normals /= counts
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= norms + 1e-14
    return vertex_normals

k_s = 2 
k_b = 0
k_pu = 2
k_pv = -2

def material_velocity(k_s: float, k_b: float, k_pu: float, k_pv:float, u: fem.Function, v: fem.Function, surface_sp:fem.FunctionSpace, timestep: float) -> fem.Function: 
    """
    REPURPOSED FROM SOPHIE'S CODE
    k_s:   tension parameter
    k_b: bending rigidity
    u: vector containing chemical species u (u[0]) and v (u[1])

    returns a scalar function on the boundary of the volume and the next lambda
    V = chemical action - surface tension + volume lambda (+ bending energy)

    ref for curvature stuff and normal vector: https://fenicsproject.discourse.group/t/normal-vector-in-manifold/14160/2
    """
    surface_mesh = surface_sp.mesh
    dM = ufl.dx(domain=surface_mesh)
    n = ufl.CellNormal(surface_mesh)
    x = ufl.SpatialCoordinate(surface_mesh)
    r = x/ufl.sqrt(ufl.dot(x, x))
    sign = ufl.sign(ufl.dot(n, r))
    n_oriented = sign*n
    w= 1000/8000
    

    # set-up for normal vector and curvature computation
    V1 = fem.functionspace(surface_mesh, ("Lagrange", 2, (surface_mesh.geometry.dim,)))
    # V2 = functionspace(surface_mesh, ("Lagrange", 2))
    normal_expr = fem.Expression(n_oriented, V1.element.interpolation_points())
    normal_vec = fem.Function(V1)
    velo_temp = fem.Function(V1)
    velo = fem.Function(V1)

    H_expr = fem.Expression(1/2*div(sign*n), surface_sp.element.interpolation_points())
    grad_sq_expr = fem.Expression(inner(grad(n), grad(n)), surface_sp.element.interpolation_points(), MPI.COMM_WORLD)
    H = fem.Function(surface_sp)
    grad_sq = fem.Function(surface_sp)

    normal_vec.interpolate(normal_expr)
    H.interpolate(H_expr)
    grad_sq.interpolate(grad_sq_expr) 


    # set-up the constants
    k_s = fem.Constant(surface_mesh, dolfinx.default_scalar_type(k_s))  # Constant(domain, default_scalar_type(0.3))
    k_b = fem.Constant(surface_mesh, dolfinx.default_scalar_type(k_b))
    k_pu = fem.Constant(surface_mesh, dolfinx.default_scalar_type(k_pu))
    k_pv = fem.Constant(surface_mesh, dolfinx.default_scalar_type(k_pv))
    
    # the normal velocity as in the paper and the docstring
    velo_temp_expr =fem.Expression(w*timestep*(k_pu * (u) + k_pv * v -k_s*H + k_b * (div(grad(H))-1/2*H**3 + dot(H, grad_sq)))*n, V1.element.interpolation_points()) #div(grad(H))
    velo_temp.interpolate(velo_temp_expr)

    lambda_proxy = fem.assemble_scalar(fem.form(dot(velo_temp, n) * dM)) / fem.assemble_scalar(fem.form(1 * dM))
    velo_expr = fem.Expression(velo_temp - lambda_proxy*n, V1.element.interpolation_points())
    velo.interpolate(velo_expr)
    return velo

def deform_mesh(V: fem.FunctionSpace, u: fem.Function):
    """Deforms the mesh according to a function u:
    Currently adds on u, tested with the normal vector."""
    mesh = V.mesh
    gdim = mesh.geometry.dim
    V_geom = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))

    v_linear = fem.Function(V_geom)
    v_linear.interpolate(u)

    deformation_array = v_linear.x.array.reshape((-1, gdim))
    mesh.geometry.x[:, :gdim] += deformation_array


with XDMFFile(MPI.COMM_WORLD, "out_gmsh/hemisphere.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="hemisphere")

V = fem.functionspace(domain, ("CG", 1,))
W = fem.functionspace(domain, mixed_element([V.ufl_element(), V.ufl_element(), V.ufl_element(), V.ufl_element()]))
D = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))
X0 = params.x1_star
Y0 = params.y1_star

X2_0 = 0# change maybe
Y2_0 = 0  # change maybe
intial_values = [X0, Y0, X2_0, Y2_0]


fdim = domain.topology.dim - 1  # 1 if surface mesh (dim=2)
domain.topology.create_connectivity(fdim, domain.topology.dim)
equator_facets = mesh.locate_entities_boundary(domain, fdim, on_equator)

## Set BCs and ICs
bcs = []
ics = []
for i, iv in enumerate(intial_values):
    sub = W.sub(i) # select i
    bc_value = initialize_function(V, iv)
    dofs = fem.locate_dofs_topological((sub, V), fdim, equator_facets)
    bcs.append(fem.dirichletbc(bc_value, dofs, sub))
    ## set intial condition
    ics.append(initialize_peturbation(V, iv))
 
# Combine into a mixed function
w0 = fem.Function(W)
for sub, ic in zip(w0.split(), ics):
    sub.interpolate(ic)

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

problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs = bcs)
solver = dolfinx.nls.petsc.NewtonSolver(domain.comm, problem)
solver.rtol = 1e-6

u_out = fem.Function(V)
v_out = fem.Function(V)
u2_out = fem.Function(V)
v2_out = fem.Function(V)

writer_u = VTXWriter(domain.comm, "hemisphere_esfem/out_u.bp", [u_out])
writer_v = VTXWriter(domain.comm, "hemisphere_esfem/out_v.bp", [v_out])
writer_u2 = VTXWriter(domain.comm, "hemisphere_esfem/out_u2.bp", [u2_out])
writer_v2 = VTXWriter(domain.comm, "hemisphere_esfem/out_v2.bp", [v2_out])

t = 0
step = 0
L2normU, L2normV, L2normU2, L2normV2 = [], [], [], []
dofs = fem.locate_dofs_topological(D, fdim, equator_facets)
u_d = fem.Function(D)
u_d.interpolate(lambda x: np.stack((0.001*np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))))
u_d.x.array[dofs] = 0.0
deformation_array = u_d.x.array.reshape((-1, D.mesh.geometry.dim))  


while t < params.T:
    t += params.dt
    step += 1

    if t > 120:
        velo = material_velocity(0.5, k_b, params.c_g, -params.c_g, u=w0.sub(2), v=w0.sub(3), surface_sp=V, timestep=params.dt)
        deform_mesh(V, velo);
        problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs = bcs)
        solver = dolfinx.nls.petsc.NewtonSolver(domain.comm, problem)
        solver.rtol = 1e-6

    n, converged = solver.solve(w)
    if not converged:
        print(f"Step {step}: Newton solver,   failed to converge")
        break

    # Split updated solution
    u_sol, v_sol, u2_sol, v2_sol = w.split()
    for out, sol in zip([u_out, v_out, u2_out, v2_out], w.split()):
        out.interpolate(sol)
    

    # Split previous solution for norms
    u_old, v_old, u2_old, v2_old = w0.split()
    
    norm_u = compute_norm(u_sol, u_old)
    norm_v = compute_norm(v_sol, v_old)
    norm_u2 = compute_norm(u2_sol, u2_old)
    norm_v2 = compute_norm(v2_sol, v2_old)

    L2normU.append(norm_u)
    L2normV.append(norm_v)
    L2normU2.append(norm_u2)
    L2normV2.append(norm_v2)
    for l2n, n in zip([L2normU, L2normV, L2normU2, L2normV2], [norm_u, norm_v, norm_u2, norm_v2]):
        l2n.append(n)
    if step % 50 == 0:
        print(f"Step {step}, t = {t:.3f}, Newton iterations = {n}")
        for writer in [writer_u, writer_v, writer_u2, writer_v2]:
            writer.write(t)
        
    
    # Update previous solution for next step
    w0.x.array[:] = w.x.array


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
