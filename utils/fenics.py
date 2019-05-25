"""Solve nonlinear corrections to Darcy flow
"""

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import sys
import os
from time import time


def solve_nonlinear_poisson(input, alpha1, alpha2, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    perm = input
    flux_order = 3
    ngy, ngx = perm.shape[0], perm.shape[1]

    # Create mesh and define function space
    mesh = df.UnitSquareMesh(ngy-1, ngx-1)
    K_CG = df.FunctionSpace(mesh, "Lagrange", 1)
    
    K = df.Function(K_CG)
    ordering = df.dof_to_vertex_map(K_CG)
    K.vector()[:] = perm.flatten(order='C')[ordering]

    def boundary(x, on_boundary):
        return x[0] < df.DOLFIN_EPS or x[0] > 1.0 - df.DOLFIN_EPS
    left = df.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
    right = df.CompiledSubDomain("near(x[0], 1.0) && on_boundary")
    top = df.CompiledSubDomain("near(x[1], 1.0) && on_boundary")
    bottom = df.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    # boundaries.set_all(0)
    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom.mark(boundaries, 4)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Discontinuous Raviart-Thomas
    DRT = df.FiniteElement("DRT", mesh.ufl_cell(), flux_order)
    # Lagrange
    CG = df.FiniteElement("CG", mesh.ufl_cell(), flux_order + 1)
    W = df.FunctionSpace(mesh, DRT * CG)

    ww = df.Function(W)
    sigma, u = df.split(ww)
    tau, v = df.TestFunctions(W)
    dw = df.TrialFunction(W)

    # Define boundary condition
    gamma = df.Expression("1.0 - x[0]", degree=1)
    bc = df.DirichletBC(W.sub(1), gamma, boundary)
    g = df.Constant(0.0)
    f = df.Constant(0.0)

    def flux(sigma):
        s1 = sigma[0]
        s2 = sigma[1]
        nonlinear1 = alpha1 * df.sqrt(K) * df.as_vector([s1 ** 2, s2 ** 2])
        nonlinear2 = alpha2 * K * df.as_vector([s1 ** 3, s2 ** 3])
        nonlinear = nonlinear1 + nonlinear2
        return nonlinear

    F = (df.dot(sigma, tau) + df.dot(K * df.grad(u), tau) + df.dot(flux(sigma), tau)\
        + df.dot(sigma, df.grad(v)) + f * v) * df.dx + g * v * ds(2) + g * v * ds(4)
    J  = df.derivative(F, ww, dw)

    # Compute solution
    problem = df.NonlinearVariationalProblem(F, ww, bc, J)
    solver = df.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-6
    prm['newton_solver']['maximum_iterations'] = 10
    prm['newton_solver']['relaxation_parameter'] = 1.0
    tic = time()
    solver.solve()
    print(f'time taken {time()-tic} seconds')

    (sigma_, u_) = ww.split()

    u_mat = u_.compute_vertex_values(mesh).reshape(64, 64)
    grad_u = sigma_.compute_vertex_values(mesh)
    sigma1 = grad_u[:4096].reshape(64, 64)
    sigma2 = grad_u[4096:].reshape(64, 64)
    output = np.stack((u_mat, sigma1, sigma2))
    print('output shape: {output.shape}')
    return output

