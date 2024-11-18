import opensim as osim
import numpy as np
import scipy.special

def gravitational_torque(model, state):
    """
    Calculates the gravity torque for each joint in an OpenSim model.
    Since this function does find Tau, st tau = -G -A (in static) with A the applied loads,
    if there are no applied loads then tau = -G ie G = -tau.
    --> The model should have no load applied!!!!.
    """
    # Create InverseDynamicsSolver
    solver = osim.InverseDynamicsSolver(model)    

    # Solve once for all coordinates
    gen_forces = solver.solve(state)
    return -gen_forces.to_numpy()

def station_jacobian(model, state, X, body_X):
    J = osim.Matrix()
    matter: osim.SimbodyMatterSubsystem = model.getMatterSubsystem()
    matter.calcStationJacobian(state, body_X.getMobilizedBodyIndex(), osim.Vec3(X), J)
    return J.to_numpy()

def frame_jacobian(model, state, body_X):
    J = osim.Matrix()
    matter: osim.SimbodyMatterSubsystem = model.getMatterSubsystem()
    matter.calcFrameJacobian(state, body_X.getMobilizedBodyIndex(), osim.Vec3(0), J)
    return J.to_numpy()[:3, :]

def lever_arm_matrix(state, muscles, coordinates):
    """This is already given as -L.T"""
    N_osim = [[x.computeMomentArm(state, c) for c in coordinates] for x in muscles]
    return np.array(N_osim).T


def nb_k_face_zonotope(k, n, m):
    """Returns the number of k-dimensional faces of a zonotope in R^n with m generators in general position."""
    return 2 * scipy.special.comb(m, k) * sum([scipy.special.comb(m-k-1, i) for i in range(0, n-k)])
