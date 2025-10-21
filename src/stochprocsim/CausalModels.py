import numpy as np
from .Libraries.OpticsLib import getUtot
from .Models.Unitaries import *

flipped = True

class CausalModel():
    def __init__(self, U, states:list, angles=None, name=None):
        self.U = U
        self.U_mat = U 
        if angles is not None:
            self.U_opt = getUtot(angles)
        self.states = list(states)
        self._state_map = {f"s{i}": s for i, s in enumerate(states)}
        self.name = name
        self.angles = angles

    def set_U_optics(self):
        self.U = self.U_opt

    def set_U_mat(self):
        self.U = self.U_mat

    def __getitem__(self, idx):
        return self.states[idx]
    
    def __getattr__(self, name):
        if name in self._state_map:
            return self._state_map[name]
        raise AttributeError(f'No state named {name}')
    
    def __len__(self):
        return len(self.states)-1

def reorder_matrix(U: sp.Matrix, row_order: list[int], col_order: list[int]) -> sp.Matrix:
    """
    Return a reordered copy of U with rows and columns rearranged according to
    row_order and col_order (0-indexed).

    Example:
        new_U = reorder_matrix(U, [1, 3, 0, 2], [1, 3, 0, 2])
    """
    return U.extract(row_order, col_order)

def reorder_states(states: list[sp.Matrix], order: list[int]) -> list[sp.Matrix]:
    """
    Return a reordered copy of a list of state vectors (sp.Matrix objects).

    """
    reordered_matrices = []
    for state in states:
        reordered = [state[i] for i in order]
        reordered_matrices.append(sp.Matrix(reordered))
    return reordered_matrices

#region Unitary Angles 
Φm1, Φm2, Φm3 = sp.symbols('Φm1 Φm2 Φm3')
θh1, θq1, θh2, θq2, θhin2, θqin2, θhf1, θqf1, θhf2, θqf2, pipi = sp.symbols('θh1 θq1 θh2 θq2 θhin2 θqin2 θhf1 θqf1 θhf2 θqf2 pipi')

U_3_angles = {
    "θh1": np.rad2deg(1.57158),
    "θq1": np.rad2deg(0.0017101),
    "θh2": np.rad2deg(3.99667),
    "θq2": np.rad2deg(1.92368),
    "θhin2": np.rad2deg(1.58668),
    "θqin2": np.rad2deg(4.79562),
    "θhf1": np.rad2deg(4.31667),
    "θqf1": np.rad2deg(4.55094),
    "θhf2": np.rad2deg(5.83508),
    "θqf2": np.rad2deg(1.38684),
    "pipi": 2.03774,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

U_4_angles = {
    "θh1": np.rad2deg(3.144),
    "θq1": np.rad2deg(0.0),
    "θh2": np.rad2deg(0.845031),
    "θq2": np.rad2deg(2.74251),
    "θhin2": np.rad2deg(0.710524),
    "θqin2": np.rad2deg(0.0),
    "θhf1": np.rad2deg(2.54209),
    "θqf1": np.rad2deg(4.49778),
    "θhf2": np.rad2deg(5.83761),
    "θqf2": np.rad2deg(3.40202),
    "pipi": 0.270094,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74
}

U_5_angles = {
    "θh1": np.rad2deg(1.5707),
    "θq1": np.rad2deg(1.57092),
    "θh2": np.rad2deg(2.48831),
    "θq2": np.rad2deg(4.44748),
    "θhin2": np.rad2deg(2.751),
    "θqin2": np.rad2deg(1.2807),
    "θhf1": np.rad2deg(4.1716),
    "θqf1": np.rad2deg(2.91341),
    "θhf2": np.rad2deg(2.83674),
    "θqf2": np.rad2deg(4.93841),
    "pipi": 3.97891,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

U_6_angles = {
    "θh1": np.rad2deg(1.57076),
    "θq1": np.rad2deg(4.71234),
    "θh2": np.rad2deg(5.99134),
    "θq2": np.rad2deg(3.60292),
    "θhin2": np.rad2deg(3.1633),
    "θqin2": np.rad2deg(2.21049),
    "θhf1": np.rad2deg(4.61802),
    "θqf1": np.rad2deg(2.06573),
    "θhf2": np.rad2deg(5.65754),
    "θqf2": np.rad2deg(5.72802),
    "pipi": 0.192194,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74
}
#endregion

if flipped == True:
    # Models
    CS_3 = CausalModel(U=reorder_matrix(U_3, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_3_states, [1, 3, 0, 2]), angles=U_3_angles, name='U_3')
    CS_4 = CausalModel(U=reorder_matrix(U_4, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_4_states, [1, 3, 0, 2]), angles=U_4_angles, name='U_4')
    CS_5 = CausalModel(U=reorder_matrix(U_5, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_5_states, [1, 3, 0, 2]), angles=U_5_angles, name='U_5')
    CS_6 = CausalModel(U=reorder_matrix(U_6, [1, 3, 0, 2], [1, 3, 0, 2]), 
                        states=reorder_states(U_6_states, [1, 3, 0, 2]), angles=U_6_angles, name='U_6')

else:
    # Models
    CS_3 = CausalModel(U=U_3, states=U_3_states, angles=U_3_angles, name='U_3')
    CS_4 = CausalModel(U=U_4, states=U_4_states, angles=U_4_angles, name='U_4')
    CS_5 = CausalModel(U=U_5, states=U_5_states, angles=U_5_angles, name='U_5')
    CS_6 = CausalModel(U=U_6, states=U_6_states, angles=U_6_angles, name='U_6')

Causal_Models = {
    3: CS_3,
    4: CS_4,
    5: CS_5,
    6: CS_6
}

