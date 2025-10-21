import numpy as np

from .CausalModels import *
from .Models.TransitionModel import QuantumTransitionModel, ExactTransitionModel, TransitionModel

class Simulator():
    def __init__(self, transition_model: TransitionModel, nphotons:int=None , mrepetitions:int=None):
        self.transition_model = transition_model
        self.name = transition_model.name
        self.N = len(transition_model)
        self.model = transition_model.model
        self.nphotons = nphotons 
        self.mrepetitions = mrepetitions 
        self.outputs = []

    def run(self):
        self.outputs = []
        for i in range(M):
            s = self.transition_model.sample_output(self.model, self.nphotons)  
            self.outputs.append(s.to_dict(orient="records"))  
            if i % 1000 == 0: print(i)

    def get_probabilities(self):
        """
        Calculate probability of 1 (a) and 0 (b) at each step 
        """
        for j in range(self.N):
            a, b = self.transition_model.get_output_probabilities(self.N, j)
            print(f'a:{a:.2f}, b:{b:.2f}')

    def get_quantities(self):
        """
        Calculate % of total samples which will end up in each output bin 
        """
        a_prod = 1
        res = []
        for j in range(self.N):
            a, b = self.transition_model.get_output_probabilities(self.N, j)
            res.append(b * a_prod)
            # print(f"s{j} p1:{(b * a_prod):0.2f}")
            a_prod *= a
        return res

    def save(self):
        out_file = f"Data/{self.name}_{self.model.name}_M_{self.mrepetitions}_photons_{self.nphotons}.npz"
        np.savez(out_file, outputs=np.array(self.outputs, dtype=object))
        print(f"Saved outputs to {out_file}")

