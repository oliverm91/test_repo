import numpy as np
from scipy.optimize import minimize

class NelsonSiegelSvenson:
    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        self.tenors = tenors
        self.rates = rates
        self.beta0 = rates[0]
        self.beta1 = -self.beta0/10
        self.beta2 = self.beta1
        self.beta3 = 1
        self.lambda1 = 1
        self.lambda2 = 1
        self.calibrate()

    def calibrate(self):
        def squared_errors(params) -> float:
            self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2 = params
            return ((100*(self.get_zero_rates(self.tenors)-self.rates))**2).sum()
        initial_guess = [self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2]
        min_res = minimize(squared_errors, initial_guess)
        self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2 = min_res.x

    def get_zero_rates(self, tenors: np.ndarray) -> np.ndarray:
        t_lambda1 = tenors/self.lambda1
        t_lambda2 = tenors/self.lambda2
        exp_t_lambda1 = np.exp(-t_lambda1) 
        exp_t_lambda2 = np.exp(-t_lambda2)
        aux_term2 = (1-exp_t_lambda1)/(t_lambda1)
        term2 = self.beta1*(aux_term2)
        term3 = self.beta2*(aux_term2-exp_t_lambda1)
        term4 = self.beta3*((1-exp_t_lambda2)/(t_lambda2)-exp_t_lambda2)
        return self.beta0 + term2 + term3 + term4