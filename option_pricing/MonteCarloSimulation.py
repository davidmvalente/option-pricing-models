# Third party imports
import numpy as np
from scipy.stats import norm, levy_stable 
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

from stochastic.processes import GeometricBrownianMotion, FractionalBrownianMotion, MultifractionalBrownianMotion

from enum import Enum
# Local package imports
from .base import OptionPricingModel

class RandomProcessType(Enum):
    GeometricBrownianMotion = 1  # Geometric Brownian Motion
    ItoProcess = 2  # Ito Process
    MandelbrotMultifractal = 3  # Multifractal Random Walk
    FractionalBrownianMotion = 4  # Fractional Brownian Motion
    LevyFlight = 5  # Levy Flight

class MonteCarloPricing(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Monte Carlo Simulation.
    We simulate underlying asset price on expiry date using random stochastic process - Brownian motion.
    For the simulation generated prices at maturity, we calculate and sum up their payoffs, average them and discount the final value.
    That value represents option price
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, 
                 number_of_simulations, process_type=RandomProcessType.GeometricBrownianMotion, H=0.5, lam=0.1, alpha=1.5, mu=0):
        """
        Initializes variables used in option pricing simulation.
        
        Parameters:
        underlying_spot_price: float, current stock or other underlying spot price
        strike_price: float, strike price for option contract
        days_to_maturity: int, option contract maturity/exercise date in days
        risk_free_rate: float, returns on risk-free assets (assumed to be constant until expiry date)
        sigma: float, volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_simulations: int, number of potential random underlying price movements
        process_type: ProcessType, type of random process to use for simulation
        H: float, Hurst exponent for fractional Brownian motion and multifractal random walk
        lam: float, intermittency parameter for multifractal random walk
        alpha: float, stability parameter for Levy flight
        mu: float, drift parameter for Ito process
        """
        # Parameters for price process
        self.S_0 = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma 
        
        # Parameters for simulation
        self.N = number_of_simulations
        self.num_of_steps = days_to_maturity
        self.dt = self.T / self.num_of_steps
        
        # Parameters for specific processes
        self.process_type = process_type
        self.H = H
        self.lam = lam
        self.alpha = alpha
        self.mu = mu
        
        # Initialize simulation results
        self.simulation_results_S = None

    def generate_trading_time(self):
        """Generate the multifractal trading time for Mandelbrot Multifractal."""
        t = np.linspace(0, self.T, self.num_of_steps)
        omega = np.random.normal(-self.lam * self.dt, np.sqrt(2 * self.lam * self.dt), self.num_of_steps)
        theta = np.exp(np.cumsum(omega))
        theta = theta / np.mean(theta)  # Normalize
        return np.cumsum(theta) * self.dt

    def generate_fbm(self, scenarios):
        """Generate fractional Brownian motion for Mandelbrot Multifractal """
        # Generate fractional Brownian motion paths
        fbm_paths = np.zeros((self.num_of_steps + 1, self.N))
        for i in range(self.N):
            fbm = FractionalBrownianMotion(t=1, hurst=self.H)
            fbm_paths[:, i] = fbm.times(self.num_of_steps)
                   

        # Initialize scenarios array for asset price paths
        drift = (self.mu - 0.5 * self.sigma**2)

        print("scenarios", scenarios.shape)
        print("fbm_paths", fbm_paths.shape)
        fbm_increment = np.diff(fbm_paths, axis=0)
        print("increments", fbm_increment.shape)
        diffusion = self.sigma * fbm_increment
        change = np.exp(drift + diffusion)
        # Simulate asset price paths using the fbm increments
        for t in range(1, self.num_of_steps):
            for i in range(self.N):
                scenarios[t, i] = scenarios[t - 1, i] * change[t, i]
  
        return scenarios

   
    def simulate_prices(self):
        """
        Simulate price movement of underlying prices using the specified random process.
        """
        np.random.seed(20)
        S = np.zeros((self.num_of_steps, self.N))
        S[0] = self.S_0

        if self.process_type == RandomProcessType.GeometricBrownianMotion:
            for t in range(1, self.num_of_steps):
                Z = np.random.standard_normal(self.N)
                S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z))

        elif self.process_type == RandomProcessType.ItoProcess:
            for t in range(1, self.num_of_steps):
                dW = np.random.normal(0, np.sqrt(self.dt), self.N)
                S[t] = S[t-1] + self.mu*S[t-1]*self.dt + self.sigma*S[t-1]*dW

        elif self.process_type == RandomProcessType.MandelbrotMultifractal:
            mm = FractionalBrownianMotion(hurst=self.H, t=1)
            for t in range(1, self.num_of_steps):
                Q = mm.sample(self.N - 1) 
                S[t] = S[t-1] + self.mu*S[t-1]*self.dt + self.sigma*S[t-1]*Q

        elif self.process_type == RandomProcessType.FractionalBrownianMotion:
            S = self.generate_fbm(S)


        elif self.process_type == RandomProcessType.LevyFlight:
            for t in range(1, self.num_of_steps):
                dL = levy_stable.rvs(self.alpha, 0, size=self.N)
                S[t] = S[t-1] * np.exp(self.r * self.dt + self.sigma * dL)

        else:
            raise ValueError("Invalid process type.")

        self.simulation_results_S = S

    def _calculate_call_option_price(self): 
        """
        Call option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Call option payoff (it's exercised only if the price at expiry date is higher than a strike price): max(S_t - K, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.simulation_results_S[-1] - self.K, 0))
    

    def _calculate_put_option_price(self): 
        """
        Put option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Put option payoff (it's exercised only if the price at expiry date is lower than a strike price): max(K - S_t, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.K - self.simulation_results_S[-1], 0))
       

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.simulation_results_S[:, 0:num_of_movements])
        ax.axhline(self.K, c='k', xmin=0, xmax=self.num_of_steps, label='Strike Price')
        ax.set_xlim([0, self.num_of_steps])
        ax.set_ylabel('Simulated price movements')
        ax.set_xlabel('Days in future')
        ax.set_title(f'First {num_of_movements}/{self.N} Random Price Movements')
        ax.legend(loc='best')
        
        return fig