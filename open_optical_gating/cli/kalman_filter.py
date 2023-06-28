import numpy as np
from scipy.stats import multivariate_normal


class KalmanFilter():    
    def __init__(self, dt, x, P, Q, R, F, H):
        """
        Kalman filter class based upon filterpy implementation

        Args:
            dt (float): Time between measurements
            x (np.ndarray): Initial system state vector
            P (np.ndarray): Initial system covariance matrix
            Q (np.ndarray): Process noise matrix
            R (float): Measurement noise matrix
            F (np.ndarray): State update matrix
            H (np.ndarray): State measurement vector
        """        
        # Initialise our KF variables, state vector, and model parameters
        self.dt = dt
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.H = H
        
        # Weighting for past Q and R matrices
        # Used for tuning our R matrix
        #self.alpha = 0.3

        self.e = 0
        self.d = 0

        self.xs_prior = []
        self.xs_posteriori = []

        self.flags = {
            "initialised" : False
        }
        
    def predict(self):
        """
        Predict next state from current state vector.

        Returns:
            x: Prior state vector
            P: Prior covariance matrix
        """        
        # Predict our state using our model of the system
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.transpose() + self.Q

        # Store prior state estimates
        self.xs_prior.append(self.x)

        return self.x, self.P

    def update(self, z):       
        """
        Make a measurement and update KF.

        Args:
            z (_type_): _description_

        Returns:
            x: Posteriori state vector
            P: Prior state vector
            e: Residual
            d: Innovation
            S: Innovation covariance
        """        
        # Innovation and innovation covariance
        self.d = z - self.H @ self.x
        self.S = self.H @ self.P @ self.H.transpose() + self.R
        
        # Kalman Gain
        self.K = self.P @ self.H.transpose() @ np.linalg.inv(self.S)
        
        # Posteriori state and covariance estimate
        self.x = self.x + self.K @ self.d
        self.P = self.P - self.K @ self.H @ self.P

        # Store posteriori state estimates
        self.xs_posteriori.append(self.x)
        
        # Residual
        self.e = z - self.H @ self.x

        # Q estimation from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8273755
        # Not entirely convinced this works as expected - IMM filter should (hopefully) replace the need for this
        #self.Q = self.alpha * self.Q + (1 - self.alpha) * (self.K * self.d**2 * self.K.transpose())

        self.L = np.exp(multivariate_normal.logpdf(z, np.dot(self.H, self.x), self.S))

        # Return the most recent state estimate
        return self.x, self.P, self.e, self.d, self.S
        

    @classmethod
    def constant_acceleration(cls, dt, q, R, x_0, P_0):
        """
        Create a constant acceleration filter with 3 dimensions

        Args:
            dt (float): Time between measurements
            q (float): System process noise
            R (float): System measurement noise
            x_0 (np.ndarray): Initial system state
            P_0 (np.ndarray): Initial system covariance

        Returns:
            KalmanFilter: Instance of the Kalman filter class
        """        

        # Set params
        x = x_0
        P = P_0
        Q = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], 
                    [dt**4 / 8, dt**3 / 3, dt**2 / 2], 
                    [dt**3 / 6, dt**2 / 2, dt]]) * q
        F = np.array([[1, dt, dt**2 / 2], 
                    [0, 1, dt], 
                    [0, 0, 1]])
        H = np.array([[1, 0, 0]])
        # Return an instance of the class with params

        return cls(dt, x, P, Q, R, F, H)
    
    @classmethod
    def constant_velocity_3(cls, dt, q, R, x_0, P_0):
        """
        Create a constant velocity filter with 3 dimensions

        NOTE: This is not a good solution for use with the IMM filter as it results in a bias due to the zero acceleration with this velocity model
        TODO: Implement paper by Yuan et al (2012) to fix this (https://ieeexplore.ieee.org/abstract/document/6324701) for now we can just use two 
        constant velocity filters with different process noise levels for our adaptive filter

        Args:
            dt (float): Time between measurements
            q (float): System process noise
            R (float): System measurement noise
            x_0 (np.ndarray): Initial system state
            P_0 (np.ndarray): Initial system covariance

        Returns:
            KalmanFilter: Instance of the Kalman filter class
        """ 
        # Set params
        x = x_0
        P = P_0
        Q = np.array([[dt**3 / 3, dt**2 / 2, 0], 
                    [dt**2 / 2, dt, 0], 
                    [0, 0, 0]]) * q
        F = np.array([[1, dt, 0], 
                    [0, 1, 0], 
                    [0, 0, 0]])
        H = np.array([[1, 0, 0]])

        # Return an instance of the class with params
        return cls(dt, x, P, Q, R, F, H)
    
    @classmethod
    def constant_velocity_2(cls, dt, q, R, x_0, P_0):
        """
        Create a constant velocity filter with 2 dimensions

        Args:
            dt (float): Time between measurements
            q (float): System process noise
            R (float): System measurement noise
            x_0 (np.ndarray): Initial system state
            P_0 (np.ndarray): Initial system covariance

        Returns:
            KalmanFilter: Instance of the Kalman filter class
        """ 
        # Set params
        x = x_0
        P = P_0
        Q = np.array([[dt**3 / 3, dt**2 / 2], 
                    [dt**2 / 2, dt]]) * q
        F = np.array([[1, dt], 
                    [0, 1]])
        H = np.array([[1, 0]])

        # Return an instance of the class with params
        return cls(dt, x, P, Q, R, F, H)
    
    
class InteractingMultipleModelFilter():
    # IMM adapted from filterpy implementation
    
    
    def __init__(self, models, mu, M):
        """
        _summary_

        Args:
            models (list): Instances of Kalman filter class in a list or tuple of length n
            mu (np.array): Initial filter probabilities of length n
            M (np.array): Filter transition matrix of length n x n
        """        
        # Add our models
        # Instances of the KalmanFilter class
        self.models = models
        
        # Initialise state probabilities
        self.mu = mu
        self.M = M
        
        self.omega = np.zeros((len(self.models), len(self.models)))

        self.compute_mixing_probabilities()
        self.compute_state_estimate()
        
    def predict(self):       
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.models, self.omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.models, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.models, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)

        # compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.models):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict()

        # compute mixed IMM state and covariance and save posterior estimate
        self.compute_state_estimate()
                    
    def update(self, z):
        self.d = z - self.x[0]
        
        for i, model in enumerate(self.models):
            model.update(z)
            self.mu[i] = model.L * self.cbar[i]
        self.mu /= np.sum(self.mu)
        
        self.compute_mixing_probabilities()
        self.compute_state_estimate()
            
    def compute_mixing_probabilities(self):
        self.cbar = self.mu @ self.M
        for i in range(len(self.models)):
            for j in range(len(self.models)):
                self.omega[i, j] = (self.M[i, j] * self.mu[i]) / self.cbar[j]
                
    def compute_state_estimate(self):
        self.x = np.zeros(self.models[0].x.shape)
        self.P = np.zeros(self.models[0].P.shape)
        for i, model in enumerate(self.models):
            self.x += model.x * self.mu[i]

        for i, model in enumerate(self.models):
            y = model.x - self.x
            self.P += self.mu[i] * np.outer(y, y) + model.P

    def run(self):
        self.xs = []
        self.Ps = []
        self.mus = []
        for i in range(self.data.shape[0]):
            self.predict()
            self.update(self.data[i])
            self.xs.append(self.x)
            self.Ps.append(self.P)
            self.mus.append(self.mu)

        self.xs = np.asarray(self.xs)
        self.Ps = np.asarray(self.Ps)
        self.mus = np.asarray(self.mus)