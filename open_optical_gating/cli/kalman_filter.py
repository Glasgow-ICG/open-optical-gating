import numpy as np
from scipy.stats import multivariate_normal


class KalmanFilter():    
    def __init__(self, settings, dt, x, P, Q, R, F, H):
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

        self.settings = settings

        # TODO: Remove Q and F setting - set these manually using methods / params
        # Initialise our KF variables, state vector, and model parameters
        self.dt = dt
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.H = H

        self.e = 0
        self.d = 0
        self.S = None

        self.xs_prior = []
        self.xs_posteriori = []

        self.alpha = self.settings["alpha"]**2 # used for fading memory Kalman filter. Set to a value close to 1 such as 1.01

        self.flags = {
            "initialised" : False
        }

    def initialise(self, x, P):
        """
        Initialise the Kalman filter with a new state vector and covariance matrix

        Args:
            x (np.ndarray): Initial system state vector
            P (np.ndarray): Initial system covariance matrix
        """        
        self.x = x
        self.P = P

        self.flags["initialised"] = True

    def set_process_noise_covariance_matrix(self, dt_matrix, float_matrix, q_matrix, dt, q):
        """
        Set the process noise covariance matrix.
        We use a slightly non-standard method here but this allows us to modify dt and q on the fly
        by splitting our Q matrix into three parts.
        The first part gives the powers to raise our dt to, the second part is our float array, and the third
        part is the process noise value.

        TODO: We should enforce the use of this for setting the Q matrix by using the @property decorator
        """
        self.Q_dt = dt_matrix
        self.Q_f = float_matrix
        self.Q_q = q_matrix
        self.q = q

        np.random.seed()
        #self.q = np.random.uniform(0.08, 0.14, 1)[0]

        self.Q = self.get_process_noise_covariance_matrix(dt)#dt**self.Q_dt * self.Q_f * self.q * self.Q_q

        return self.Q

    def get_process_noise_covariance_matrix(self, dt):
        """
        Get the process noise covariance matrix with the given dt

        Args:
            dt (_type_): 
        """

        Q = dt**self.Q_dt * self.Q_f * self.q * self.Q_q

        return Q

    def set_state_transition_matrix(self, dt_matrix, float_matrix, dt):
        """
        Set the state transition matrix.
        We use a slightly non-standard method here but this allows us to modify dt on the fly
        by splitting our F matrix into two parts.
        The first part gives the powers to raise our dt to, the second part defines what to multiply
        each value of the matrix by.
        """

        self.F_dt = dt_matrix
        self.F_f = float_matrix

        self.F = dt**dt_matrix * float_matrix

    def get_state_transition_matrix(self, dt):
        """
        Return the state transition matrix with the given dt. 
        As with above this allows us to modify dt on the fly for forward predction
        """
        return dt**self.F_dt * self.F_f


    def predict(self):
        """
        Predict next state from current state vector.

        Returns:
            x: Prior state vector
            P: Prior covariance matrix
        """        
        # Predict our state using our model of the system
        self.x = self.F @ self.x
        self.P = self.alpha * (self.F @ self.P @ self.F.transpose()) + self.Q

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
        
        # This code attempts to detect outliers by ignoring measurements which are more than "gating_level" away
        # using the mahalanobis distance.
        if self.settings["gating_method"] == "mahalanobis":
            dist = np.sqrt(np.dot(np.dot((self.d), np.linalg.inv(self.S)), (self.d)))
            if dist < self.settings["gating_level"]:
                # Posteriori state and covariance estimate
                self.x = self.x + self.K @ self.d
                self.P = self.P - self.K @ self.H @ self.P
            else:
                # TODO: Add support for logger
                print("Outlier ignored")
        else:
            # If not using a gating method then peform the usual update step
            self.x = self.x + self.K @ self.d
            self.P = self.P - self.K @ self.H @ self.P

        # Store posteriori state estimates
        self.xs_posteriori.append(self.x)
        
        # Residual
        self.e = z - self.H @ self.x

        # Get likelihood
        self.L = multivariate_normal.logpdf(z, self.H @ self.x, self.S)
        self.L = (1 / np.sqrt(2 * np.pi * self.S)) * np.exp(-0.5 * self.e * np.linalg.inv(self.S) * self.e)

        # Return the most recent state estimate
        return self.x, self.P, self.e, self.d, self.S, self.L
    
    def get_current_state_vector(self):
        """
        Returns the current Kalman filter state vector
        """

        return self.x
    
    def get_current_covariance_matrix(self):
        """
        Returns the current Kalman filter covariance matrix
        """

        return self.P
    
    def get_normalised_innovation_squared(self):
        if self.S is not None:
            NIS = self.e @ np.linalg.inv(self.S) @ self.e
            return NIS
        

    @classmethod
    def constant_acceleration(cls, settings, dt, q, R, x_0, P_0):
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

        # Create the instance
        kf = cls(settings, dt, x, P, Q, R, F, H)

        # Set the process noise covariance matrix
        dt_matrix = np.array([[5, 4, 3],
                              [4, 3, 2],
                              [3, 2, 1]])
        float_matrix = np.array([[1/20, 1/8, 1/6],
                                [1/8, 1/3, 1/2],
                                [1/6, 1/2, 1]])
        kf.set_process_noise_covariance_matrix(dt_matrix, float_matrix, q, dt, q)

        # Set the state transition matrix
        dt_matrix = np.array([[0, 1, 2],
                            [0, 0, 1],
                            [0, 0, 0]])
        float_matrix = np.array([[1, 1, 1],
                                [0, 1, 1],
                                [0, 0, 1]])
        kf.set_state_transition_matrix(dt_matrix, float_matrix, dt)

        return kf

    @classmethod
    def constant_velocity_3(cls, settings, dt, q, R, x_0, P_0):
        """
        Create a constant velocity filter with 3 dimensions

        NOTE: This is not a good solution for use with the IMM filter as it results in a bias due to the zero acceleration with this velocity model
        TODO: Implement paper by Yuan et al (2012) to fix this (https://ieeexplore.ieee.org/abstract/document/6324701) for now we can just use two 
        constant velocity filters with different process noise levels for our adaptive filter.

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

        # Create the instance
        kf = cls(settings, dt, x, P, Q, R, F, H)

        # Set the process noise covariance matrix
        dt_matrix = np.array([[3, 2, 0],
                              [2, 1, 0],
                              [0, 0, 0]])
        float_matrix = np.array([[1/3, 1/2, 0],
                                [1/2, 1, 0],
                                [0, 0, 0]])
        kf.set_process_noise_covariance_matrix(dt_matrix, float_matrix, q, dt, q)

        # Set the state transition matrix
        dt_matrix = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        float_matrix = np.array([[1, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
        kf.set_state_transition_matrix(dt_matrix, float_matrix, dt)

        return kf
    
    @classmethod
    def constant_velocity_2(cls, settings, dt, q, R, x_0, P_0):
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

        # Create the instance
        kf = cls(settings, dt, x, P, Q, R, F, H)

        # Set the process noise covariance matrix
        dt_matrix = np.array([[3, 2],
                              [2, 1]])
        float_matrix = np.array([[1/3, 1/2],
                                [1/2, 1]])
        kf.set_process_noise_covariance_matrix(dt_matrix, float_matrix, q, dt, q)

        # Set the state transition matrix
        dt_matrix = np.array([[0, 1],
                            [0, 0]])
        float_matrix = np.array([[1, 1],
                                [0, 1]])
        kf.set_state_transition_matrix(dt_matrix, float_matrix, dt)

        return kf
    
    @staticmethod
    def get_time_til_phase(x, targetSyncPhase):
        # Get KF Parameters
        thisFramePhase = x[0] % (2 * np.pi)
        radsPerSec = x[1]

        # Estimate time til trigger
        phaseToWait = targetSyncPhase - thisFramePhase
        while phaseToWait < 0:
            phaseToWait += 2 * np.pi
        timeToWait_s = phaseToWait / radsPerSec
        timeToWait_s = max(timeToWait_s, 0.0)

        # Get the estimated heart period
        estHeartPeriod_s = 2 * np.pi / radsPerSec

        return timeToWait_s, estHeartPeriod_s
    
    def make_forward_prediction(self, x, t0, t):
        """
        Attempt to forward predict using a custom dt.

        Args:
            x (_type_): _description_
            t0 (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        # First get our timestep
        dt = t - t0

        # Get our matrices with the new dt
        Q = self.get_process_noise_covariance_matrix(dt)
        F = self.get_state_transition_matrix(dt)

        # Predict our future state
        x = F @ x
        P = F @ self.P @ F.transpose() + Q

        return x, P


class InteractingMultipleModelFilter():
    # IMM adapted from filterpy implementation
    
    def __init__(self, models, mu, M):
        """
        _summary_

        Args:
            models (list): Instances of Kalman filter class in a list or tuple of length n
            mu (np.array): Initial filter probabilities of length n
            M (np.array): Filter transition matrix of length n x n

            NOTE: This is problematic if our different filter models are of different order (e.g we have a constant
            acceleration filter and a constant-velocity filter.
            TODO: Add some checks to ensure our vectors/matrices are of the correct dimensions.
            Few solutions:
                Stick to filters of matching dimensions and vary our process noise in our models
                Augment our lower dimensional filter with zeros in the higher order states
                A few papers have alternative solutions:
                    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6324701
                    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717415

                First two are very easy to implement so will work on this 
        """        
        # Add our models
        # Instances of the KalmanFilter class
        self.models = models
        
        # Initialise state probabilities
        self.mu = mu
        self.M = M
        
        self.omega = np.zeros((len(self.models), len(self.models)))
        self.likelihood = np.zeros(len(self.models))

        self.compute_mixing_probabilities()
        self.compute_state_estimate()

        # NOTE : The inhomogeneous fix is not implemented yet - will see how current method performs to see if it's necessary.
        self.settings = {
            "inhomogeneousfix" : "augmented" # Fix for when the IMM model state vectors have differing dimensions. Options are none, zero-augmented, augmented
        }

        self.mus = []

    def predict(self):
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.models, self.omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.models, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros_like(self.P)
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
            #self.mu[i] = model.L * self.cbar[i]
            self.likelihood[i] = model.L

        self.mu = self.likelihood * self.cbar
        self.mu /= np.sum(self.mu)
        self.mus.append(self.mu)

        
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

    def get_current_state_vector(self):
        """
        Returns the current Kalman filter state vector
        """

        return self.x
    
    def get_current_covariance_matrix(self):
        """
        Returns the current Kalman filter covariance matrix
        """

        return self.P
    
    def make_forward_prediction(self, x, t0, t):
        """
        Attempt to forward predict using a custom dt.
        TODO: This only uses one of the models - need to adapt this to use multiple models

        Args:
            x (_type_): _description_
            t0 (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        # First get our timestep
        dt = t - t0

        # Get our matrices with the new dt
        Q = self.models[0].get_process_noise_covariance_matrix(dt)
        F = self.models[0].get_state_transition_matrix(dt)

        # Predict our future state
        x = F @ x
        P = F @ self.P @ F.transpose() + Q

        return x, P