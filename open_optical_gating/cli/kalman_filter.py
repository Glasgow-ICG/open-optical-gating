import numpy as np

class KalmanFilter():    
    def __init__(self, dt, x, P, Q, R, F, H):
        """
        Kalman filter class based upon filterpy implementation

        Args:
            dt (_type_): Time between measurements
            x (_type_): Initial system state vector
            P (_type_): Initial system covariance matrix
            Q (_type_): Process noise matrix
            R (_type_): Measurement noise matrix
            F (_type_): State update matrix
            H (_type_): State measurement vector
        """        
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

        self.xs_prior = []
        self.xs_posteriori = []
        
    def predict(self):
        # Predict our state using our model of the system
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.transpose() + self.Q

        # Store prior state estimates
        self.xs_prior.append(self.x)

        return self.x, self.P

    def update(self, z):       
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

        # Return the most recent state estimate
        return self.x, self.P, self.e, self.d
        

    @classmethod
    def constant_acceleration(cls, dt, q, R, x_0, P_0):
        # Create a constant acceleration filter with 3 dimensions
        x = x_0
        P = P_0
        Q = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], 
                    [dt**4 / 8, dt**3 / 3, dt**2 / 2], 
                    [dt**3 / 6, dt**2 / 2, dt]]) * q
        F = np.array([[1, dt, dt**2 / 2], 
                    [0, 1, dt], 
                    [0, 0, 1]])
        H = np.array([[1, 0, 0]])

        return cls(dt, x, P, Q, R, F, H)
    
    @classmethod
    def constant_velocity_3(cls, dt, q, R, x_0, P_0):
        # Create a constant velocity filter with 3 dimensions
        # NOTE: This is not a good solution for use with the IMM filter as it results in a bias due to the zero acceleration with this velocity model
        # TODO: Implement paper by Yuan et al (2012) to fix this (https://ieeexplore.ieee.org/abstract/document/6324701) for now we can just use two 
        # constant velocity filters with different process noise levels for our adaptive filter
        x = x_0
        P = P_0
        Q = np.array([[dt**3 / 3, dt**2 / 2, 0], 
                    [dt**2 / 2, dt, 0], 
                    [0, 0, 0]]) * q
        F = np.array([[1, dt, 0], 
                    [0, 1, 0], 
                    [0, 0, 0]])
        H = np.array([[1, 0, 0]])

        return cls(dt, x, P, Q, R, F, H)
    
    @classmethod
    def constant_velocity_2(cls, dt, q, R, x_0, P_0):
        # This is a constant velocity filter with 2 dimensions
        x = x_0
        P = P_0
        Q = np.array([[dt**3 / 3, dt**2 / 2], 
                    [dt**2 / 2, dt]]) * q
        F = np.array([[1, dt], 
                    [0, 1]])
        H = np.array([[1, 0]])

        return cls(dt, x, P, Q, R, F, H)

