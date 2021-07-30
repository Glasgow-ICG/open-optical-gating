''' KalmanFilter class for phase prediction '''

# NOTE: There are three TODOs in this file:
#   - The first concerns the measurement variance calculation. Currently, the variance on the 
#     current measurement is assigned to be sqrt((var_m*var_m_factor)**2 + baseline_var**2)
#     where var_m is the greatest value in var_m_cache. This cache contains the second 
#     derivative values of a number of recent points. The length of this cache is controlled 
#     by var_m_cache_length. It is expected that at some point in the future, this logic will 
#     refined in one of two ways:
#       + The code may start taking into account the second derivative of the current point.
#         Currently this is not possible since we need the subsequent point to calculate
#         the second derivative. We could get around this by making an initial var_m estimate
#         using var_m_cache, and then going back to revise this estimate when we have access
#         to the subsequent phase measurement.
#       + The code may start using a metric based on the phase measurement's SAD array. This
#         would be the more elegant solution and would only require data from the current point.
#   - The second concerns the way that the code determines delta_t. Currently, the code trusts 
#     the user-defined "brightfield_framerate" option in the json file, however it would be very
#     easy to use the difference between timetamps instead. Also, it may be useful to add some
#     logic to deal with dropped frames/inaccurate timestamps etc in the future.
#   - The third concerns the Kalman filter error returned by the filter. Currently this error
#     on the prediction is simply logged. It is hoped that in the future this error might be 
#     returned for further use in the optical gating code. This would require the current 
#     model to be refined so that the prediction error is more useful.  


# Module imports
import numpy as np
from loguru import logger


class KalmanFilter2D:
    ''' Controls Kalman filter predictions and updates
    '''

    def __init__(self, var_m_factor, var_a_factor, baseline_var, phase_cache_length, var_m_cache_length, 
                 init_v, init_var_x, init_var_a, delta_t):
        ''' Initialises instance of class.
            Parameters:
                var_m_factor:           float   Factor applied to the measurement variance
                var_a_factor:           float   Factor applied to the acceleration variance 
                baseline_var:           float   Baseline variance to ensure that error given to 
                                                Kalman filter is never too close to zero. Added
                                                in quadrature with (var_m * var_m_factor)
                phase_cache_length:     int     Number of previous phase values to store in the cache
                                                for the purpose of calculating the acceleration 
                                                variance used for the adaptive Q matrix
                var_m_cache_length:     int     Number of previous measurement variances to store in 
                                                the cache. The biggest measurement in this cache is 
                                                used as the variance for the current phase
                                                measurement when performing the Kalman update
                init_v:                 float   Initial velocity estimate
                init_var_x:             float   Initial state variance
                init_var_a:             float   Initial acceleration variance
                delta_t:                float   User specified constant timestep between measurements
            Returns:
                None
        '''

        # Check if phase_cache_length is long enough to allow second derivatives to be calculated
        if phase_cache_length < 3:
            logger.warning("Kalman filter initialisation error: phase_cache_length is too small. All variance measurements will default to init_var_x")

        # Initialise general properties
        self.status = "AwaitingPrediction"                  # Status variable
        self.mode = "initialised"                           # Mode variable
        self.init_v = init_v                                # Initial velocity variable
        self.init_var_x = init_var_x                        # Initial variance of state vector elements
        self.init_var_a = init_var_a                        # Initial acceleration variance
        self.phase_cache_length = phase_cache_length        # Phase cache length
        self.var_m_cache_length = var_m_cache_length        # Measurement variance cache length
        self.var_m_factor = var_m_factor                    # Measurement variance factor
        self.var_a_factor = var_a_factor                    # Acceleration variance factor
        self.baseline_var = baseline_var                    # Baseline variance
        self.delta_t = delta_t                              # Timestamp delta

        # Log success
        logger.info("Kalman filter initialised")


    def initialise_state_variables(self, init_x, init_v, init_var_x, init_var_a):
        ''' Initialises state variables (variables which change during the life of the filter).
            Called when self.mode is changed
            Parameters:
                init_x:         float   Initial position (phase)
                init_v:         float   Initial speed (rad/s)
                init_var_x:     float   Initial variance of state vector elements
                init_var_a:     float   Initial acceleration variance
        '''
        self.x = np.array([[init_x],                        # State vector
                           [init_v]])                                          
        self.P = np.array([[init_var_x, 0],                 # Covariance matrix
                           [0, init_var_x]])     
        self.H = np.array([[1, 0]])                         # Observation matrix                                                  
        self.Q = 0                                          # Process noise matrix
        self.F = 0                                          # State transition matrix
        self.R = 0                                          # Measurement covariance matrix
        self.K = 0                                          # Kalman gain
        self.var_a = init_var_a                             # Acceleration variance
        self.phase_cache = []                               # Phase cache for adaptive Q
        self.var_m_cache = []                               # Measurement variance cache


    def update_F(self, delta_t):
        ''' Updates the 2D F matrix according to the given delta_t.
            Parameters:
                delta_t:    float   Time interval
            Returns:
                None   
        '''

        self.F = np.array([[1, delta_t], 
                           [0, 1      ]])

    
    def update_Q(self, delta_t):
        ''' Updates the 2D discrete noise matrix according to the given delta_t and var_a.
            Parameters:
                delta_t:    float   Time interval
            Returns:
                None
        '''

        self.Q = np.array([[delta_t**4/4, delta_t**3/2], 
                           [delta_t**3/2, delta_t**2  ]]) * self.var_a * self.var_a_factor
        # Check eigenvalues
        self.check_eigs(stage="predict (Q matrix)", matrix=self.Q)

    
    def update_var_a(self):
        ''' Updates the acceleration variance according to the values in the phase_cache.
            Parameters:
                None
            Returns:
                None
        '''

        # Check cache size. Cache is only used if it is at least the correct size,
        # if it is smaller then current value is not changed
        if len(self.phase_cache) >= self.phase_cache_length:
            # Trim cache if it is too long
            if len(self.phase_cache) > self.phase_cache_length:
                self.phase_cache = self.phase_cache[1:]

            # Calculate acceleration variance
            accs = []
            for i in range(1, len(self.phase_cache)-1):
                accs.append(np.abs(self.phase_cache[i-1] - 2*self.phase_cache[i] + self.phase_cache[i+1]))
            accs = np.array(accs)
            avg_accs = sum(accs)/len(accs)
            temp_var_a = sum((accs - avg_accs)**2)/len(accs)
            self.var_a = temp_var_a


    def find_var_m(self, z):
        ''' Given new phase measurement, produces a variance value based on the second derivative.
            Note: this currently assigns the second derivative from the previous point as the variance
                for the current point. This is because the future point is not yet known. This will 
                need to be tidied up at some point.
            Parameters:
                z:      float   New phase measurement 
            Returns:
                var_m:  float   Variance on phase measurement  
        '''
        #TODO: Revise variance calculation (see the note at the top of the file)

        # Add measurement to cache
        self.phase_cache.append(z)
        # Check phase cache has at least three points. If not, self.init_var_x is used
        if len(self.phase_cache) >= 3:
            # Find second derivative of last point
            second_derivative = np.abs(self.phase_cache[-3] - 2*self.phase_cache[-2] + self.phase_cache[-1])
            # Add second derivative to cache
            self.var_m_cache.append(second_derivative)
            # Check var_m cache length and trim if too long
            if len(self.var_m_cache) > self.var_m_cache_length:
                self.var_m_cache = self.var_m_cache[1:]
            # Find biggest value in var_m cache
            temp_var_m = max(self.var_m_cache)
            # Combine value with var_m_factor and baseline_var
            temp_var_m = np.sqrt((temp_var_m * self.var_m_factor)**2 + self.baseline_var**2)
        else:
            temp_var_m = self.init_var_x
        return(temp_var_m)


    def predict(self, save, delta_t):
        ''' Performs Kalman prediction, with option to save prediction to state.
            Parameters:
                save:       bool    Decides whether prediction should be saved to state. If false,
                                    predicted x and P are returned
                delta_t:    float   Time interval
            Returns:
                temp_x:     np.array    Predicted x (only returned if save==False)
                temp_P:     np.array    Predicted P (only returned if save==False)
        '''

        # Check KalmanFilter has the correct status
        if self.status == "AwaitingMeasurement":
            logger.warning("Kalman filter predict error: status must be \"AwaitingPrediction\" not \"AwaitingMeasurement\"")
        else:
            # Update F and Q
            self.update_F(delta_t)
            self.update_Q(delta_t)
            # Predict temp_x
            temp_x = self.F @ self.x
            # Predict temp_P
            temp_P = self.F @ self.P @ self.F.T + self.Q
            # Save/return predicted values and check eigenvalues
            if save:
                # Update status
                self.status = "AwaitingMeasurement"
                self.check_eigs(stage="predict (P matrix) (saved)", matrix=temp_P)
                self.P = temp_P
                self.x = temp_x
            else:
                self.check_eigs(stage="predict (P matrix) (not saved)", matrix=temp_P)
                return(temp_x, temp_P)


    def update(self, z, var_m):
        ''' Performs Kalman update when given a new measurement and variance.
            Parameters:
                z:      float   New phase measurement
                var_m:  float   Measurement variance
            return:
                None
        '''

        # Check KalmanFilter has the correct status
        if self.status == "AwaitingPrediction":
            logger.warning("Kalman filter update error: status must be \"AwaitingMeasurement\" not \"AwaitingPrediction\"")
        else:
            # Update var_a
            self.update_var_a()
            # Compute Kalman gain
            self.R = np.array([[var_m]])
            self.K = (self.P @ self.H.T) * np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            # Update x
            self.x = self.x + self.K * (z - self.H @ self.x)
            # Update P using the "Joseph form" equation. This is far more numerically stable than
            # the standard update equation
            I = np.identity((self.K @ self.H).shape[0])
            temp = I - (self.K @ self.H)
            self.P = (temp @ self.P @ temp.T) + (self.K @ self.R @ self.K.T)
            # Check eigenvalues
            self.check_eigs(stage="update (P matrix)", matrix=self.P)
            # Update status
            self.status = "AwaitingPrediction"

    
    def check_eigs(self, stage, matrix):
        ''' Checks the eigenvalues of the given matrix are positive and prints a warning if not.
            Parameters:
                stage:      string      Description of the filter stage to print along with error
                matrix:     np.array    Array to check the eigenvalues of
            Returns:
                None
        '''

        # Compute eigenvalues
        eig, _ = np.linalg.eig(matrix)
        warning_eigs = []
        # Count negative eigenvalues (however, we are not interested in really small ones)
        for i in eig:
            if i < -1e-15:
                warning_eigs.append(i)
        if len(warning_eigs) > 0:
            # Print eigenvalue warning
            logger.warning("Kalman filter eigenvalue warning(s) in {} stage: {}".format(stage, warning_eigs))

    
    def prepare_to_sync(self):
        ''' Switches mode to "determine", so that filter can tell when first "sync" measurement occurs.
            Parameters:
                None
            Returns:
                None
        '''

        # Check if mode has changed. If mode has changed, calculate init_v. Otherwise, raise warning
        if self.mode != "determine":
            logger.info("Kalman filter changing mode: \"{}\" to \"{}\"".format(self.mode, "determine"))
            # Update mode
            self.mode = "determine"
        else:
            # Log warning
            logger.warning("Kalman filter mode warning: \"prepare_to_sync()\" should not be called more than once in a row.")
    
    def process_measurement(self, z, timestamp):
        ''' Process new phase measurement in "sync" mode.
            Parameters:
                z:              float       New phase measurement
                timestamp:      float       Timestamp of phase measurement
            Returns:
                None
        '''

        # Check if mode has changed. If mode has changed, state variables are reset with the current measurement
        # set as init_x. Otherwise, Kalman filter predict and update stages are carried out
        if self.mode != "sync":
            logger.info("Kalman filter changing mode: \"{}\" to \"{}\"".format(self.mode, "sync"))
            # Update mode
            self.mode = "sync"
            # Reset variables
            self.initialise_state_variables(init_x=z, init_v=self.init_v, init_var_x=self.init_var_x, init_var_a=self.init_var_a)
        else:
            # Find measurement variance
            var_m = self.find_var_m(z)
            # Perform Kalman predict step, and save result to state 
            # TODO: Revise handling of delta_t (see note at the top of the file)   
            self.predict(save=True, delta_t=self.delta_t)
            # Perform Kalman update step
            self.update(z, var_m)



    def kalman_predict_trigger_wait(self, frame_history, pog_settings):
        ''' Predicts how long we must wait until the next target phase occurs. This function is
            intended to be an alternative to prospective_optical_gating.predict_trigger_wait().
            Parameters:
                frame_history   nparray     Nx3 array of [timestamp in seconds, unwrapped phase, argmin(SAD)]
                pog_settings    dict        Parameters controlling the sync algorithms
            Returns:
                time_to_wait    float       Time (in seconds) to wait until next target phase
                rads_per_sec    float       Current heart rate
        '''

        # Determine unwrapped phase of next target phase
        current_phase = self.x[0][0]
        rads_per_sec = self.x[1][0]
        unwrapped_target_phase = pog_settings["targetSyncPhase"]
        while unwrapped_target_phase <= current_phase:
            unwrapped_target_phase += 2*np.pi
        
        # Check rads_per_sec
        if rads_per_sec < 0:
            logger.warning("Heart rate given by Kalman filter is negative! This will be a problem for prediction.")
        elif rads_per_sec == 0:
            logger.warning("Heart rate given by Kalman filter is zero! This will be a problem for prediction (divByZero).")
        
        # Find time to wait until next target phase. This assumes that the Kalman
        # filter is using a simple model, and just uses t=d/v
        time_to_wait = (unwrapped_target_phase - current_phase)/rads_per_sec

        # Perform Kalman predict step, without saving, to obtain variance on 
        # prediction. Note that this is not currently used for anything.
        # TODO: Revise handling of prediction error (see note at the top of the file)
        predicted_phase, predicted_variance = self.predict(save=False, delta_t=time_to_wait)

        # Do some logging
        logger.info("Current time: {};\tTime to wait: {};".format(frame_history[-1, 0], time_to_wait))
        logger.debug("Current phase: {};\tPhase to wait: {};".format(current_phase, unwrapped_target_phase - current_phase))
        logger.debug("Target phase:{};\tPredicted phase:{};\tKalman filter prediction variance:{};".format(unwrapped_target_phase, predicted_phase, predicted_variance))


        return(time_to_wait, rads_per_sec)
        
