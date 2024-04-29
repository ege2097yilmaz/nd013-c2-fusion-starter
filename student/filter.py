# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import logging
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # process model dimension
        self.dt = params.dt # time increment
        self.q = params.q # process noise variable for Kalman filter Q


    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        n = self.dim_state  
        F = np.eye(n)  
        
        for i in range(3):
            F[i, i + 3] = self.dt  

        print("state matrix is completed")

        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dt = self.dt
        q = self.q
        q1 = dt * q
        Q = np.zeros((self.dim_state, self.dim_state))
        np.fill_diagonal(Q, q1)
        
        Q = np.matrix([
            [(dt**5)/20, 0, (dt**4)/8, 0, (dt**3)/6, 0],
            [0, (dt**5)/20, 0, (dt**4)/8, 0, (dt**3)/6],
            [(dt**4)/8, 0, (dt**3)/3, 0, (dt**2)/2, 0],
            [0, (dt**4)/8, 0, (dt**3)/3, 0, (dt**2)/2],
            [(dt**3)/6, 0, (dt**2)/2, 0, dt, 0],
            [0, (dt**3)/6, 0, (dt**2)/2, 0, dt]
        ]) * q

        return Q
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        # Compute the state transition matrix and process noise covariance matrix
        F = self.F()
        Q = self.Q()
        
        # Predict the next state using the linear motion model assumption
        x_predicted = F @ track.x  # Using @ for matrix multiplication
        
        # Predict the next covariance matrix incorporating process noise
        P_predicted = F @ track.P @ F.T + Q  # Using @ for matrix multiplication and .T for transpose
        
        # Update the track object with the new predicted state and covariance
        track.set_x(x_predicted)
        track.set_P(P_predicted)

        print("prediction step is completed")
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        try:
            # Measurement matrix from sensor model
            H = meas.sensor.get_H(track.x)
            
            # Calculate the residual between the predicted state and the measurement
            gamma = self.gamma(track, meas)
            
            # Calculate the covariance of the residual
            S = self.S(track, meas, H)
            
            # Compute the Kalman gain
            K = track.P @ H.T @ np.linalg.inv(S)
            
            # Update the state estimate with the new measurement
            x_updated = track.x + K @ gamma
            
            # Update the error covariance estimate
            I = np.identity(self.dim_state)  # Identity matrix of the same dimension as the state
            P_updated = (I - K @ H) @ track.P
            
            # Set the updated state and covariance in the track object
            track.set_x(x_updated)
            track.set_P(P_updated)
            
            # Optionally update other attributes based on the measurement
            track.update_attributes(meas)

            print("update step is completed")
        
        except Exception as e:
            logging.error("Unable to update the state and covariance: {}".format(e))
        ############
        # END student code
        ############ 
        
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############        
        try:
            # Calculate the expected measurement from the current state estimate
            expected_measurement = meas.sensor.get_hx(track.x)
            
            # Compute the residual
            residual = meas.z - expected_measurement
            
            return residual
        
        except Exception as e:
            logging.error(f"Error in computing the residual: {e}")
            raise
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        s = H * track.P * H.transpose() + meas.R # covariance of residual
        return s
        
        ############
        # END student code
        ############ 