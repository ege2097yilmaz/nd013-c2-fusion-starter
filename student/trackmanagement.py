# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os, logging
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print(f'Creating track no. {id}')
        self.id = id

        # Extract the rotation matrix from the sensor to vehicle coordinates
        Rot = meas.sensor.sens_to_veh[0:3, 0:3]

        # Initialize position in homogeneous coordinates and transform it to vehicle coordinates
        pos_sens = np.ones((4, 1))
        pos_sens[0:3] = meas.z[0:3]
        pos_veh = meas.sensor.sens_to_veh @ pos_sens  # Using @ for matrix multiplication

        # Initialize state vector with position (from sensor to vehicle coordinates) and zero velocity
        self.x = np.zeros((6, 1))
        self.x[0:3] = pos_veh[0:3]

        # Compute the position covariance component
        P_pos = Rot @ meas.R @ Rot.T  # Rotate the measurement covariance

        # Initialize velocity covariance with predefined parameters
        P_vel = np.diag([params.sigma_p44**2, params.sigma_p55**2, params.sigma_p66**2])

        # Construct the overall covariance matrix
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel

        # Initialize state and score
        self.state = 'initialized'
        self.score = 1 / params.window

        # Additional track attributes derived from measurements and transformations
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        # Transform yaw from sensor to vehicle coordinates
        self.yaw = np.arccos(Rot[0, 0] * np.cos(meas.yaw) + Rot[0, 1] * np.sin(meas.yaw))
        self.t = meas.t

        # Logging successful initialization
        logging.info(f'Track {self.id} initialized with state {self.state} and score {self.score}')

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            R = meas.sensor.sens_to_veh
            self.yaw = np.arccos(R[0,0]*np.cos(meas.yaw) + R[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: implement track management:
        
        # Decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # Check if the sensor has the track in its field of view (FOV)
            if meas_list and meas_list[0].sensor.in_fov(track.x):
                track.state = 'tentative'
                # Ensure score does not exceed a defined maximum threshold
                track.score = min(track.score, params.delete_threshold + 1)
                # Decrease score based on the defined window parameter
                track.score -= 1 / params.window
        
        # Delete tracks if they meet the criteria for deletion
        self.track_list = [
            track for track in self.track_list if not (
                track.score <= params.delete_threshold and
                (track.P[0, 0] >= params.max_P or track.P[1, 1] >= params.max_P)
            )
        ]

        # Initialize new tracks from unassigned measurements using only lidar data
        for j in unassigned_meas:
            if meas_list[j].sensor.name == 'lidar':
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        ############
        # TODO Step 2: implement track management for updated tracks:
        # Increment the track score proportionally based on the window parameter
        track.score += 1. / params.window

        # Update the state based on the score compared to a predefined threshold
        if track.score > params.confirmed_threshold:
            track.state = 'confirmed'
        else:
            track.state = 'tentative'

        # Log state change for debugging and monitoring
        logging.info(f"Track {track.id} updated: Score = {track.score:.2f}, State = {track.state}")
            

        
        ############
        # END student code
        ############ 
