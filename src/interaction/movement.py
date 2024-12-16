from typing import override
import numpy as np
from det2d import Keys
from coordinates import cartesian2polar
from .limits import Limit

class MovementLimit(Limit):
    def __init__(self,
            maximum_velocity_magnitude: float,
            frame_step: int,
            keypoint_classes: (tuple, list, np.ndarray),
            n_keypoints_required: int=-1,
            confidence_threshold: float=0.0):
        '''
        A class representing a movement magnitude limit within which a group of keypoints is considered to stand still
        
        Inputs:
        - maximum_velocity_magnitude: restriction on the maximum magnitude the keypoints velocity can have whilst standing still
        - frame_step: frame_step which governs the calculation of velocity for testing
        - keypoint_classes: tuple of keypoint classes to test maximum_velocity_magnitude on
        - n_keypoints_required: how many non-absent keypoints from keypoint_classes must be still for the group to be still, make -1 to require all classes to be in patient area
        - confidence_threshold: threshold below which a keypoint is said to be absent (and hence still)
        '''
        super().__init__(keypoint_classes, n_keypoints_required, confidence_threshold)
        
        assert maximum_velocity_magnitude > 0, "maximum_velocity_magnitude must be greater than 0"
        assert frame_step > 0, "frame_step must be at least 1"
        
        self._maximum_velocity_magnitude = maximum_velocity_magnitude
        self._frame_step = frame_step
        
    @override
    def tracklet_interacting(self, tracklet: dict) -> np.ndarray:
        '''
        See keypoints_still()
        '''
        return self.keypoints_still(tracklet)
        
    def keypoints_still(self, tracklet: dict) -> np.ndarray:
        '''
        Detect whether a group of keypoints is still or not
        
        Inputs:
        - tracklet: the personnel tracklet
        
        Outputs:
        - Array with bools showing whether the keypoints were standing still (True) or not (False) per frame
        '''
        keypoint_velocities = self.velocity_history(tracklet)
        keypoint_velocities = keypoint_velocities[:,self.keypoint_classes,:]
        keypoints_still = (keypoint_velocities[:,:,0] <= self.maximum_velocity_magnitude) | ~(keypoint_velocities[:,:,2] >= self.confidence_threshold)
        keypoints_still = np.sum(keypoints_still, axis=1) >= self.n_keypoints_required
        keypoints_still = np.append(np.ones(shape=(min(self.frame_step, tracklet[Keys.keypoints].shape[0]),), dtype=bool), keypoints_still)
        
        return keypoints_still
    
    def velocity_history(self, tracklet: dict) -> np.ndarray:
        '''
        Calculate tracklet velocity magnitude and orientation
        
        Inputs:
        - tracklet: the personnel tracklet
        
        Outputs:
        - velocity history with confidences, shape [F-frame_step,K,3]
        '''
        keypoints = tracklet[Keys.keypoints].copy()
        cartesian_velocities = keypoints[self.frame_step:,:,:2]-keypoints[:-self.frame_step,:,:2]
        polar_velocities = cartesian2polar(cartesian_velocities)
        velocity_confidences = keypoints[self.frame_step:,:,2,None]*keypoints[:-self.frame_step,:,2,None]
        velocities = np.concatenate([polar_velocities, velocity_confidences], axis=-1)
        
        return velocities
    
    @classmethod
    @override
    def from_dict(cls, movement_limit_dict: dict):
        '''
        Build a MovementLimit instance from a descriptive dictionary
        
        Inputs:
        - movement_limit_dict: descriptive dictionary
        
        Outputs:
        - MovementLimit
        '''
        return MovementLimit(
            maximum_velocity_magnitude=movement_limit_dict['maximum_velocity_magnitude'],
            frame_step=movement_limit_dict['frame_step'],
            keypoint_classes=movement_limit_dict['keypoint_classes'],
            n_keypoints_required=movement_limit_dict['n_keypoints_required'],
            confidence_threshold=movement_limit_dict['confidence_threshold']
        )
    
    @override
    def to_dict(self) -> dict:
        '''
        Represent the PositionLimit as a dict
        
        Outputs:
        - the dict
        '''
        return super().to_dict() | {
            'vertices': str(self.patient_area),
            'frame_step': self.frame_step
        }
        
    @property
    def maximum_velocity_magnitude(self):
        return self._maximum_velocity_magnitude
    
    @property
    def frame_step(self):
        return self._frame_step