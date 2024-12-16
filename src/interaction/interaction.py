import numpy as np
from .limits import Limits

class InteractionDetector:
    def __init__(self, position_limits: Limits, movement_limits: Limits):
        '''
        Class to detect interaction with the patient from a set of constraints
        
        Inputs:
        - position_limits: position limits to satisfy for interaction
        - movement_limits: movement limits to satisfy for interaction
        '''
        self._position_limits = position_limits
        self._movement_limits = movement_limits
    
    def __call__(self, tracklet: dict) -> np.ndarray:
        '''
        See detect_interaction()
        '''
        return self.detect_interaction(tracklet)
    
    def detect_interaction(self, tracklet: dict) -> np.ndarray:
        '''
        Detect when a personnel tracklet interacts with the patient
        
        Inputs:
        - tracklet: personnel tracklet
        
        Outputs:
        - array which is True on frames where the personnel interacts with the patient and False elsewhere
        '''
        return self.position_limits(tracklet) & self.movement_limits(tracklet)
    
    @property
    def position_limits(self):
        return self._position_limits
    
    @property
    def movement_limits(self):
        return self._movement_limits