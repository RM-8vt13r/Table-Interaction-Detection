from abc import ABC, abstractmethod
import json
import numpy as np
from det2d import Keys
from coordinates import cartesian2polar

class Limit(ABC):
    def __init__(self,
            keypoint_classes: (tuple, list, np.ndarray),
            n_keypoints_required: int=-1,
            confidence_threshold: float=0.0
        ):
        '''
        A class representing a limit within which a group of keypoints is considered to interact with the patient
        
        Inputs:
        - keypoint_classes: tuple of keypoint classes to test on
        - n_keypoints_required: how many non-absent keypoints from keypoint_classes must satisfy this limit for interaction, make -1 to require all classes to be in patient area
        - confidence_threshold: threshold below which a keypoint is said to be absent
        '''
        assert len(keypoint_classes), "keypoint_classes can't be empty"
        assert confidence_threshold >= 0 and confidence_threshold <= 1, "confidence_threshold must be between 0 and 1 (inclusive)"
        assert n_keypoints_required >= 1 and n_keypoints_required <= len(keypoint_classes), "n_keypoints_required must be between 1 and len(keypoint_classes) (inclusive)"
        
        self._keypoint_classes = np.array(keypoint_classes)
        self._n_keypoints_required = n_keypoints_required
        self._confidence_threshold = confidence_threshold
    
    def __call__(self, tracklet: dict) -> np.array:
        '''
        See keypoints_interacting()
        '''
        return self.tracklet_interacting(tracklet)
        
    @abstractmethod
    def tracklet_interacting(self, tracklet: dict) -> np.ndarray:
        '''
        Detect whether a tracklet is interacting with the patient or not
        
        Inputs:
        - tracklet: the personnel tracklet
        
        Outputs:
        - Array with bools showing whether the keypoints were interacting (True) or not (False) per frame
        '''
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, limit_dict: dict):
        '''
        Build a Limit instance from a descriptive dictionary
        
        Inputs:
        - limit_dict: descriptive dictionary
        
        Outputs:
        - Limit
        '''
        return Limit(
            limit_dict['keypoint_classes'],
            limit_dict['n_keypoints_required'],
            limit_dict['confidence_threshold']
        )
    
    def to_dict(self) -> dict:
        '''
        Represent the Limit as a dict
        
        Outputs:
        - the dict
        '''
        return {
            'keypoint_classes': self.keypoint_classes,
            'n_keypoints_required': self.n_keypoints_required,
            'confidence_threshold': self.confidence_threshold
        }
    
    @property
    def keypoint_classes(self):
        return self._keypoint_classes
    
    @property
    def n_keypoints_required(self):
        return self._n_keypoints_required
    
    @property
    def confidence_threshold(self):
        return self._confidence_threshold

class Limits:
    def __init__(self, limits: (tuple, list, np.ndarray), n_limits_required: int):
        '''
        A class representing limits within which a person is considered to interact with the patient
        
        Inputs:
        - limits: list of restrictions a tracklet should adhere to to be interacting
        - n_limits_required: how many of limits must be satisfied for the group to be satisfied
        '''
        assert n_limits_required >= 1 and n_limits_required <= len(limits), "n_limits_required must be at least 1 and at most the length of limits"
        self._limits = limits
        self._n_limits_required = n_limits_required
        self._limit_class = type(limits[0])
    
    def __call__(self, tracklet: dict) -> np.array:
        '''
        See tracklet_interacting()
        '''
        return self.tracklet_interacting(tracklet)
        
    def tracklet_interacting(self, tracklet: dict) -> np.ndarray:
        '''
        Detect whether someone is interacting with the patient or not; only the case if enough individual Limits are satisfied
        
        Inputs:
        - tracklet: the personnel tracklet
        
        Outputs:
        - Array with bools showing whether the tracklet was interacting (True) or not (False) per frame
        '''
        limits_satisfied = [limit(tracklet) for limit in self.limits]
        tracklet_interacting = np.sum(limits_satisfied, axis=0) >= self.n_limits_required
        return tracklet_interacting
    
    @classmethod
    def from_file(cls, path: str, limit_class: type):
        '''
        Build a limits instance from a .json file
        
        Inputs:
        - path: path to dictionary file
        - limit_class: the type of limit
        
        Outputs:
        - Limits
        '''
        with open(path, 'r') as f:
            limits_dictionary = json.load(f)
        return cls.from_dict(limits_dictionary, limit_class)
        
    @classmethod
    def from_dict(cls, limits_dict: dict, limit_class: type):
        '''
        Build a Limits instance from a descriptive dictionary
        
        Inputs:
        - limits_dict: descriptive dictionary
        - limit_class: the type of Limit
        
        Outputs:
        - Limits
        '''
        return Limits(
            limits=[limit_class.from_dict(d) for d in limits_dict['limits']],
            n_limits_required=limits_dict['n_limits_required'],
        )
    
    def to_dict(self) -> dict:
        '''
        Represent the Limits as a dict
        
        Outputs:
        - the dict
        '''
        return {
            'n_limits_required': self.n_limits_required,
            'limits': [limit.to_dict() for limit in self.limits]
        }
        
    @property
    def limits(self):
        return self._limits
    
    @property
    def n_limits_required(self):
        return self._n_limits_required
    
    @property
    def limit_class(self):
        return self._limit_class
    
    @property
    def limits(self):
        return self._limits