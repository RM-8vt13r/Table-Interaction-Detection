from typing import override
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import transform
from det2d import Keys
from .limits import Limit

class PatientArea:
    def __init__(self, vertices: np.ndarray):
        '''
        An area within which a personnel member must be present to interact with the patient
        
        Inputs:
        - vertices: annotated vertices of the patient area, array of shape [A,2] where A is the number of vertices
        '''
        assert isinstance(vertices, np.ndarray), "vertices must be an array"
        assert len(vertices.shape)==2, "vertices must have two dimensions"
        assert vertices.shape[0]>=3, "vertices must have at least 3 vertices"
        assert vertices.shape[1]==2, "vertices second dimension must have length 2"
        self._area = Polygon(vertices)
    
    @classmethod
    def from_str(cls, area_str: str):
        '''
        Load a PatientArea from a str
        
        Inputs:
        - area_str: str encoding the area vertices
        '''
        vertices = area_str.split(';')
        vertices = np.array([vertex.split(',') for vertex in vertices], dtype=float)
        return PatientArea(vertices)
    
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        '''
        See includes()
        '''
        return self.includes(keypoints)
    
    def includes(self, keypoints: np.ndarray) -> np.ndarray:
        '''
        Check if keypoints in an array are within the patient area
        
        Inputs:
        - keypoints: the keypoints to test
        
        Outputs:
        - True where a keypoint was within the area, False elsewhere
        '''
        assert isinstance(keypoints, np.ndarray), "keypoints must be an array"
        assert keypoints.shape[-1]==2, "keypoints last dimension must have length 2"
        
        keypoints_shape = keypoints.shape
        keypoints = keypoints.reshape((-1,2))
        
        keypoints_in_patient_area = np.array([self.area.contains(Point(*keypoint)) for keypoint in keypoints])
        
        keypoints_in_patient_area = keypoints_in_patient_area.reshape(keypoints_shape[:-1])
        return keypoints_in_patient_area
    
    def __str__(self):
        '''
        Get a str representation of this area
        '''
        vertices = self.area.vertices ###########
        vertices = [','.join(vertex.astype(str)) for vertex in vertices]
        vertices = ';'.join(vertices)
        return vertices
    
    @property
    def area(self):
        return self._area

class PositionLimit(Limit):
    def __init__(self,
            patient_area: PatientArea,
            keypoint_classes: (tuple, list, np.ndarray),
            n_keypoints_required: int=1,
            confidence_threshold: float=0.0):
        '''
        A class representing a positioning limit within which a group of keypoints is considered to stand in a patient area
        
        Inputs:
        - patient_area: restriction on the area the keypoints can be in
        - keypoint_classes: tuple of keypoint classes to test patient_area on
        - n_keypoints_required: how many non-absent keypoints from keypoint_classes must be in the area for the group to be in the area, make -1 to require all classes to be in patient area
        - confidence_threshold: threshold below which a keypoint is said to be absent (and hence still)
        '''
        super().__init__(keypoint_classes, n_keypoints_required, confidence_threshold)
        self._patient_area = patient_area
    
    @override
    def tracklet_interacting(self, tracklet: dict) -> np.ndarray:
        '''
        See tracklet_inside()
        '''
        return self.tracklet_inside(tracklet)
        
    def tracklet_inside(self, tracklet: dict) -> np.ndarray:
        '''
        Evaluate whether keypoints satisfy PositionLimit
        
        Inputs:
        - tracklet: tracklet to test
        
        Outputs:
        - array of bools showing whether the tracklet satisfies this position limit
        '''
        keypoints = tracklet[Keys.keypoints][:,self.keypoint_classes,:]
        
        keypoints_present = keypoints[:,:,2] >= self.confidence_threshold
        keypoints_inside = self.patient_area.includes(keypoints[:,:,:2])
        
        in_area = np.sum(keypoints_present & keypoints_inside, axis=1) >= self.n_keypoints_required
        
        return in_area
    
    @classmethod
    @override
    def from_dict(cls, position_limit_dict: dict):
        '''
        Build a PositionLimit instance from a descriptive dictionary
        
        Inputs:
        - position_limit_dict: descriptive dictionary
        
        Outputs:
        - PositionLimit
        '''
        return PositionLimit(
            patient_area=PatientArea.from_str(position_limit_dict['vertices']),
            keypoint_classes=position_limit_dict['keypoint_classes'],
            n_keypoints_required=position_limit_dict['n_keypoints_required'],
            confidence_threshold=position_limit_dict['confidence_threshold']
        )
    
    @override
    def to_dict(self) -> dict:
        '''
        Represent the PositionLimit as a dict
        
        Outputs:
        - the dict
        '''
        return super().to_dict() | {
            'vertices': str(self.patient_area)
        }
    
    @property
    def patient_area(self):
        return self._patient_area