import sys
sys.path.append('.src')

import numpy as np

import det2d
import interaction as intr

def test_interaction():
    categories = det2d.read_categories('cats.json')
    tracklets  = det2d.read_tracklets('testing/test.det2d.json')
    human_tracklet = tracklets[categories.Human][0]
    
    position_limits = intr.Limits.from_file('testing/test.poslims.json', intr.PositionLimit)
    assert position_limits.limit_class==intr.PositionLimit, "position_limits.limit_class has the wrong type"
    assert position_limits.n_limits_required==2, "position_limits.n_limits_required has the wrong value"
    assert len(position_limits.limits)==3, "position_limits has the wrong number of limits"
    assert list(position_limits.limits[0].keypoint_classes)==[0,1], "position_limits.limits[0] has the wrong keypoint classes"
    assert position_limits.limits[0].n_keypoints_required==1, "position_limits.limits[0].n_keypoints_required has the wrong value"
    assert position_limits.limits[0].confidence_threshold==.3, "position_limits.limits[0].confidence_threshold has the wrong value"
    
    assert list(position_limits(human_tracklet))==[False, True, True, False, True], "position_limits returns wrong test values"
    
    movement_limits = intr.Limits.from_file('testing/test.movlims.json', intr.MovementLimit)
    assert movement_limits.limit_class==intr.MovementLimit, "movement_limits.limit_class has the wrong type"
    assert movement_limits.n_limits_required==2, "movement_limits.n_limits_required has the wrong value"
    assert len(movement_limits.limits)==2, "movement_limits has the wrong number of limits"
    assert list(movement_limits.limits[0].keypoint_classes)==[0,1,2], "movement_limits.limits[0] has the wrong keypoint classes"
    assert movement_limits.limits[0].n_keypoints_required==2, "movement_limits.limits[0].n_keypoints_required has the wrong value"
    assert movement_limits.limits[0].confidence_threshold==.3, "movement_limits.limits[0].confidence_threshold has the wrong value"
    
    assert list(movement_limits(human_tracklet))==[True, True, False, True, False], "movement_limits returns wrong test values"
    
    interaction_detector = intr.InteractionDetector(position_limits, movement_limits)
    interactions = interaction_detector(human_tracklet)
    assert list(interactions)==[False, True, False, False, False], "interaction_detector returns wrong test values"
    
    assert intr.n_interactions(interactions)==1, "n_interactions returns wrong value"
    assert intr.interaction_ratio(interactions)==1/5, "interaction_ratio returns wrong value"
    assert intr.interaction_ranges(interactions)==[[1,2]] and \
            intr.interaction_ranges(np.array([True, True, False, True, False]))==[[0,2],[3,4]], "interaction_ranges returns wrong value"
    assert intr.summarize(interactions)=='True on 1 of 5 frames (20.0%) over 1 intervals', "summarize returns wrong str"
    