import numpy as np

def n_interactions(interactions: np.ndarray) -> int:
    '''
    Given an array of per-frame interactions (dtype bool), calculate the number of frames that are spent on patient interaction
    
    Inputs:
    - interactions: 1-dimensional array of boolean interactions per frame
    
    Outputs:
    - percentage of time spent on patient interaction
    '''
    assert len(interactions.shape) == 1, "interactions must have 1 dimension"
    return np.sum(interactions)
    
def interaction_ratio(interactions: np.ndarray) -> float:
    '''
    Given an array of per-frame interactions (dtype bool), calculate the percentage of time that is spent on patient interaction
    
    Inputs:
    - interactions: 1-dimensional array of boolean interactions per frame
    
    Outputs:
    - percentage of time spent on patient interaction
    '''
    return n_interactions(interactions)/len(interactions)
    
def interaction_ranges(interactions: np.ndarray) -> np.ndarray:
    '''
    Given an array of per-frame interactions (dtype bool), return frame ranges on which patient interaction occurred
    
    Inputs:
    - interactions: 1-dimensional array of boolean interactions per frame
    
    Outputs:
    - array of ranges where each entry contains [start, stop], start is inclusive and stop is exclusive
    '''
    assert len(interactions.shape) == 1, "interactions must have 1 dimension"
    
    ranges = []
    last_frame = -1
    for frame in np.nonzero(interactions)[0]:
        if last_frame >= 0 and frame > last_frame+1:
            ranges[-1].append(last_frame+1)
        if len(ranges)==0 or len(ranges[-1])==2:
            ranges.append([frame])
        last_frame = frame
    if len(ranges): ranges[-1].append(last_frame+1)
    
    return ranges

def summarize(interactions: np.ndarray) -> np.ndarray:
    '''
    Generate a str that summarizes an array of interaction
    
    Inputs:
    - interactions: 1-dimensional array of boolean interactions per frame
    
    Outputs:
    - summarizing str
    '''
    summary_str = 'True on {} of {} frames ({:.2f}%) over {} intervals'.format(
        n_interactions(interactions),
        len(interactions),
        100*interaction_ratio(interactions),
        len(interaction_ranges(interactions))
    )
    return summary_str