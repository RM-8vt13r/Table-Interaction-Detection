import sys
sys.path.append('src')
import os
from wcmatch import glob
import json
from tqdm import tqdm
import subprocess
from types import SimpleNamespace

from argparse import ArgumentParser
import configparser
import numpy as np
import scipy as sp
import det2d

from interaction import PositionLimit, MovementLimit, Limits, InteractionDetector

event_types = SimpleNamespace(**{
    'START': 'state start',
    'STOP': 'state stop'
})

subjects = SimpleNamespace(**{
    'STAFF': 'staflid',
    'SCRUBNURSE': 'scrub nurse',
    'ENVIRONMENT': 'omgeving'
})

phases = [None, 'inleiding anesthesie', 'chirurgische voorbereidingen', 'snijtijd', 'uitleiding anesthesie', 'patient kamer uit']

activities = SimpleNamespace(**{
    'ACTIVE': ['actief bij operatietafel', 'handeling bij operatietafel', 'handeling operatietafel', 'instrumenten aangeven', 'robot inpakken', 'iets terugleggen/aanpakken van chirurg', 'iets terugleggen/aanpakken chirurg'], # Staff, Scrub
    'IDLE':   ['actief elders', 'inactief', 'geen actie', 'instrumentenpakket uitpakken', 'kar rijden'], # Staff, Staff, Scrub, Scrub
    'ABSENT': ['absent', 'actie onbekend', 'afwezig'] # Staff, Scrub
})

def cli():
    parser = ArgumentParser("Make boxplots of personnel-patient interactions from a pose file over time")
    parser.add_argument('--configs', type=str, nargs='+', required=True, help="Path to the config file or a directory of config files")
    parser.add_argument('--window-length', type=int, default=7500, help="Length of the window in frames")
    parser.add_argument('--window-interval', type=int, default=7500, help="Distance between two window starts in frames")
    parser.add_argument('--welch-path', type=str, default=None, help="Path to save cross-config Welch t-test results. Leave None not to save those")
    return parser.parse_args()

def load_configs(config_paths: list) -> list:
    """
    Create a list of configurations from paths
    
    Inputs:
    - config_paths: list of configuration paths, leading to files or directories. In the latter case, all configurations from the directories are loaded recursively.
    
    Outputs:
    - list of configurations
    """
    configs = []
    for config_path in config_paths:
        if os.path.isdir(config_path):
            configs.extend(load_configs([os.path.join(config_path, path) for path in os.listdir(config_path) if path[-4:]=='.ini']))
            continue
        new_config = configparser.ConfigParser()
        new_config.read(config_path)
        configs.append(new_config)
    
    return configs

def get_paths(config: configparser.ConfigParser) -> (str, str, str, list, str):
    """
    Load all paths necessary to execute the script for a configuration
    
    Inputs:
    - config: the configuration
    
    Outputs:
    - Path to summary file to write
    - Path to root where all pose detections can be found
    - Path to root where all noldus annotations can be found
    - Path to all video files that are included
    - Path to movement limits parameter file to use
    """
    summary_path = config.get('PATHS', 'summary_boxplots')
    assert os.path.splitext(summary_path)[1].lower() in ('.txt'), "summary path must be a .txt file"
    if not os.path.isdir(os.path.split(summary_path)[0]): os.makedirs(os.path.split(summary_path)[0])
    
    det2d_root = config.get('PATHS', 'det2d')
    det2d_names = os.listdir(det2d_root)
    noldus_root = config.get('PATHS', 'noldus')
    noldus_names = os.listdir(noldus_root)
    data_root = config.get('PATHS', 'data')
    video_unparsed_paths = json.loads(data_root)
    video_parsed_paths = [video_parsed_path for video_unparsed_path in video_unparsed_paths for video_parsed_path in glob.glob(video_unparsed_paths, flags=glob.EXTGLOB)]
    video_parsed_paths = [
        video_parsed_path for video_parsed_path in video_parsed_paths \
        if video_parsed_path[-4:]=='.mp4' \
        and video_parsed_path[:-4]+'.poslims.json' in video_parsed_paths \
        and os.path.split(video_parsed_path[:-4]+'.det2d.json')[1] in det2d_names \
        and 'RESA - ' + os.path.split(video_parsed_path[:-4]+' - Event Logs.txt')[1] in noldus_names
    ]
    movlims_path = config.get('PATHS', 'movlims')
    
    return summary_path, det2d_root, noldus_root, video_parsed_paths, movlims_path

def get_video_paths(det2d_root: str, noldus_root: str, video_parsed_path: str) -> (str, str, str):
    """
    Load paths corresponding to a single video
    
    Inputs:
    - det2d_root: path to root where all pose detections can be found
    - noldus_root: path to root where all noldus annotations can be found
    - video_parsed_path: path to video file to load corresponding pose detections and annotations of
    
    Outputs:
    - Path to the pose detection file corresponding to the video
    - Path to the noldus annotation file corresponding to the video
    - Path to the position limits parameter file corresponding to the video
    """
    video_root, video_name = os.path.split(video_parsed_path)
    video_name = os.path.splitext(video_name)[0]
    det2d_path = os.path.join(det2d_root, video_name+'.det2d.json')
    noldus_path = os.path.join(noldus_root, 'RESA - ' + video_name + ' - Event Logs.txt')
    poslims_path = os.path.join(video_root, video_name+'.poslims.json')
    if not os.path.isfile(det2d_path) or not os.path.isfile(poslims_path): return
    return det2d_path, noldus_path, poslims_path
    
def get_frame_timestamps_s(video_path: str) -> np.ndarray:
    """
    Given a video, obtain the timestamps in seconds corresponding to each frame
    
    Inputs:
    - video_path: path to the video from which to extract frame timestamps
    
    Outputs:
    - Array of timestamps in seconds
    """
    ffprobe_command = [
        'ffprobe',
        '-loglevel', 'quiet',
        '-output_format', 'csv',
        '-show_entries', 'packet=pts_time',
        video_path
    ]
    print(f"Executing ffprobe on {video_path} (might take a while)")
    out,_ = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return np.array([float(line.split(',')[1].strip()) for line in out.decode().split('\n') if len(line)])

def get_noldus_phases(noldus_path: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Load annotated phases and activities from a Noldus file
        
        Inputs:
        - noldus_path: path to the noldus annotation file to read
        
        Outputs:
        - Timestamps in seconds corresponding to entries in the following outputs
        - Annotated phase index at every timestamp in the first output array. Phase indices are relative to the global variable 'phases'
        - The number of persons annotated as being active at every timestamp in the first output array. 'Active' activities are classified using the global variable 'activities'
        - The number of persons annotated as being idle at every timestamp in the first output array. 'Idle' activities are classified using the global variable 'activities'
        - The number of persons annotated as being absent at every timestamp in the first output array. 'Absent' activities are classified using the global variable 'activities'
        """
        with open(noldus_path, 'r', encoding='utf-16-le') as f:
            lines = f.readlines()
        
        header = [json.loads(s[1:] if i==0 else s) for i,s in enumerate(lines[0].split(';')[:-1])]
        header[0] = header[0][1:]
        
        start_seconds   = [0]
        phase_sequence  = [0] # Start the sequence with None at 0 seconds
        active_sequence = [0] # Start sequence with no interaction at 0 seconds
        idle_sequence   = [0]
        absent_sequence = [0]
        for line_index, line in enumerate(lines[1:]):
            tokens = [json.loads(token) for token in line.split(';')[:-1]]
            line_dict = {k: v.lower() if isinstance(v, str) else v for k, v in zip(header, tokens)}
            
            if line_dict['Time_Relative_sf'] == '': continue
            timestamp = line_dict['Time_Relative_sf']
            
            if line_dict['Subject'] == subjects.ENVIRONMENT and line_dict['Behavior'] in phases: # Only do anything if the line describes a phase transition
                if timestamp not in start_seconds:
                    start_seconds.append(timestamp)
                    active_sequence.append(active_sequence[-1])
                    idle_sequence.append(idle_sequence[-1])
                    absent_sequence.append(absent_sequence[-1])
                    phase_sequence.append(phase_sequence[-1]) #phases.index(line_dict['Behavior'].lower()) if line_dict['Event_Type'] == event_types.START else 0)
                
                # elif line_dict['Event_Type'] == event_types.START:
                    # phase_sequence[-1] = phases.index(line_dict['Behavior'].lower())
                phase_sequence[-1] = phases.index(None) if line_dict['Event_Type'] == event_types.STOP else phases.index(line_dict['Behavior'].lower())
                    
            elif line_dict['Subject'].startswith(subjects.STAFF) or line_dict['Subject'].startswith(subjects.SCRUBNURSE):
                if line_dict['Behavior'] in ('', 'deur ingang beweegt'): continue
                assert line_dict['Behavior'] in activities.ACTIVE + activities.IDLE + activities.ABSENT, f"Unknown staff/nurse activity in file '{file_name}' at relative timestamp {line_dict['Time_Relative_sf']}: '{line_dict['Behavior']}'"
                if timestamp not in start_seconds:
                    start_seconds.append(timestamp)
                    phase_sequence.append(phase_sequence[-1])
                    active_sequence.append(active_sequence[-1]) # + (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.ACTIVE))
                    idle_sequence.append(idle_sequence[-1]) #     + (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.IDLE))
                    absent_sequence.append(absent_sequence[-1]) # + (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.ABSENT))
                
                # else:
                    # active_sequence[-1] += (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.ACTIVE)
                    # idle_sequence[-1]   += (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.IDLE)
                    # absent_sequence[-1] += (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.ABSENT)
                active_sequence[-1] += (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.ACTIVE)
                idle_sequence[-1]   += (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.IDLE)
                absent_sequence[-1] += (-1 if line_dict['Event_Type'] == event_types.STOP else 1)*(line_dict['Behavior'] in activities.ABSENT)
                
        sort_indices    = np.argsort(start_seconds)
        start_seconds   = np.array(start_seconds)[sort_indices]
        phase_sequence  = np.array(phase_sequence)[sort_indices]
        active_sequence = np.array(active_sequence)[sort_indices]
        idle_sequence   = np.array(idle_sequence)[sort_indices]
        absent_sequence = np.array(absent_sequence)[sort_indices]
        
        return start_seconds, phase_sequence, active_sequence, idle_sequence, absent_sequence
    
def get_phases_per_window(window_ends_s: np.ndarray, phase_sequence: np.ndarray, phase_start_seconds: np.ndarray) -> np.ndarray:
    """
    Assign a phase to each of a sequence of windows
    
    Inputs:
    - window_ends_s: the timestamp of the final frame of each window in seconds
    - phase_sequence: Annotated phase indices relative to the global variable 'phases'
    - phase_start_seconds: Timestamps in seconds corresponding to the entries in phase_sequence
    
    Outputs:
    - The phase index per window in window_ends_s
    """
    phase_indices = np.maximum(np.searchsorted(phase_start_seconds, window_ends_s, 'right')-1, 0)
    phases_per_window = phase_sequence[phase_indices]
    return phases_per_window
    
def get_n_interactions_per_window(video_index: int, detection_loader: det2d.DetectionLoader, interaction_detector: InteractionDetector) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    For each of a sequence of windows, detect the total number of interactions with the table and movements
    
    Inputs:
    - video_index: numerical identifier of the video the windows were loaded from, for logging purposes
    - detection_loader: iterable that loads or contains pose detections
    - interaction_detector: class that can classify interaction from positional- and movement limits
    
    Outputs:
    - The total number of detected interactions between personnel and the table per window
    - The total number of detected non-interactions between personnel and the table per window
    - The total number of detected movements of personnel per window
    """
    movement_limits = interaction_detector.movement_limits
    n_interactions_per_window = np.zeros((len(detection_loader),), dtype=int)
    n_non_interactions_per_window = np.zeros((len(detection_loader),), dtype=int)
    n_movements_per_window = np.zeros_like(n_interactions_per_window)
    for window_index, detections in tqdm(enumerate(detection_loader), desc=f"det2d file {video_index+1}", total=len(detection_loader)):
        tracklets = det2d.detections2tracklets(detections)
        human_tracklets = tracklets[categories.Human]
        
        window_interactions = np.zeros(shape=(len(human_tracklets),), dtype=int)
        window_non_interactions = np.zeros_like(window_interactions)
        window_movements = np.zeros_like(window_interactions)
        for tracklet_index, (ID, tracklet) in enumerate(human_tracklets.items()):
            interactions = interaction_detector(tracklet)
            window_interactions[tracklet_index] = np.sum(interactions)
            window_non_interactions[tracklet_index] = np.sum(~interactions)
            movements = 1-movement_limits(tracklet)
            window_movements[tracklet_index] = np.sum(movements)
            
        n_interactions_per_window[window_index] = np.sum(window_interactions)
        n_non_interactions_per_window[window_index] = np.sum(window_non_interactions)
        n_movements_per_window[window_index] = np.sum(window_movements)
        
    return n_interactions_per_window, n_non_interactions_per_window, n_movements_per_window

def get_noldus_activity_s_per_window(noldus_timestamps_s: np.ndarray, noldus_active_counts: np.ndarray, noldus_idle_counts: np.ndarray, window_starts_s: np.ndarray, window_ends_s: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    For each of a sequence of windows, retrieve the total number of annotated interactions with the table and movements
    
    Inputs:
    - noldus_timestamps_s: a series of annotation timestamps in seconds
    - noldus_active_counts: The number of persons that was active starting at each timestamp in noldus_timestamps_s. 'Active' activities are classified using the global variable 'activities'.
    - noldus_idle_counts: The number of persons that was idle starting at each timestamp in noldus_timestamps_s. 'Idle' activities are classified using the global variable 'activities'.
    - window_starts_s: the timestamp of the first frame of each window in seconds
    - window_ends_s: the timestamp of the final frame of each window in seconds
    
    Outputs:
    - The total number of seconds per window during which personnel was active
    - The total number of seconds per window during which personnel was idle
    """
    active_s_per_window = np.zeros((len(window_starts_s),), dtype=float)
    idle_s_per_window = np.zeros_like(active_s_per_window)
    for window_index, (window_start, window_end) in enumerate(zip(window_starts_s, window_ends_s)):
        window_active_s, window_idle_s = 0, 0
        for timestamp_index in np.arange(*np.searchsorted(noldus_timestamps_s, [window_start, window_end], 'right'))-1:
            section_start = np.maximum(noldus_timestamps_s[timestamp_index] if timestamp_index >= 0 else -np.inf, window_start)
            section_end   = np.minimum(noldus_timestamps_s[timestamp_index+1] if timestamp_index+1 < len(noldus_timestamps_s) else np.inf, window_end)
            section_duration = section_end-section_start
            if timestamp_index >= 0:
                timestamp_index_clipped = min(timestamp_index, len(noldus_timestamps_s)-1)
                window_active_s += noldus_active_counts[timestamp_index_clipped]*section_duration
                window_idle_s   += noldus_idle_counts[timestamp_index_clipped]*section_duration
        
        active_s_per_window[window_index] = window_active_s
        idle_s_per_window[window_index]   = window_idle_s
    
    return active_s_per_window, idle_s_per_window

def get_five_number_summary(numbers: np.ndarray) -> (float, float, float, float, float, np.ndarray):
    """
    Calculate the five-number-summary of a sequence of measurements to make a box plot
    
    Inputs:
    - numbers: the numbers to summarize
    
    Outputs:
    - The lowest number that was not an outlier
    - The first quantile
    - The median
    - The third quantile
    - The highest number that was not an outlier
    - All outliers
    """
    if len(numbers) == 0:
        return 0, 0, 0, 0, 0, np.array([], dtype=float)
    
    Q1 = np.quantile(numbers, .25)
    Q2 = np.median(numbers)
    Q3 = np.quantile(numbers, .75)
    
    IQR = Q3-Q1
    Q0 = np.min(numbers[numbers >= Q1 - 1.5*IQR])
    Q4 = np.max(numbers[numbers <= Q3 + 1.5*IQR])
    
    outliers = numbers[(numbers < Q0) | (numbers > Q4)]
    
    return Q0, Q1, Q2, Q3, Q4, outliers

def get_interaction_percentages_per_window(n_movements_per_window: np.ndarray, n_interactions_per_window: np.ndarray, n_non_interactions_per_window: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the total percentage of interaction with the table and movements per window
    
    Inputs:
    - n_movements_per_window: the total number of movements in each window
    - n_interactions_per_window: the total number of interactions in each window
    - n_non_interactions_per_window: the total number of non-interacting detections in each window
    
    Outputs:
    - The percentage of movement in each window
    - The percentage of interaction with the table in each window
    - The percentage of non-interaction with the table in each window
    """
    movement_percentages_per_window = np.zeros(shape=n_movements_per_window.shape, dtype=float)
    interaction_percentages_per_window = np.zeros_like(movement_percentages_per_window)
    non_interaction_percentages_per_window = np.zeros_like(movement_percentages_per_window)

    n_present_frames_per_window = n_interactions_per_window + n_non_interactions_per_window
    
    movement_percentages_per_window[n_present_frames_per_window > 0] = n_movements_per_window[n_present_frames_per_window > 0]/n_present_frames_per_window[n_present_frames_per_window > 0]*100
    interaction_percentages_per_window[n_present_frames_per_window > 0] = n_interactions_per_window[n_present_frames_per_window > 0]/n_present_frames_per_window[n_present_frames_per_window > 0]*100
    non_interaction_percentages_per_window[n_present_frames_per_window > 0] = n_non_interactions_per_window[n_present_frames_per_window > 0]/n_present_frames_per_window[n_present_frames_per_window > 0]*100
    
    return movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window

def get_active_percentages_per_window(active_s_per_window: np.ndarray, idle_s_per_window: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the total percentage of annotated activity per window
    
    Inputs:
    - active_s_per_window: the number of seconds during which persons were active, times the number of visible active persons per second
    - idle_s_per_window: the number of seconds during which persons were idle, times the number of visible active persons per second
    
    Outputs:
    - The percentage of 'active' activity per window
    """
    present_s_per_window = active_s_per_window + idle_s_per_window
    active_percentages_per_window = np.divide(active_s_per_window, present_s_per_window, out=np.zeros_like(active_s_per_window, dtype=float), where=present_s_per_window > 0)*100
    idle_percentages_per_window = np.divide(idle_s_per_window, present_s_per_window, out=np.zeros_like(idle_s_per_window, dtype=float), where=present_s_per_window > 0)*100
    return active_percentages_per_window, idle_percentages_per_window

def get_detection_percentages_per_window(window_length: int, n_interactions_per_window: np.ndarray, n_non_interactions_per_window: np.ndarray, window_starts_s: np.ndarray, window_ends_s: np.ndarray, active_s_per_window: np.ndarray, idle_s_per_window: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the ratio between the number of detected poses and annotated poses per window
    
    Inputs:
    - window_length: total number of frames per window
    - n_interactions_per_window: the total number of pose detections (x frames) that interact with the operating table per window
    - n_non_interactions_per_window: the total number of pose detections (x frames) that don't interact with the operating table per window
    - window_starts_s: window start timestamps in seconds
    - window_ends_s: window end timestamps in seconds
    - active_s_per_window: the total number of pose annotations (x seconds) that interact with the operating table per window
    - idle_s_per_window: the total number of pose annotations (x seconds) that don't interact with the operating table per window
    
    Outputs:
    - Number of detected persons per frame in each window
    - Number of annotated persons per frame in each window
    - Ratio between detected and annotated poses in each window
    - Number of detected persons per frame in each window who interact with the operating table
    - Number of annotated persons per frame in each window who interact with the operating table
    - Ratio between detected and annotated poses in each window who interact with the operating table
    - Number of detected persons per frame in each window who don't interact with the operating table
    - Number of annotated persons per frame in each window who don't interact with the operating table
    - Ratio between detected and annotated poses in each window who don't interact with the operating table
    """
    n_total_detections_per_frame_per_window  = (n_interactions_per_window + n_non_interactions_per_window)/window_length
    n_total_annotations_per_frame_per_window = (active_s_per_window + idle_s_per_window)/(window_ends_s-window_starts_s)
    total_detection_percentages_per_window   = np.divide(n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, out=np.ones_like(n_total_detections_per_frame_per_window, dtype=float), where=n_total_annotations_per_frame_per_window > 0)*100
    
    n_interacting_detections_per_frame_per_window = n_interactions_per_window/window_length
    n_active_annotations_per_frame_per_window     = active_s_per_window/(window_ends_s-window_starts_s)
    interacting_detection_percentages_per_window  = np.divide(n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, out=np.ones_like(n_interacting_detections_per_frame_per_window, dtype=float), where=n_active_annotations_per_frame_per_window > 0)*100
    
    n_non_interacting_detections_per_frame_per_window = n_non_interactions_per_window/window_length
    n_idle_annotations_per_frame_per_window           = idle_s_per_window/(window_ends_s-window_starts_s)
    non_interacting_detection_percentages_per_window  = np.divide(n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, out=np.ones_like(n_non_interacting_detections_per_frame_per_window, dtype=float), where=n_idle_annotations_per_frame_per_window > 0)*100
    
    return n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, total_detection_percentages_per_window,\
        n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, interacting_detection_percentages_per_window,\
        n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, non_interacting_detection_percentages_per_window

def welch_test(samples1: np.ndarray, samples2: np.ndarray) -> (float, float, float):
    """
    Apply a Welch's t-test to two samples
    
    Inputs:
    - samples1: a collection of samples
    - samples2: another collection of samples
    
    Outputs:
    - The statistic t
    - The statistic degrees of freedom
    - The p-value
    """
    N1 = np.size(samples1)
    N2 = np.size(samples2)
    
    if N1 <= 1 or N2 <= 1: return np.inf, np.inf, 0
    
    mean1 = np.mean(samples1)
    mean2 = np.mean(samples2)
    dmean = mean1-mean2
    
    squared_standard_error1 = np.var(samples1, ddof=1)/N1
    squared_standard_error2 = np.var(samples2, ddof=1)/N2
    
    if squared_standard_error1 == 0 and squared_standard_error2 == 0: return np.inf, np.inf, 0
    
    t_numerator = mean1-mean2
    t_denominator = np.sqrt(squared_standard_error1 + squared_standard_error2)
    t = t_numerator/t_denominator
    
    degrees_of_freedom_numerator = (squared_standard_error1 + squared_standard_error2)**2
    degrees_of_freedom_denominator = squared_standard_error1**2/(N1-1) + squared_standard_error2**2/(N2-1)
    degrees_of_freedom = degrees_of_freedom_numerator/degrees_of_freedom_denominator
    
    p = sp.stats.t.sf(x=t, df=degrees_of_freedom)
    
    return t, degrees_of_freedom, p
    
def write_video_header(det2d_path: str, summary_filehandle):
    """
    Write a header to a summary file to signal that a new video will be processed
    
    Inputs:
    - det2d_path: the path to the pose detections that will be analysed
    - summary_filehandle: filehandle of the open summary file to write to
    """
    summary_filehandle.write(f"Next det2d file: {det2d_path}\n")

def write_config_header(summary_filehandle):
    """
    Write a header to a summary file to signal that an entire configuration will be summarized
    
    Inputs:
    - summary_filehandle: filehandle of the open summary file to write to
    """
    summary_filehandle.write(f"Total:\n")
    
def write_interaction_over_time(window_phases: np.ndarray, window_ends_s: np.ndarray, n_movements_per_window: np.ndarray, n_interactions_per_window: np.ndarray, n_non_interactions_per_window: np.ndarray, summary_filehandle) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Write the interaction over a number of windows to a summary file
    
    Inputs:
    - window_phases: the workflow phase per window
    - window_ends_s: the end timestamp in seconds per window
    - n_movements_per_window: the total number of movements per window
    - n_interactions_per_window: the total number of pose detections (x frames) that interact with the operating table per window
    - n_non_interactions_per_window: the total number of pose detections (x frames) that don't interact with the operating table per window
    - summary_filehandle: filehandle of the open summary file to write to
    
    Outputs:
    - The percentage of movement in each window
    - The percentage of interaction with the table in each window
    """
    n_present_frames_per_window = n_interactions_per_window + n_non_interactions_per_window
    
    summary_filehandle.write(f"Window phases: " + " ".join([f"{phase}" for phase in window_phases]) + "\n")
    summary_filehandle.write(f"Window total frames: " + " ".join([f"({window_end},{n_frames})" for window_end, n_frames in zip(window_ends_s, n_present_frames_per_window)]) + "\n")
    summary_filehandle.write(f"Window movement frames: " + " ".join([f"({window_end},{n_movements})" for window_end, n_movements in zip(window_ends_s, n_movements_per_window)]) + "\n")
    summary_filehandle.write(f"Window interaction frames: " + " ".join([f"({window_end},{n_interactions})" for window_end, n_interactions in zip(window_ends_s, n_interactions_per_window)]) + "\n")
    summary_filehandle.write(f"Window non-interaction frames: " + " ".join([f"({window_end},{n_non_interactions})" for window_end, n_non_interactions in zip(window_ends_s, n_non_interactions_per_window)]) + "\n")
    
    movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window = get_interaction_percentages_per_window(n_movements_per_window, n_interactions_per_window, n_non_interactions_per_window)
    summary_filehandle.write(f"Window movement frames (percent): " + " ".join([f"({window_end},{ratio:.2f})" for window_end, ratio in zip(window_ends_s, movement_percentages_per_window)]) + "\n")
    summary_filehandle.write(f"Window interaction frames (percent): " + " ".join([f"({window_end},{ratio:.2f})" for window_end, ratio in zip(window_ends_s, interaction_percentages_per_window)]) + "\n")
    summary_filehandle.write(f"Window non-interaction frames (percent): " + " ".join([f"({window_end},{ratio:.2f})" for window_end, ratio in zip(window_ends_s, non_interaction_percentages_per_window)]) + "\n")
    
    summary_filehandle.write(
        "\n".join([f"Window movement frames (percent, phase '{phase}'): " + 
            " ".join([
                f"({window_end},{ratio:.2f})"
                for window_end, ratio in zip(window_ends_s[window_phases == phase_index], movement_percentages_per_window[window_phases == phase_index])
            ])
            for phase_index, phase in enumerate(phases)
        ]) + "\n"
    )
    
    summary_filehandle.write(
        "\n".join([f"Window interaction frames (percent, phase '{phase}'): " + 
            " ".join([
                f"({window_end},{ratio:.2f})"
                for window_end, ratio in zip(window_ends_s[window_phases == phase_index], interaction_percentages_per_window[window_phases == phase_index])
            ])
            for phase_index, phase in enumerate(phases)
        ]) + "\n"
    )
    
    summary_filehandle.write(
        "\n".join([f"Window non-interaction frames (percent, phase '{phase}'): " + 
            " ".join([
                f"({window_end},{ratio:.2f})"
                for window_end, ratio in zip(window_ends_s[window_phases == phase_index], non_interaction_percentages_per_window[window_phases == phase_index])
            ])
            for phase_index, phase in enumerate(phases)
        ]) + "\n"
    )
    
    return movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window
    
def write_noldus_activity_over_time(window_phases: np.ndarray, window_ends_s: np.ndarray, active_s_per_window: np.ndarray, idle_s_per_window: np.ndarray, summary_filehandle) -> (np.ndarray, np.ndarray):
    """
    Write annotated activity over a number of windows to a summary file
    
    Inputs:
    - window_phases: the workflow phase per window
    - window_ends_s: the end timestamp in seconds per window
    - active_s_per_window: the total number of seconds of 'active' activity per window
    - idle_s_per_window: the total number of seconds of 'idle' activity per window
    - summary_filehandle: filehandle of the open summary file to write to
    
    Outputs:
    - The percentage of activity in each window
    """
    present_s_per_window = active_s_per_window + idle_s_per_window
    
    summary_filehandle.write(f"Window total seconds: " + " ".join([f"({window_end},{window_s})" for window_end, window_s in zip(window_ends_s, present_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window active seconds: " + " ".join([f"({window_end},{active_s})" for window_end, active_s in zip(window_ends_s, active_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window idle seconds: " + " ".join([f"({window_end},{idle_s})" for window_end, idle_s in zip(window_ends_s, idle_s_per_window)]) + "\n")
    
    active_percentages_per_window, idle_percentages_per_window = get_active_percentages_per_window(active_s_per_window, idle_s_per_window)
    summary_filehandle.write(f"Window active seconds (percent): " + " ".join([f"({window_end},{ratio:.2f})" for window_end, ratio in zip(window_ends_s, active_percentages_per_window)]) + "\n")
    summary_filehandle.write(f"Window idle seconds (percent): " + " ".join([f"({window_end},{ratio:.2f})" for window_end, ratio in zip(window_ends_s, idle_percentages_per_window)]) + "\n")
    
    summary_filehandle.write(
        "\n".join([f"Window active seconds (percent, phase '{phase}'): " +
            " ".join([
                f"({window_end},{ratio:.2f})"
                for window_end, ratio in zip(window_ends_s[window_phases == phase_index], active_percentages_per_window[window_phases == phase_index])
            ])
            for phase_index, phase in enumerate(phases)
        ]) + "\n"
    )
    
    summary_filehandle.write(
        "\n".join([f"Window idle seconds (percent, phase '{phase}'): " +
            " ".join([
                f"({window_end},{ratio:.2f})"
                for window_end, ratio in zip(window_ends_s[window_phases == phase_index], idle_percentages_per_window[window_phases == phase_index])
            ])
            for phase_index, phase in enumerate(phases)
        ]) + "\n"
    )
    
    return active_percentages_per_window, idle_percentages_per_window
    
def write_pose_counts_over_time(window_phases: np.ndarray, window_starts_s: np.ndarray, window_ends_s: np.ndarray, window_length: int, n_interactions_per_window: np.ndarray, n_non_interactions_per_window: np.ndarray, active_s_per_window: np.ndarray, idle_s_per_window: np.ndarray, summary_filehandle) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Plot detected and annotated pose counts, and their ratio.
    
    Inputs:
    - window_phases: the workflow phase per window
    - window_starts_s: the start timestamp in seconds per window
    - window_ends_s: the end timestamp in seconds per window
    - window_length: the window length in frames
    - n_interactions_per_window: the total number of pose detections (x frames) that interact with the operating table per window
    - n_non_interactions_per_window: the total number of pose detections (x frames) that don't interact with the operating table per window
    - active_s_per_window: the total number of pose annotations (x seconds) that interact with the operating table per window
    - idle_s_per_window: the total number of pose annotations (x seconds) that don't interact with the operating table per window
    - summary filehandle: filehandle of the open summary file to write to
    
    Outputs:
    - Number of detected persons per frame in each window
    - Number of annotated persons per frame in each window
    - Ratio between detected and annotated poses in each window
    - Number of detected persons per frame in each window who interact with the operating table
    - Number of annotated persons per frame in each window who interact with the operating table
    - Ratio between detected and annotated poses in each window who interact with the operating table
    - Number of detected persons per frame in each window who don't interact with the operating table
    - Number of annotated persons per frame in each window who don't interact with the operating table
    - Ratio between detected and annotated poses in each window who don't interact with the operating table
    """
    summary_filehandle.write(f"Window annotated poses per frame: " + " ".join([f"({window_end},{window_annotations/(window_end-window_start)})" for window_start, window_end, window_annotations in zip(window_starts_s, window_ends_s, active_s_per_window + idle_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window detected poses per frame: " + " ".join([f"({window_end},{window_detections/window_length})" for window_end, window_detections in zip(window_ends_s, n_interactions_per_window + n_non_interactions_per_window)]) + "\n")
    summary_filehandle.write(f"Window detected poses (percent): " + " ".join([f"({window_end},{np.divide(window_detections, window_annotations, out=np.ones_like(window_detections, dtype=float), where=window_annotations > 0)*100})" for window_end, window_detections, window_annotations in zip(window_ends_s, n_interactions_per_window + n_non_interactions_per_window, active_s_per_window + idle_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window annotated interacting poses per frame: " + " ".join([f"({window_end},{window_annotations/(window_end-window_start)})" for window_start, window_end, window_annotations in zip(window_starts_s, window_ends_s, active_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window detected interacting poses per frame: " + " ".join([f"({window_end},{window_detections/window_length})" for window_end, window_detections in zip(window_ends_s, n_interactions_per_window)]) + "\n")
    summary_filehandle.write(f"Window detected interacting poses (percent): " + " ".join([f"({window_end},{np.divide(window_detections, window_annotations, out=np.ones_like(window_detections, dtype=float), where=window_annotations > 0)*100})" for window_end, window_detections, window_annotations in zip(window_ends_s, n_interactions_per_window, active_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window annotated idle poses per frame: " + " ".join([f"({window_end},{window_annotations/(window_end-window_start)})" for window_start, window_end, window_annotations in zip(window_starts_s, window_ends_s, idle_s_per_window)]) + "\n")
    summary_filehandle.write(f"Window detected idle poses per frame: " + " ".join([f"({window_end},{window_detections/window_length})" for window_end, window_detections in zip(window_ends_s, n_non_interactions_per_window)]) + "\n")
    summary_filehandle.write(f"Window detected idle poses (percent): " + " ".join([f"({window_end},{np.divide(window_detections, window_annotations, out=np.ones_like(window_detections, dtype=float), where=window_annotations > 0)*100})" for window_end, window_detections, window_annotations in zip(window_ends_s, n_non_interactions_per_window, idle_s_per_window)]) + "\n")
    
    n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, total_detection_percentages_per_window,\
        n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, interacting_detection_percentages_per_window,\
        n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, non_interacting_detection_percentages_per_window = \
        get_detection_percentages_per_window(window_length, n_interactions_per_window, n_non_interactions_per_window, window_starts_s, window_ends_s, active_s_per_window, idle_s_per_window)
    
    summary_filehandle.write(
        "\n".join(
            [f"Window annotated poses per frame (phase {phase}): " +
                " ".join([
                    f"({window_end},{window_annotations/(window_end-window_start)})"
                    for window_start, window_end, window_annotations in zip(window_starts_s[window_phases == phase_index], window_ends_s[window_phases == phase_index], n_total_annotations_per_frame_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window detected poses per frame (phase '{phase}'): " +
                " ".join([
                    f"({window_end},{window_detections/window_length})"
                    for window_end, window_detections in zip(window_ends_s[window_phases == phase_index], n_total_detections_per_frame_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window detected poses per frame (percent, phase '{phase}'): " +
                " ".join([
                    f"({window_end},{detection_percentages})"
                    for window_end, detection_percentages in zip(window_ends_s[window_phases == phase_index], total_detection_percentages_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window annotated interacting poses per frame (phase {phase}): " +
                " ".join([
                    f"({window_end},{window_annotations/(window_end-window_start)})"
                    for window_start, window_end, window_annotations in zip(window_starts_s[window_phases == phase_index], window_ends_s[window_phases == phase_index], n_active_annotations_per_frame_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window detected interacting poses per frame (phase '{phase}'): " +
                " ".join([
                    f"({window_end},{window_detections/window_length})"
                    for window_end, window_detections in zip(window_ends_s[window_phases == phase_index], n_interacting_detections_per_frame_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window detected interacting poses per frame (percent, phase '{phase}'): " +
                " ".join([
                    f"({window_end},{detection_percentages})"
                    for window_end, detection_percentages in zip(window_ends_s[window_phases == phase_index], interacting_detection_percentages_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +[f"Window annotated idle poses per frame (phase {phase}): " +
                " ".join([
                    f"({window_end},{window_annotations/(window_end-window_start)})"
                    for window_start, window_end, window_annotations in zip(window_starts_s[window_phases == phase_index], window_ends_s[window_phases == phase_index], n_idle_annotations_per_frame_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window detected idle poses per frame (phase '{phase}'): " +
                " ".join([
                    f"({window_end},{window_detections/window_length})"
                    for window_end, window_detections in zip(window_ends_s[window_phases == phase_index], n_non_interacting_detections_per_frame_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ] +
            [f"Window detected idle poses per frame (percent, phase '{phase}'): " +
                " ".join([
                    f"({window_end},{detection_percentages})"
                    for window_end, detection_percentages in zip(window_ends_s[window_phases == phase_index], non_interacting_detection_percentages_per_window[window_phases == phase_index])
                ])
                for phase_index, phase in enumerate(phases)
            ]) + "\n"
        )
    
    return n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, total_detection_percentages_per_window,\
        n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, interacting_detection_percentages_per_window,\
        n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, non_interacting_detection_percentages_per_window
    
def write_interaction_boxplots(window_phases: np.ndarray, movement_percentages_per_window: np.ndarray, interaction_percentages_per_window: np.ndarray, non_interaction_percentages_per_window: np.ndarray, summary_filehandle):
    """
    Write a LaTeX boxplot summarizing interaction over a number of windows to a summary file
    
    Inputs:
    - window_phases: the workflow phase per window
    - movement_percentages_per_window: the percentage of movement in each window
    - interaction_percentages_per_window: the percentage of interaction in each window
    - non_interaction_percentages_per_window: the percentage of non-interaction in each window
    - summary_filehandle: filehandle of the open summary file to write to
    """
    movement_summary = get_five_number_summary(movement_percentages_per_window)
    movement_summary_without_none = get_five_number_summary(movement_percentages_per_window[window_phases != phases.index(None)])
    movement_summary_without_none_and_leave = get_five_number_summary(movement_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    movement_summary_per_phase = [get_five_number_summary(movement_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("Movement boxplots:\n")
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [movement_summary, movement_summary_without_none, movement_summary_without_none_and_leave] + movement_summary_per_phase)
    ]) + "\n")
    
    summary_filehandle.write("Interaction boxplots:\n")
    interaction_summary = get_five_number_summary(interaction_percentages_per_window)
    interaction_summary_without_none = get_five_number_summary(interaction_percentages_per_window[window_phases != phases.index(None)])
    interaction_summary_without_none_and_leave = get_five_number_summary(interaction_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    interaction_summary_per_phase = [get_five_number_summary(interaction_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [interaction_summary, interaction_summary_without_none, interaction_summary_without_none_and_leave] + interaction_summary_per_phase)
    ]) + "\n")
    
    summary_filehandle.write("Non-interaction boxplots:\n")
    non_interaction_summary = get_five_number_summary(non_interaction_percentages_per_window)
    non_interaction_summary_without_none = get_five_number_summary(non_interaction_percentages_per_window[window_phases != phases.index(None)])
    non_interaction_summary_without_none_and_leave = get_five_number_summary(non_interaction_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    non_interaction_summary_per_phase = [get_five_number_summary(non_interaction_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [non_interaction_summary, non_interaction_summary_without_none, non_interaction_summary_without_none_and_leave] + non_interaction_summary_per_phase)
    ]) + "\n")

def write_noldus_activity_boxplots(window_phases: np.ndarray, active_percentages_per_window: np.ndarray, idle_percentages_per_window: np.ndarray, summary_filehandle):
    """
    Write a LaTeX boxplot summarizing annotated 'active' activity over a number of windows to a summary file
    
    Inputs:
    - window_phases: the workflow phase per window
    - active_percentages_per_window: the percentage of 'active' activity in each window
    - summary_filehandle: filehandle of the open summary file to write to
    """
    active_summary = get_five_number_summary(active_percentages_per_window)
    active_summary_without_none = get_five_number_summary(active_percentages_per_window[window_phases != phases.index(None)])
    active_summary_without_none_and_leave = get_five_number_summary(active_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    active_summary_per_phase = [get_five_number_summary(active_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("Noldus activity boxplots:\n")
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [active_summary, active_summary_without_none, active_summary_without_none_and_leave] + active_summary_per_phase)
    ]) + "\n")
    
    idle_summary = get_five_number_summary(idle_percentages_per_window)
    idle_summary_without_none = get_five_number_summary(idle_percentages_per_window[window_phases != phases.index(None)])
    idle_summary_without_none_and_leave = get_five_number_summary(idle_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    idle_summary_per_phase = [get_five_number_summary(idle_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("Noldus idle boxplots:\n")
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [idle_summary, idle_summary_without_none, idle_summary_without_none_and_leave] + idle_summary_per_phase)
    ]) + "\n")

def write_pose_count_boxplots(window_phases: np.ndarray, total_detection_percentages_per_window: np.ndarray, interacting_detection_percentages_per_window: np.ndarray, non_interacting_detection_percentages_per_window: np.ndarray, summary_filehandle):
    """
    Write a LaTeX boxplot summarizing detected poses per frame divided by annotated poses per frame, over a number of windows, to a summary file
    
    Inputs:
    - window_phases: the workflow phase per window
    - total_detection_percentages_per_window: ratio between detected and annotated poses per frame in each window, x100
    - interacting_detection_percentages_per_window: ratio between detected and annotated interacting poses per frame in each window, x100
    - non_interacting_detection_percentages_per_window: ratio between detected and annotated non-interacting poses per frame in each window, x100
    - summary_filehandle: filehandle of the open summary file to write to
    """
    total_detection_summary = get_five_number_summary(total_detection_percentages_per_window)
    total_detection_summary_without_none = get_five_number_summary(total_detection_percentages_per_window[window_phases != phases.index(None)])
    total_detection_summary_without_none_and_leave = get_five_number_summary(total_detection_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    total_detection_summary_per_phase = [get_five_number_summary(total_detection_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("Detected/annotated total poses boxplots:\n")
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [total_detection_summary, total_detection_summary_without_none, total_detection_summary_without_none_and_leave] + total_detection_summary_per_phase)
    ]) + "\n")
    
    interacting_detection_summary = get_five_number_summary(interacting_detection_percentages_per_window)
    interacting_detection_summary_without_none = get_five_number_summary(interacting_detection_percentages_per_window[window_phases != phases.index(None)])
    interacting_detection_summary_without_none_and_leave = get_five_number_summary(interacting_detection_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    interacting_detection_summary_per_phase = [get_five_number_summary(interacting_detection_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("Detected/annotated interacting poses boxplots:\n")
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [interacting_detection_summary, interacting_detection_summary_without_none, interacting_detection_summary_without_none_and_leave] + interacting_detection_summary_per_phase)
    ]) + "\n")
    
    non_interacting_detection_summary = get_five_number_summary(non_interacting_detection_percentages_per_window)
    non_interacting_detection_summary_without_none = get_five_number_summary(non_interacting_detection_percentages_per_window[window_phases != phases.index(None)])
    non_interacting_detection_summary_without_none_and_leave = get_five_number_summary(non_interacting_detection_percentages_per_window[(window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit'))])
    non_interacting_detection_summary_per_phase = [get_five_number_summary(non_interacting_detection_percentages_per_window[window_phases == phase_index]) for phase_index in range(len(phases))]
    summary_filehandle.write("Detected/annotated non-interacting poses boxplots:\n")
    summary_filehandle.write("\n".join([
        "\\addplot [name={}, fill=Gray, mark=*, boxplot prepared={{lower whisker={},lower quartile={},median={},upper quartile={},upper whisker={}}}] coordinates {{{}}};".format(name, *summary[:5], ' '.join([f'({0},{outlier})' for outlier in summary[5]]))
        for name, summary in zip(["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases, [non_interacting_detection_summary, non_interacting_detection_summary_without_none, non_interacting_detection_summary_without_none_and_leave] + non_interacting_detection_summary_per_phase)
    ]) + "\n")

def write_interaction_correlation(window_phases: np.ndarray, interaction_percentages_per_window: np.ndarray, active_percentages_per_window: np.ndarray, summary_filehandle):
    """
    Plot measured and annotated interaction against each other, and calculate a correlation efficient between the two (per phase, and over all phases jointly)
    
    Inputs:
    - window_phases: the workflow phase per window
    - interaction_percentages_per_window: the percentage of interaction in each window
    - active_percentages_per_window: the percentage of 'active' activity in each window
    - summary_filehandle: filehandle of the open summary file to write to
    """
    summary_filehandle.write("Annotated/detected interaction correlation:\n")
    names = ["Totaal", "Zonder 'None'", "zonder 'None' \\& 'Patient kamer uit'"] + phases
    active_percentages = []
    interaction_percentages = []
    for name in names:
        if name == "Totaal":
            active_percentages.append(active_percentages_per_window)
            interaction_percentages.append(interaction_percentages_per_window)
        else:
            if name == "Zonder 'None'": indices = np.where(window_phases != phases.index(None))
            elif name == "zonder 'None' \\& 'Patient kamer uit'": indices = np.where((window_phases != phases.index(None)) & (window_phases != phases.index('patient kamer uit')))
            else: indices = np.where(window_phases == phases.index(name))
            active_percentages.append(active_percentages_per_window[indices])
            interaction_percentages.append(interaction_percentages_per_window[indices])
    
    summary_filehandle.write("\n".join(["\\addplot [name={},mark=*,mark options={{Gray}},draw=none] coordinates {{{}}};".format(name, ' '.join([f'({a},{i})' for a, i in zip(active, interaction)])) for name, active, interaction in zip(names, active_percentages, interaction_percentages)]) + "\n")
    summary_filehandle.write("\n".join(["Measured/annotated interaction Pearson correlation coefficient ({}): {}".format(name, np.corrcoef(active, interaction)[0,1] if len(active) > 1 and np.std(active) > 0 and np.std(interaction) > 0 else '-') for name, active, interaction in zip(names, active_percentages, interaction_percentages)]) + "\n")
    summary_filehandle.write("\n".join(["Measured/annotated interaction Spearman correlation coefficient ({}): {}".format(name, np.corrcoef(np.argsort(active), np.argsort(interaction))[0,1] if len(active) > 1 and np.std(active) > 0 and np.std(interaction) > 0 else '-') for name, active, interaction in zip(names, active_percentages, interaction_percentages)]) + "\n")
    
def write_interaction_statistics(window_labels: np.ndarray, movement_percentages_per_window: np.ndarray, interaction_percentages_per_window: np.ndarray, active_percentages_per_window: np.ndarray, total_detection_percentages_per_window: np.ndarray, interacting_detection_percentages_per_window: np.ndarray, non_interacting_detection_percentages_per_window: np.ndarray, summary_filehandle):
    """
    Write p-values between interaction, movement and annotated activity in different phases to a summary file
    
    Inputs:
    - window_labels: a label to assign each window to a population
    - movement_percentages_per_window: the percentage of movement in each window
    - interaction_percentages_per_window: the percentage of interaction in each window
    - active_percentages_per_window: the percentage of 'active' activity in each window
    - total_detection_percentages_per_window: the number of detected poses divided by the number of annotated poses per frame in each window
    - interacting_detection_percentages_per_window: the number of detected interacting poses divided by the number of interacting annotated poses per frame in each window
    - non_interacting_detection_percentages_per_window: the number of detected non-interacting poses divided by the number of non-interacting annotated poses per frame in each window
    - summary_filehandle: filehandle of the open summary file to write to
    """
    unique_labels = np.unique(window_labels)
    for label_index1, label1 in enumerate(unique_labels[:-1]):
        _,_,interaction_vs_active_p_value = welch_test(interaction_percentages_per_window[window_labels == label1], active_percentages_per_window[window_labels == label1])
        summary_filehandle.write(f"Welch p-value between interactions and Noldus activity ('{label1}'): {interaction_vs_active_p_value}\n")
        for label_index2, label2 in enumerate(unique_labels[label_index1+1:]):
            _,_,movement_p_value = welch_test(movement_percentages_per_window[window_labels == label1], movement_percentages_per_window[window_labels == label2])
            _,_,interaction_p_value = welch_test(interaction_percentages_per_window[window_labels == label1], interaction_percentages_per_window[window_labels == label2])
            _,_,active_p_value = welch_test(active_percentages_per_window[window_labels == label1], active_percentages_per_window[window_labels == label2])
            _,_,total_detection_p_value = welch_test(total_detection_percentages_per_window[window_labels == label1], total_detection_percentages_per_window[window_labels == label2])
            _,_,interacting_detection_p_value = welch_test(interacting_detection_percentages_per_window[window_labels == label1], interacting_detection_percentages_per_window[window_labels == label2])
            _,_,non_interacting_detection_p_value = welch_test(non_interacting_detection_percentages_per_window[window_labels == label1], non_interacting_detection_percentages_per_window[window_labels == label2])
            summary_filehandle.write(f"Welch p-value between '{label1}' and '{label2}' movements: {movement_p_value}\n")
            summary_filehandle.write(f"Welch p-value between '{label1}' and '{label2}' interactions: {interaction_p_value}\n")
            summary_filehandle.write(f"Welch p-value between '{label1}' and '{label2}' Noldus activity: {active_p_value}\n")
            summary_filehandle.write(f"Welch p-value between '{label1}' and '{label2}' Total detection/annotation ratio: {total_detection_p_value}\n")
            summary_filehandle.write(f"Welch p-value between '{label1}' and '{label2}' Interacting detection/annotation ratio: {interacting_detection_p_value}\n")
            summary_filehandle.write(f"Welch p-value between '{label1}' and '{label2}' Non-interacting detection/annotation ratio: {non_interacting_detection_p_value}\n")

def write_video_summary(det2d_path: str, window_length: int, window_phases: np.ndarray, window_starts_s: np.ndarray, window_ends_s: np.ndarray, n_movements_per_window: np.ndarray, n_interactions_per_window: np.ndarray, n_non_interactions_per_window: np.ndarray, active_s_per_window: np.ndarray, idle_s_per_window: np.ndarray, summary_filehandle):
    """
    Extract statistics and write a summary for a single video
    
    Inputs:
    - det2d_path: the path to the pose detections that will be analysed
    - window_length: the length of a window in frames
    - window_phases: the workflow phase per window
    - window_starts_s: the start timestamp in seconds per window
    - window_ends_s: the end timestamp in seconds per window
    - n_movements_per_window: the total number of movements per window
    - n_interactions_per_window: the total number of pose detections (x frames) that interact with the operating table per window
    - n_non_interactions_per_window: the total number of pose detections (x frames) that don't interact with the operating table per window
    - active_s_per_window: the total number of pose annotations (x seconds) that interact with the operating table per window
    - idle_s_per_window: the total number of pose annotations (x seconds) that don't interact with the operating table per window
    - summary_filehandle: filehandle of the open summary file to write to
    """
    write_video_header(det2d_path, summary_filehandle)
    
    movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window = write_interaction_over_time(window_phases, window_ends_s, n_movements_per_window, n_interactions_per_window, n_non_interactions_per_window, summary_filehandle)
    active_percentages_per_window, idle_percentages_per_window = write_noldus_activity_over_time(window_phases, window_ends_s, active_s_per_window, idle_s_per_window, summary_filehandle)
    n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, total_detection_percentages_per_window,\
        n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, interacting_detection_percentages_per_window,\
        n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, non_interacting_detection_percentages_per_window = write_pose_counts_over_time(window_phases, window_starts_s, window_ends_s, window_length, n_interactions_per_window, n_non_interactions_per_window, active_s_per_window, idle_s_per_window, summary_filehandle)
    
    write_interaction_boxplots(window_phases, movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window, summary_filehandle)
    write_noldus_activity_boxplots(window_phases, active_percentages_per_window, idle_percentages_per_window, summary_filehandle)
    write_pose_count_boxplots(window_phases, total_detection_percentages_per_window, interacting_detection_percentages_per_window, non_interacting_detection_percentages_per_window, summary_filehandle)
    write_interaction_correlation(window_phases, interaction_percentages_per_window, active_percentages_per_window, summary_filehandle)
    write_interaction_statistics(np.array(phases, dtype=str)[window_phases], movement_percentages_per_window, interaction_percentages_per_window, active_percentages_per_window, total_detection_percentages_per_window, interacting_detection_percentages_per_window, non_interacting_detection_percentages_per_window, summary_filehandle)
    
    summary_filehandle.write(f"\n")
    
def write_config_summary(window_length: int, window_phases: np.ndarray, window_starts_s: np.ndarray, window_ends_s: np.ndarray, n_movements_per_window: np.ndarray, n_interactions_per_window: np.ndarray, n_non_interactions_per_window: np.ndarray, active_s_per_window: np.ndarray, idle_s_per_window: np.ndarray, summary_filehandle):
    """
    Extract statistics and write a summary for a single configuration
    
    Inputs:
    - window_length: the length of a window in frames
    - window_phases: the workflow phase per window
    - window_starts_s: the start timestamp in seconds per window
    - window_ends_s: the end timestamp in seconds per window
    - n_movements_per_window: the total number of movements per window
    - n_interactions_per_window: the total number of pose detections (x frames) that interact with the operating table per window
    - n_non_interactions_per_window: the total number of pose detections (x frames) that don't interact with the operating table per window
    - active_s_per_window: the total number of pose annotations (x seconds) that interact with the operating table per window
    - idle_s_per_window: the total number of pose annotations (x seconds) that don't interact with the operating table per window
    - summary_filehandle: filehandle of the open summary file to write to
    """
    write_config_header(summary_filehandle)
    
    movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window = get_interaction_percentages_per_window(n_movements_per_window, n_interactions_per_window, n_non_interactions_per_window)
    active_percentages_per_window, idle_percentages_per_window = get_active_percentages_per_window(active_s_per_window, idle_s_per_window)
    n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, total_detection_percentages_per_window,\
        n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, interacting_detection_percentages_per_window,\
        n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, non_interacting_detection_percentages_per_window = get_detection_percentages_per_window(window_length, n_interactions_per_window, n_non_interactions_per_window, window_starts_s, window_ends_s, active_s_per_window, idle_s_per_window)
    
    write_interaction_boxplots(window_phases, movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window, summary_filehandle)
    write_noldus_activity_boxplots(window_phases, active_percentages_per_window, idle_percentages_per_window, summary_filehandle)
    write_pose_count_boxplots(window_phases, total_detection_percentages_per_window, interacting_detection_percentages_per_window, non_interacting_detection_percentages_per_window, summary_filehandle)
    write_interaction_correlation(window_phases, interaction_percentages_per_window, active_percentages_per_window, summary_filehandle)
    write_interaction_statistics(np.array(phases, dtype=str)[window_phases], movement_percentages_per_window, interaction_percentages_per_window, active_percentages_per_window, total_detection_percentages_per_window, interacting_detection_percentages_per_window, non_interacting_detection_percentages_per_window, summary_filehandle)
    
    summary_filehandle.write(f"\n")
    
def process_video(det2d_root: str, noldus_root: str, video_index: int, video_parsed_path: str, movement_limits: Limits, window_length: int, window_interval: int, summary_filehandle) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Analyse personnel interaction with the table, movement and annotated activity for a range of windows and write it to a summary for a single video
    
    Inputs:
    - det2d_root: path to root where all pose detections can be found
    - noldus_root: path to root where all noldus annotations can be found
    - video_index: numerical identifier of the video the windows were loaded from, for logging purposes
    - video_parsed_path: path to video file to load corresponding pose detections and annotations of
    - movement_limits: limits to calculate whether a person is moving or not
    - window_length: the length of a window in frames
    - window_interval: the distance between two window start frames
    - summary_filehandle: filehandle of the open summary file to write to
    
    Outputs:
    - The number of detected interactions per window
    - The number of detected movements per window
    - The number of frames times the number of people present per window
    - The number of seconds of annotated 'active' activity per window
    - The number of seconds times the number of people present per window
    - The annotated phase per window
    """
    frame_timestamps_s = get_frame_timestamps_s(video_parsed_path)
    det2d_path, noldus_path, poslims_path = get_video_paths(det2d_root, noldus_root, video_parsed_path)
    
    position_limits = Limits.from_file(poslims_path, PositionLimit)
    interaction_detector = InteractionDetector(position_limits, movement_limits)
    
    detection_loader = det2d.DetectionLoader(det2d_path, window_length=window_length, window_interval=window_interval)
    
    noldus_timestamps_s, noldus_phases, noldus_active_counts, noldus_idle_counts, noldus_absent_counts = get_noldus_phases(noldus_path)
    
    window_starts_frames = np.arange(len(detection_loader))*args.window_interval
    window_starts_s      = frame_timestamps_s[window_starts_frames]
    window_ends_frames   = np.minimum(window_starts_frames + args.window_length, len(frame_timestamps_s)-1)
    window_ends_s        = frame_timestamps_s[window_ends_frames]
    window_phases        = get_phases_per_window(window_ends_s, noldus_phases, noldus_timestamps_s)
    n_interactions_per_window, n_non_interactions_per_window, n_movements_per_window = get_n_interactions_per_window(video_index, detection_loader, interaction_detector)
    active_s_per_window, idle_s_per_window = get_noldus_activity_s_per_window(noldus_timestamps_s, noldus_active_counts, noldus_idle_counts, window_starts_s, window_ends_s)
    
    write_video_summary(det2d_path, window_length, window_phases, window_starts_s, window_ends_s, n_movements_per_window, n_interactions_per_window, n_non_interactions_per_window, active_s_per_window, idle_s_per_window, summary_filehandle)
    
    return window_starts_s, window_ends_s, window_phases, n_interactions_per_window, n_non_interactions_per_window, n_movements_per_window, active_s_per_window, idle_s_per_window
    
def process_config(config, window_length: int, window_interval: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Analyse personnel interaction with the table, movement and annotated activity for a range of windows and write it to a summary for a configuration and all videos it contains
    
    Inputs:
    - config: the configuration
    - window_length: the length of a window in frames
    - window_interval: the distance between two window start frames
    
    Outputs:
    - The number of detected interactions per window
    - The number of detected movements per window
    - The number of frames times the number of people present per window
    - The number of seconds of annotated 'active' activity per window
    - The number of seconds times the number of people present per window
    - The annotated phase per window
    """
    summary_path, det2d_root, noldus_root, video_parsed_paths, movlims_path = get_paths(config)
    if not os.path.isfile(movlims_path): return
    movement_limits = Limits.from_file(movlims_path, MovementLimit)
    summary_filehandle = open(summary_path, 'w')
    
    window_starts_s = np.zeros(shape=(0,), dtype=float)
    window_ends_s = np.zeros_like(window_starts_s)
    window_phases = np.zeros_like(window_starts_s, dtype=int)
    n_interactions_per_window = np.zeros_like(window_starts_s, dtype=int)
    n_non_interactions_per_window = np.zeros_like(window_starts_s, dtype=int)
    n_movements_per_window = np.zeros_like(window_starts_s, dtype=int)
    active_s_per_window = np.zeros_like(window_starts_s)
    idle_s_per_window = np.zeros_like(window_starts_s)
    for video_index, video_parsed_path in enumerate(video_parsed_paths):
        measurements = process_video(det2d_root, noldus_root, video_index, video_parsed_path, movement_limits, window_length, window_interval, summary_filehandle)
        window_starts_s = np.append(window_starts_s, measurements[0])
        window_ends_s = np.append(window_ends_s, measurements[1])
        window_phases = np.append(window_phases, measurements[2])
        n_interactions_per_window = np.append(n_interactions_per_window, measurements[3])
        n_non_interactions_per_window = np.append(n_non_interactions_per_window, measurements[4])
        n_movements_per_window = np.append(n_movements_per_window, measurements[5])
        active_s_per_window = np.append(active_s_per_window, measurements[6])
        idle_s_per_window = np.append(idle_s_per_window, measurements[7])
        
    write_config_summary(window_length, window_phases, window_starts_s, window_ends_s, n_movements_per_window, n_interactions_per_window, n_non_interactions_per_window, active_s_per_window, idle_s_per_window, summary_filehandle)
    summary_filehandle.close()
    
    return window_starts_s, window_ends_s, window_phases, n_interactions_per_window, n_non_interactions_per_window, n_movements_per_window, active_s_per_window, idle_s_per_window

if __name__=="__main__":
    args = cli()
    configs = load_configs(args.configs)
    
    categories = det2d.read_categories('cats.json')
    
    window_starts_s = np.zeros(shape=(0,), dtype=float)
    window_ends_s = np.zeros_like(window_starts_s)
    window_labels = np.zeros_like(window_starts_s, dtype=int)
    n_interactions_per_window = np.zeros_like(window_starts_s, dtype=int)
    n_non_interactions_per_window = np.zeros_like(window_starts_s, dtype=int)
    n_movements_per_window = np.zeros_like(window_starts_s, dtype=int)
    active_s_per_window = np.zeros_like(window_starts_s)
    idle_s_per_window = np.zeros_like(window_starts_s)
    for config_path, config in zip(args.configs, configs):
        procedure_type = os.path.splitext(os.path.split(config_path)[1])[0]
        
        measurements = process_config(config, args.window_length, args.window_interval)
        window_starts_s = np.append(window_starts_s, measurements[0])
        window_ends_s = np.append(window_ends_s, measurements[1])
        window_labels = np.append(window_labels, [f'{procedure_type}+{phases[phase_index]}' for phase_index in measurements[2]])
        n_interactions_per_window = np.append(n_interactions_per_window, measurements[3])
        n_non_interactions_per_window = np.append(n_non_interactions_per_window, measurements[4])
        n_movements_per_window = np.append(n_movements_per_window, measurements[5])
        active_s_per_window = np.append(active_s_per_window, measurements[6])
        idle_s_per_window = np.append(idle_s_per_window, measurements[7])
        
    if args.welch_path is not None:
        print("Performing cross-config t-tests")
        with open(args.welch_path, 'w') as f:
            movement_percentages_per_window, interaction_percentages_per_window, non_interaction_percentages_per_window = get_interaction_percentages_per_window(n_movements_per_window, n_interactions_per_window, n_non_interactions_per_window)
            active_percentages_per_window, idle_percentages_per_window = get_active_percentages_per_window(active_s_per_window, idle_s_per_window)
            n_total_detections_per_frame_per_window, n_total_annotations_per_frame_per_window, total_detection_percentages_per_window,\
                n_interacting_detections_per_frame_per_window, n_active_annotations_per_frame_per_window, interacting_detection_percentages_per_window,\
                n_non_interacting_detections_per_frame_per_window, n_idle_annotations_per_frame_per_window, non_interacting_detection_percentages_per_window = get_detection_percentages_per_window(args.window_length, n_interactions_per_window, n_non_interactions_per_window, window_starts_s, window_ends_s, active_s_per_window, idle_s_per_window)
            
            write_interaction_statistics(
                window_labels,
                movement_percentages_per_window,
                interaction_percentages_per_window,
                active_percentages_per_window,
                total_detection_percentages_per_window,
                interacting_detection_percentages_per_window,
                non_interacting_detection_percentages_per_window,
                f
            )
            f.write('\n')
            
            f.write('Between procedure types (all phases):')
            window_procedures = np.array([label.split('+')[0] for label in window_labels])
            write_interaction_statistics(
                window_procedures,
                movement_percentages_per_window,
                interaction_percentages_per_window,
                active_percentages_per_window,
                total_detection_percentages_per_window,
                interacting_detection_percentages_per_window,
                non_interacting_detection_percentages_per_window,
                f
            )
            f.write('\n')
            
            f.write('Between procedure types (all phases except \'None\'):')
            window_phases = np.array([label.split('+')[1].lower() for label in window_labels])
            window_mask = window_phases != 'none'
            window_procedures = np.array([label.split('+')[0] for label in window_labels])
            write_interaction_statistics(
                window_procedures[window_mask],
                movement_percentages_per_window[window_mask],
                interaction_percentages_per_window[window_mask],
                active_percentages_per_window[window_mask],
                total_detection_percentages_per_window[window_mask],
                interacting_detection_percentages_per_window[window_mask],
                non_interacting_detection_percentages_per_window[window_mask],
                f
            )
            f.write('\n')
            
            f.write('Between procedure types (all phases except \'None\' and \'Patient kamer uit\'):')
            window_mask &= (window_phases != 'patient kamer uit')
            write_interaction_statistics(
                window_procedures[window_mask],
                movement_percentages_per_window[window_mask],
                interaction_percentages_per_window[window_mask],
                active_percentages_per_window[window_mask],
                total_detection_percentages_per_window[window_mask],
                interacting_detection_percentages_per_window[window_mask],
                non_interacting_detection_percentages_per_window[window_mask],
                f
            )
            f.write('\n')
            
            f.write('Between phases (all procedure types):')
            window_phases = np.array([label.split('+')[1] for label in window_labels])
            write_interaction_statistics(
                window_phases,
                movement_percentages_per_window,
                interaction_percentages_per_window,
                active_percentages_per_window,
                total_detection_percentages_per_window,
                interacting_detection_percentages_per_window,
                non_interacting_detection_percentages_per_window,
                f
            )
