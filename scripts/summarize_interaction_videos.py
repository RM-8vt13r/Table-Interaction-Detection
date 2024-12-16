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
import det2d

from interaction import PositionLimit, MovementLimit, Limits, InteractionDetector, summarize, interaction_ratio

event_types = SimpleNamespace(**{
    'START': 'State start',
    'STOP': 'State stop'
})
    
subjects = SimpleNamespace(**{
    'ENVIRONMENT': 'Omgeving'
})

phases = [None, 'Inleiding anesthesie', 'Chirurgische voorbereidingen', 'Snijtijd', 'Uitleiding anesthesie', 'Patient kamer uit']

def cli():
    parser = ArgumentParser("Quantify personnel-patient interactions from a pose file")
    parser.add_argument('--configs', type=str, nargs='+', required=True, help="Path to the config file or a directory of config files")
    return parser.parse_args()
    
def get_paths(config: configparser.ConfigParser) -> (str, list):
    summary_path = config.get('PATHS', 'summary_videos')
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

def get_video_paths(det2d_root, noldus_root, video_parsed_path):
    video_root, video_name = os.path.split(video_parsed_path)
    video_name = os.path.splitext(video_name)[0]
    det2d_path = os.path.join(det2d_root, video_name+'.det2d.json')
    noldus_path = os.path.join(noldus_root, 'RESA - ' + video_name + ' - Event Logs.txt')
    poslims_path = os.path.join(video_root, video_name+'.poslims.json')
    if not os.path.isfile(det2d_path) or not os.path.isfile(poslims_path): return
    return det2d_path, noldus_path, poslims_path
    
def load_configs(config_paths):
    configs = []
    for config_path in config_paths:
        if os.path.isdir(config_path):
            configs.extend(load_configs([os.path.join(config_path, path) for path in os.listdir(config_path) if path[-4:]=='.ini']))
            continue
        new_config = configparser.ConfigParser()
        new_config.read(config_path)
        configs.append(new_config)
    
    return configs

def get_frame_timestamps_s(video_path):
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
        Load annotated phases from a Noldus file
        
        Inputs:
        - noldus_path: path to the noldus annotation file to read
        
        Outputs:
        - Timestamps in seconds corresponding to entries in the following outputs
        - Annotated phase index at every timestamp in the first output array. Phase indices are relative to the global variable 'phases'
        """
        with open(noldus_path, 'r', encoding='utf-16-le') as f:
            lines = f.readlines()
        
        header = [json.loads(s[1:] if i==0 else s) for i,s in enumerate(lines[0].split(';')[:-1])]
        header[0] = header[0][1:]
        
        start_seconds   = [0]
        phase_sequence  = [0] # Start the sequence with None at 0 seconds
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
            
        sort_indices    = np.argsort(start_seconds)
        start_seconds   = np.array(start_seconds)[sort_indices]
        phase_sequence  = np.array(phase_sequence)[sort_indices]
        
        return start_seconds, phase_sequence

def write_video_header(det2d_path: str, summary_filehandle):
    summary_filehandle.write(f"Next det2d file: {det2d_path}\n")

def write_config_header(summary_filehandle):
    summary_filehandle.write(f"Total:\n")
    
def write_interaction(n_frames_per_phase: np.ndarray, n_movements_per_phase: np.ndarray, n_interactions_per_phase: np.ndarray, summary_filehandle):
    summary_filehandle.write("\n".join([f"Procedure movement ({phase}): True on {movements} of {frames} frames ({movements/frames*100 if frames > 0 else 0:.2f}%)" for phase, movements, frames in zip(phases, n_movements_per_phase, n_frames_per_phase)]) + "\n")
    summary_filehandle.write(f"Procedure movement (total): True on {np.sum(n_movements_per_phase)} of {np.sum(n_frames_per_phase)} frames ({np.sum(n_movements_per_phase)/np.sum(n_frames_per_phase)*100 if np.sum(n_frames_per_phase) > 0 else 0:.2f}%)\n")
    summary_filehandle.write("\n".join([f"Procedure interaction ({phase}): True on {interactions} of {frames} frames ({interactions/frames*100 if frames > 0 else 0:.2f}%)" for phase, interactions, frames in zip(phases, n_interactions_per_phase, n_frames_per_phase)]) + "\n")
    summary_filehandle.write(f"Procedure interaction (total): True on {np.sum(n_interactions_per_phase)} of {np.sum(n_frames_per_phase)} frames ({np.sum(n_interactions_per_phase)/np.sum(n_frames_per_phase)*100 if np.sum(n_frames_per_phase) > 0 else 0:.2f}%)\n")
    summary_filehandle.write("\n")
    
def process_video(det2d_root: str, noldus_root: str, video_index: int, video_parsed_path: str, movement_limits: Limits, summary_filehandle):
    frame_timestamps_s = get_frame_timestamps_s(video_parsed_path)
    det2d_path, noldus_path, poslims_path = get_video_paths(det2d_root, noldus_root, video_parsed_path)
    
    position_limits = Limits.from_file(poslims_path, PositionLimit)
    interaction_detector = InteractionDetector(position_limits, movement_limits)
    
    # detection_loader = det2d.DetectionLoader(det2d_path, window_length=1, window_interval=1)
    human_tracklets = det2d.read_tracklets(det2d_path, verbose=True)[categories.Human]
    
    noldus_timestamps_s, noldus_phases = get_noldus_phases(noldus_path)
    
    n_interactions_per_phase = np.zeros(shape=(len(phases),), dtype=int)
    n_movements_per_phase = np.zeros_like(n_interactions_per_phase)
    n_frames_per_phase = np.zeros_like(n_interactions_per_phase)
    # for frame, detections in tqdm(enumerate(detection_loader), desc=f"det2d file {video_index+1}", total=len(detection_loader)):
    for id, tracklet in tqdm(human_tracklets.items(), desc=f"det2d file {video_index+1}"):
        keypoints = tracklet[det2d.Keys.keypoints]
        
        frames = np.arange(keypoints.shape[0]) + tracklet[det2d.Keys.start]
        current_frame_timestamps_s = frame_timestamps_s[frames]
        current_frame_phase_indices = np.maximum(np.searchsorted(noldus_timestamps_s, current_frame_timestamps_s)-1, 0)
        current_frame_phases = noldus_phases[current_frame_phase_indices]
        
        # tracklets = det2d.detections2tracklets(detections)
        # if categories.Human not in tracklets.keys(): continue
        # human_tracklets = tracklets[categories.Human]
        frame_phase_assignment_mask = current_frame_phase_indices[:,None] == np.arange(len(phases))[None,:] # [F,1]==[1,P] -> [F,P]
        n_interactions_per_phase += np.sum(interaction_detector(tracklet)[:,None] * frame_phase_assignment_mask, axis=0) # sum([F,1]*[F,P], axis=0) -> [P]
        n_movements_per_phase += np.sum((1-movement_limits(tracklet))[:,None] * frame_phase_assignment_mask, axis=0)
        n_frames_per_phase += np.sum(frame_phase_assignment_mask, axis=0)
    
    write_video_header(det2d_path, summary_filehandle)
    write_interaction(n_frames_per_phase, n_movements_per_phase, n_interactions_per_phase, summary_filehandle)
    
    return n_interactions_per_phase, n_movements_per_phase, n_frames_per_phase

def process_config(config):
    summary_path, det2d_root, noldus_root, video_parsed_paths, movlims_path = get_paths(config)
    if not os.path.isfile(movlims_path): return
    movement_limits = Limits.from_file(movlims_path, MovementLimit)
    summary_filehandle = open(summary_path, 'w')
    
    n_interactions_per_phase = np.zeros(shape=(len(phases),), dtype=int)
    n_movements_per_phase = np.zeros_like(n_interactions_per_phase)
    n_frames_per_phase = np.zeros_like(n_interactions_per_phase)
    for video_index, video_parsed_path in enumerate(video_parsed_paths):
        measurements = process_video(det2d_root, noldus_root, video_index, video_parsed_path, movement_limits, summary_filehandle)
        n_interactions_per_phase += measurements[0]
        n_movements_per_phase += measurements[1]
        n_frames_per_phase += measurements[2]
        
    summary_filehandle.write("\n")
    write_config_header(summary_filehandle)
    write_interaction(n_frames_per_phase, n_movements_per_phase, n_interactions_per_phase, summary_filehandle)
    summary_filehandle.close()

if __name__=="__main__":
    args = cli()
    configs = load_configs(args.configs)
    categories = det2d.read_categories('cats.json')
    
    for config in configs:
        process_config(config)