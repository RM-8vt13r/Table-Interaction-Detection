import os
import json
from wcmatch import glob
import subprocess
import configparser
import argparse
import numpy as np

def cli():
    """
    Obtain command line arguments
    
    Outputs:
    - Namespace with the arguments
    """
    parser = argparse.ArgumentParser("Count minimum, maximum and mean frames per seconds")
    parser.add_argument('--configs', type=str, nargs='+', required=True, help="Path to the config file or a directory of config files")
    return parser.parse_args()

def load_configs(config_paths: list) -> list:
    """
    Load configurations from paths
    
    Inputs:
    - config_paths: path(s) to config file(s)
    
    Outputs:
    - Loaded ConfigParser instances
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

def get_paths(config: configparser.ConfigParser) -> (str, list):
    """
    Parse video paths from a configuration
    
    Inputs:
    - config: the configuration
    
    Outputs:
    - Path(s) to the video(s) defined in the configuration
    """
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
    
    return video_parsed_paths
    
def get_frame_timestamps_s(video_path: str) -> np.ndarray:
    """
    Load all frame timestamps from a video
    
    Inputs:
    - video_path: path to video
    
    Outputs:
    - Array of frame timestamps in seconds
    """
    ffprobe_command = [
        'ffprobe',
        '-loglevel', 'quiet',
        '-output_format', 'csv',
        '-show_entries', 'packet=pts_time',
        video_path
    ]
    out,_ = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return np.array([float(line.split(',')[1].strip()) for line in out.decode().split('\n') if len(line)])

def process_video(video_path: str) -> np.ndarray:
    """
    Obtain and print fps from a video
    
    Inputs:
    - video_path: path to the video to analyse
    
    Outputs:
    - Inverse of frame time per frame
    """
    frame_timestamps_s = get_frame_timestamps_s(video_path)
    frame_times_s = np.diff(frame_timestamps_s)
    frame_fps = 1/frame_times_s
    print(f"Video \"{os.path.split(video_path)[1]}\": {np.unique(np.round(frame_fps, 3))}")
    print(f"Video \"{os.path.split(video_path)[1]}\": min fps = {np.min(frame_fps):.3f}, max fps = {np.max(frame_fps):.3f}, mean fps = {np.mean(frame_fps):.3f}, median fps = {np.median(frame_fps):.3f}")
    return frame_fps
    
def process_config(config: configparser.ConfigParser) -> np.ndarray:
    """
    Obtain and print fps from a config
    
    Inputs:
    - config: the configuration
    
    Outputs:
    - Inverse of frame time per frame
    """
    video_paths = get_paths(config)
    frame_fps = []
    for video_path in video_paths:
        frame_fps.extend(process_video(video_path))
    
    print(f"Config: {np.unique(np.round(frame_fps, 3))}")
    print(f"Config: min fps = {np.min(frame_fps):.3f}, max fps = {np.max(frame_fps):.3f}, mean fps = {np.mean(frame_fps):.3f}, median fps = {np.median(frame_fps):.3f}")
    print()
    return frame_fps

if __name__ == '__main__':
    args = cli()
    configs = load_configs(args.configs)
    frame_fps = []
    for config in configs:
        frame_fps.extend(process_config(config))
    
    print(f"Total: {np.unique(np.round(frame_fps, 3))}")
    print(f"Total: min fps = {np.min(frame_fps):.3f}, max fps = {np.max(frame_fps):.3f}, mean fps = {np.mean(frame_fps):.3f}, median fps = {np.median(frame_fps):.3f}")