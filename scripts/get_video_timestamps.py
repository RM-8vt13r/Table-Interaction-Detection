import os
import json
from wcmatch import glob
import subprocess
import configparser
import argparse
import numpy as np
from tqdm import tqdm

def cli():
    """
    Obtain command line arguments
    
    Outputs:
    - Namespace with the arguments
    """
    parser = argparse.ArgumentParser("Extract frame timestamps from each video")
    parser.add_argument('--configs', type=str, nargs='+', required=True, help="Path to the config file or a directory of config files")
    parser.add_argument('--out', type=str, required=True, help="Path to the output file")
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

if __name__ == '__main__':
    args = cli()
    configs = load_configs(args.configs)
    with open(args.out, 'w') as f:
        for config in configs:
            video_parsed_paths = get_paths(config)
            for path in tqdm(video_parsed_paths):
                video_timestamps = get_frame_timestamps_s(path)
                f.write(os.path.split(path)[1] + ": " + ",".join([str(timestamp) for timestamp in video_timestamps]) + "\n")