if __name__=='__main__':
    from argparse import ArgumentParser
    import configparser
    import json
    from wcmatch import glob
    import os
    import subprocess
    from types import SimpleNamespace
    
    import numpy as np
    import det2d
    # import matplotlib.pyplot as plt

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
        parser = ArgumentParser("Count the number of personnel detections and annotations per time window")
        parser.add_argument('--configs', type=str, nargs='+', required=True, help="Path to the config file or a directory of config files")
        parser.add_argument('--window-length', type=int, default=7500, help="Length of the window in frames")
        parser.add_argument('--window-interval', type=int, default=7500, help="Distance between two window starts in frames")
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

    def get_paths(config: configparser.ConfigParser) -> (str, list):
        """
        Load all paths necessary to execute the script for a configuration
        
        Inputs:
        - config: the configuration
        
        Outputs:
        - Path to root where all pose detections can be found
        - Path to root where all noldus annotations can be found
        - Path to all video files that are included
        """
        summary_path = config.get('PATHS', 'summary_poses')
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
        return summary_path, det2d_root, noldus_root, video_parsed_paths

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
        """
        video_root, video_name = os.path.split(video_parsed_path)
        video_name = os.path.splitext(video_name)[0]
        det2d_path = os.path.join(det2d_root, video_name+'.det2d.json')
        noldus_path = os.path.join(noldus_root, 'RESA - ' + video_name + ' - Event Logs.txt')
        if not os.path.isfile(det2d_path): return
        return det2d_path, noldus_path
        
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
    
    def get_phases_per_frame(frame_timestamps: np.ndarray, phase_sequence: np.ndarray, phase_start_seconds: np.ndarray) -> np.ndarray:
        """
        Assign a phase to each of a sequence of frames
        
        Inputs:
        - frame_timestamps: the timestamp of each frame in seconds
        - phase_sequence: Annotated phase indices relative to the global variable 'phases'
        - phase_start_seconds: Timestamps in seconds corresponding to the entries in phase_sequence
        
        Outputs:
        - The phase index per frame in frame_timestamps
        """
        phase_indices = np.maximum(np.searchsorted(phase_start_seconds, frame_timestamps, 'right')-1, 0)
        phases_per_window = phase_sequence[phase_indices]
        return phases_per_window
    
    def get_detections_per_window(det2d_path: str) -> np.ndarray:
        """
        Get the number of detected poses during a number of windows
        
        Inputs:
        - det2d_path: path to pose detection file
        
        Outputs:
        - Array with number of detected poses per frame
        """
        detections = det2d.read_detections(det2d_path)
        n_detections_per_frame = np.array([len(frame_dict[0]) for frame_dict in detections.values()])
        return n_detections_per_frame
    
    def get_annotations_per_window(frame_timestamps: np.ndarray, start_seconds: np.ndarray, present_sequence: np.ndarray) -> np.ndarray:
        """
        Get the number of annotated persons during a number of windows
        
        Inputs:
        - frame_timestamps: a timestamp per frame in seconds
        - start_seconds: array with the start timestamp of each annotation
        - present_sequence: array with the number of annotated individuals corresponding to each entry in start_seconds
        """
        annotation_indices = np.maximum(0, np.minimum(len(start_seconds)-1, np.searchsorted(start_seconds, frame_timestamps)))
        n_annotations_per_frame = present_sequence[annotation_indices]
        return n_annotations_per_frame
    
    def process_video(det2d_root: str, noldus_root: str, video_path: str, summary_filehandle):
        """
        Count poses and detections in a single video
        
        Inputs:
        - det2d_root: path to root directory where all pose detections are stored
        - noldus_root: path to root directory where all noldus annotations are stored
        - video_path: path to video file to process
        - summary_filehandle: filehandle to file to write results to
        """
        frame_timestamps = get_frame_timestamps_s(video_path)
        det2d_path, noldus_path = get_video_paths(det2d_root, noldus_root, video_path)
        
        detection_loader = det2d.DetectionLoader(det2d_path, window_length=args.window_length, args.window_interval=window_interval)
        
        start_seconds, phase_sequence, active_sequence, idle_sequence, absent_sequence = get_noldus_phases(noldus_path)
        
        n_detections_per_window = get_detections_per_window(det2d_path)
        frame_timestamps = frame_timestamps[:len(n_detections_per_frame)]
        
        n_annotations_per_window = get_annotations_per_window(frame_timestamps, start_seconds, active_sequence + idle_sequence)
        
        phases_per_frame = get_phases_per_frame(frame_timestamps, phase_sequence, start_seconds)
        n_detections_per_phase = n_detections_per_frame
        
        print(f"Video {index+1} of {len(video_parsed_paths)}: \"{det2d_path}\":")
        print(f"{np.sum(n_detections_per_frame)} detections")
        print(f"{np.sum(n_annotations_per_frame)} annotations")
        print(f"{np.sum(n_detections_per_frame)/np.sum(n_annotations_per_frame)*100}% detected")
        print()
        
        # ax = figs[index].add_subplot()
        # ax.set_title(det2d_path)
        # ax.set_xlabel("Timestamp [s]")
        # ax.set_ylabel("Person count")
        # ax.plot(frame_timestamps, n_detections_per_frame)
        # ax.plot(frame_timestamps, n_annotations_per_frame, '--')
        # ax.plot(frame_timestamps, np.divide(n_detections_per_frame, n_annotations_per_frame, out=np.ones_like(n_detections_per_frame, dtype=float), where=n_annotations_per_frame > 0))
        # ax.legend(["Detections", "Annotations", "Ratio"])
        
        f.write(det2d_path + ":\n" + \
            f"Video annotated poses: {np.sum(n_annotations_per_frame)}\n" +
            "\n".join([
                f"Video annotated poses ({phase}): {np.sum(n_annotations_per_frame[phases_per_frame == phase_index])}"
                for phase_index, phase in enumerate(phases)
            ]) + "\n" +
            f"Video detected poses: {np.sum(n_detections_per_frame)} ({np.sum(n_detections_per_frame)/np.sum(n_annotations_per_frame)*100:.2f}%)\n" +
            "\n".join([
                f"Video detected poses ({phase}): {np.sum(n_detections_per_frame[phases_per_frame == phase_index])} ({np.sum(n_detections_per_frame[phases_per_frame == phase_index])/np.sum(n_annotations_per_frame[phases_per_frame == phase_index])*100:.2f}%)"
                for phase_index, phase in enumerate(phases)
            ]) + "\n" +
            f"Annotated poses per frame: {' '.join([f'({timestamp},{count})' for timestamp, count in zip(frame_timestamps, n_annotations_per_frame)])}\n" +
            "\n".join([
                f"Annotated poses per frame ({phase}): {' '.join([f'({timestamp},{count})' for timestamp, count in zip(frame_timestamps[phases_per_frame == phase_index], n_annotations_per_frame[phases_per_frame == phase_index])])}"
                for phase_index, phase in enumerate(phases)
            ]) + "\n" +
            f"Detected poses per frame: {' '.join([f'({timestamp},{count})' for timestamp, count in zip(frame_timestamps, n_detections_per_frame)])}\n" +
            "\n".join([
                f"Detected poses per frame ({phase}): {' '.join([f'({timestamp},{count})' for timestamp, count in zip(frame_timestamps[phases_per_frame == phase_index], n_detections_per_frame[phases_per_frame == phase_index])])}"
                for phase_index, phase in enumerate(phases)
            ]) + "\n" +
            f"Detected poses per frame (%): {' '.join([f'({timestamp},{percentage})' for timestamp, percentage in zip(frame_timestamps, np.round(np.divide(n_detections_per_frame, n_annotations_per_frame, out=np.ones_like(n_detections_per_frame, dtype=float), where=n_annotations_per_frame > 0)*100, 2))])}\n" +
            "\n".join([
                f"Detected poses per frame (%, {phase}): {' '.join([f'({timestamp},{percentage})' for timestamp, percentage in zip(frame_timestamps[phases_per_frame == phase_index], np.round(np.divide(n_detections_per_frame[phases_per_frame == phase_index], n_annotations_per_frame[phases_per_frame == phase_index], out=np.ones_like(n_detections_per_frame[phases_per_frame == phase_index], dtype=float), where=n_annotations_per_frame[phases_per_frame == phase_index] > 0)*100, 2))])}"
                for phase_index, phase in enumerate(phases)
            ]) +
            "\n\n"
        )
        
        return n_detections_per_phase, n_annotations_per_phase
        
    def process_config(config, args):
        """
        Count pose annotations and detections in a single configuration
        
        Inputs:
        - config: the configuration
        - args: the command line arguments
        """
        n_detections = np.zeros((len(phases),), dtype=int)
        n_annotations = np.zeros_like(n_detections)
        summary_path, det2d_root, noldus_root, video_parsed_paths = get_paths(config)
        # figs = [plt.figure() for _ in range(len(video_parsed_paths))]
        with open(summary_path, 'w') as f:
            for index, video_path in enumerate(video_parsed_paths):
                process_video(det2d_root, noldus_root, video_path, f)
                n_detections += np.sum(n_detections_per_frame[:,None]*(phases_per_frame[:,None] == np.arange(len(phases))[None,:]), axis=0)
                n_annotations += np.sum(n_annotations_per_frame[:,None]*(phases_per_frame[:,None] == np.arange(len(phases))[None,:]), axis=0)
        
            print(f"Total:")
            print(f"{np.sum(n_detections)} detections")
            print(f"{np.sum(n_annotations)} annotations")
            print(f"{np.sum(n_detections)/np.sum(n_annotations)*100}% detected")
            print()
            
            f.write("Total:\n" +
                f"Config annotated poses: {np.sum(n_annotations)}\n" +
                "\n".join([
                    f"Config annotated poses ({phase}): {n_annotations[phase_index]}"
                    for phase_index, phase in enumerate(phases)
                ]) + "\n" +
                f"Config detected poses: {np.sum(n_detections)} ({(np.sum(n_detections)/np.sum(n_annotations) if np.sum(n_annotations > 0) else 1)*100:.2f}%)\n" +
                "\n".join([
                    f"Config detected poses ({phase}): {n_detections[phase_index]} ({(n_detections[phase_index]/n_annotations[phase_index] if n_annotations[phase_index] > 0 else 1)*100:.2f}%)"
                    for phase_index, phase in enumerate(phases)
                ]) + "\n"
            )
            
        # plt.show(block=True)
    
    args = cli()
    configs = load_configs(args.configs)
    for config in configs:
        process_config(config, args)
        