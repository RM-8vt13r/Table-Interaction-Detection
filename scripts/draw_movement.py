import sys
sys.path.append('src')
import os
import subprocess

from wcmatch import glob
import shutil
import logging
import argparse
import configparser
import json

from tqdm import tqdm
import cv2
import numpy as np
import ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.text
import matplotlib.animation
from cycler import cycler

import det2d
from interaction import MovementLimit, Limits, summarize

def round(x, decimals):
    return np.floor(x*(10**decimals)+.5)/(10**decimals)
    
def pose_bbox(keypoints, confidence_threshold=None):
    keypoints = np.array(keypoints)
    if confidence_threshold is not None: keypoints = keypoints[(keypoints[:,2]>=confidence_threshold) & (keypoints[:,2]>0),:]
    x1, y1, x2, y2 = np.min(keypoints[:,0]), np.min(keypoints[:,1]), np.max(keypoints[:,0]), np.max(keypoints[:,1])
    return x1, y1, x2-x1, y2-y1
    
def pose_score(keypoints):
    keypoints = np.array(keypoints)
    return np.mean(keypoints[keypoints[:,2]>0,2])

def cli():
    ## Read configuration
    parser = argparse.ArgumentParser("Draw 2D poses in images or videos, colour-coded on movement.")
    parser.add_argument('--configs', nargs='+', type=str, required=True, help="Paths to the det2d config files.")
    parser.add_argument('--cats', default='./cats.json', type=str, help="Path to the categories file.")
    parser.add_argument('--vis', default='./vis.ini', type=str, help="Path to visualization file.")
    parser.add_argument('--frames', type=str, default=':', help="List of frames to convert. start:stop and start:stop:step define frame ranges. start, stop and step can be left empty to signal the video beginning, end, and step size 1. A single index defines a single frame. Multiple ranges and/or indices can be combined by separating with ','.")
    parser.add_argument('--format', type=str, help="Output file format. Determines whether output is a video ('mp4', 'mkv'), raster image ('png', 'jpg'), vector image ('pdf', 'eps'), or LaTeX script ('tex'). In case of tex, it is recommended to use the latin modern font (\\usepackage{{lmodern}}).")
    parser.add_argument('--decimals', default=2, type=int, help="How many decimal places to save")
    parser.add_argument('--draw-annotations', action=argparse.BooleanOptionalAction, default=False, type=bool, help="If included, draw annotations rather than estimations.")
    parser.add_argument('--overwrite-all', action=argparse.BooleanOptionalAction, default=False, type=bool, help="If included, overwrite all pre-existing paths without asking.")
    parser.add_argument('--overwrite-none', action=argparse.BooleanOptionalAction, default=False, type=bool, help="If included, preserve all pre-existing paths without asking.")
    parser.add_argument('--optimize', action=argparse.BooleanOptionalAction, default=False, type=bool, help="Whether to optimize file size with lossless compression, only supported for mp4, mkv, pdf formats.")
    return parser.parse_args()

def load_configs(config_paths):
    configs = []
    for config_path in config_paths:
        if os.path.isdir(config_path):
            configs.extend(load_configs([os.path.join(config_path, path) for path in os.listdir(config_path) if path[-4:]=='.ini']))
            continue
        configs.append(config_path)
    return configs

if __name__=='__main__':
    args = cli()
    
    ## Check and process arguments
    # --configs
    args.configs = load_configs(args.configs)
    
    # --cats
    assert args.cats[-5:]=='.json', f"cats should be a .json file, but ended with {args.cats[-5:]}"
    assert os.path.isfile(args.cats), f"cats file \"{args.cats}\" does not exist"
    
    with open(args.cats, 'r') as f:
        cats = json.load(f, object_hook=lambda d: {int(k) if k.isdigit() else k: np.array(v) if isinstance(v, list) and k != "keypoint_subset_indices" else v for k,v in d.items()})
    
    categories = det2d.read_categories('cats.json')
    
    # --vis
    assert args.vis[-4:]=='.ini', f"vis should be a .ini file, but ended with {args.vis[-4:]}"
    assert os.path.isfile(args.vis), f"vis file \"{args.vis}\" does not exist"
    
    vis = configparser.ConfigParser()
    vis.read(args.vis)
    
    # --frames
    assert len(args.frames.strip()), "Frame list was empty"
    frames = []
    for frame_str in args.frames.split(','):
        assert len(frame_str.strip()), "One of the provided ranges/indices was empty"
        range_params = frame_str.strip().split(':')
        assert len(range_params)<=3, "Each element in frames may contain at most 2 colons to define a range"
        if len(range_params)==1:
            frames.append(int(range_params[0].strip()))
        else:
            frames += [[int(p.strip()) if len(p.strip()) else None for p in range_params]]
    
    # --format
    assert args.format in ('mp4', 'mkv', 'png', 'jpg', 'pdf', 'eps', 'tex'), f'format must be mp4, mkv, png, jpg, pdf, eps or tex, but was {args.format}'
    
    ## Define constants
    matplotlib_font_weights = {
        'ultralight': 200,
        'light': 300,
        'normal': 400,
        'medium': 500,
        'semibold': 600,
        'bold': 700,
        'extra bold': 800,
        'heavy': 900
    }
    
    latex_font_weights = { # Latex 
        #'ul': 250,
        #'el': 312.5,
        #'l': 375,
        #'sl': 437.5,
        #'m': 500,
        #'sb': 562.5,
        #'b': 625,
        #'eb': 750,
        #'ub': 1000
        'm': 333.33,
        'b': 666.67,
        'bx': 1000
    }
    
    conf_fontdict = {
        'family': 'sans-serif',
        'color': np.array(json.loads(vis.get('VISUALIZATION', 'font_colour')), dtype=float)/255,
        'size': vis.getfloat('VISUALIZATION', 'confidence_font_size_pt'),
        'weight': list(matplotlib_font_weights.keys())[min(np.searchsorted(list(matplotlib_font_weights.values()), vis.getint('VISUALIZATION', 'confidence_font_weight'), side='left'), len(matplotlib_font_weights.keys())-1)],
        'horizontalalignment': 'center',
        'verticalalignment': 'center_baseline',
        'zorder': 3
    }
    
    id_fontdict = {
        'family': 'sans-serif',
        'color': np.array(json.loads(vis.get('VISUALIZATION', 'font_colour')), dtype=float)/255,
        'size': vis.getfloat('VISUALIZATION', 'confidence_font_size_pt'),
        'weight': list(matplotlib_font_weights.keys())[min(np.searchsorted(list(matplotlib_font_weights.values()), vis.getint('VISUALIZATION', 'id_font_weight'), side='left'), len(matplotlib_font_weights.keys())-1)],
        'horizontalalignment': 'center',
        'verticalalignment': 'center_baseline',
        'zorder': 3
    }
    
    if args.format=='tex':
        conf_fontdict_latex = {
            'weight': list(latex_font_weights.keys())[min(np.searchsorted(list(latex_font_weights.values()), vis.getint('VISUALIZATION', 'confidence_font_weight'), side='left'), len(latex_font_weights.keys())-1)]
        }
        id_fontdict_latex = {
            'weight': list(latex_font_weights.keys())[min(np.searchsorted(list(latex_font_weights.values()), vis.getint('VISUALIZATION', 'id_font_weight'), side='left'), len(latex_font_weights.keys())-1)]
        }
    
    conf_fontprop = mpl.font_manager.FontProperties(
        family = conf_fontdict['family'],
        size = conf_fontdict['size'],
        weight = conf_fontdict['weight']
    )
    
    id_fontprop = mpl.font_manager.FontProperties(
        family = id_fontdict['family'],
        size = id_fontdict['size'],
        weight = id_fontdict['weight']
    )
    
    limb_width_pt = vis.getfloat('VISUALIZATION', 'limb_width_pt')
    joint_width_pt = vis.getfloat('VISUALIZATION', 'joint_width_pt')
    textbox_colour = np.array(json.loads(vis.get('VISUALIZATION', 'textbox_colour')), dtype=float)/255
    textbox_margin_pt = vis.getfloat('VISUALIZATION', 'textbox_margin_pt')
    figure_height_pt = vis.getfloat('VISUALIZATION', 'figure_height_pt')
    pt_per_inch = 72 # matplotlib constant
    
    ## Check inputs and ask permission for overwriting
    video_paths    = [None]*len(args.configs)
    vipr_paths     = [None]*len(args.configs)
    poslims_paths  = [None]*len(args.configs)
    det2d_paths    = [None]*len(args.configs)
    visualization_paths = [None]*len(args.configs)
    visualization_fpss  = [None]*len(args.configs)
    #frames         = [None]*len(args.configs)
    config_skip_inds = set()
    for c, config_path in enumerate(args.configs):
        config = configparser.ConfigParser()
        config.read(config_path)
    
        ## Setup logger
        LOG = logging.getLogger()
        #logging.basicConfig(level=logging.DEBUG if config.getboolean('DEBUG', 'debug') else logging.INFO)
        logging.basicConfig(level=logging.INFO)
        LOG.info(f"Configuration {c+1} of {len(args.configs)} read")
        LOG.debug("Debug messages enabled")
        
        ## Load input paths
        LOG.info("Preparing data paths")
        
        # Load data paths from configuration
        data_unparsed_paths = json.loads(config.get('PATHS', 'data')) # List of strings, pointing to files and/or directories, with wildcards
        data_paths = []
        for data_unparsed_path in data_unparsed_paths:
            for data_parsed_path in glob.glob(data_unparsed_path, flags=glob.EXTGLOB): # Parse wildcards
                if os.path.isdir(data_parsed_path): # Add directory contents
                    data_paths.extend([os.path.join(data_parsed_path, file_name) for file_name in os.listdir(data_parsed_path)])
                elif os.path.isfile(data_parsed_path): # Add files directly
                    data_paths.append(data_parsed_path)
        data_names = [os.path.split(data_path)[1] for data_path in data_paths]
        
        # Load det2d paths from configuration
        results_path = config.get('PATHS', 'det2d')
        assert os.path.isdir(results_path), "PATHS->det2d_path should be an existing directory, but it isn't"
        results_names = os.listdir(results_path)
        results_paths = [os.path.join(results_path, name) for name in results_names]
        
        # List all video, vipr and det2d files from the data paths. Only load videos that have a corresponding vipr file, poslims file, and det2d file
        names = [name[:-4] for name in data_names if name[-4:] in ('.mp4', '.mkv')] # All video names excluding .mp4
        names = [name for name in names if name+'.vipr.json' in data_names and name+'.poslims.json' in data_names and name+'.det2d.json' in results_names] # Include only videos that have a corresponding vipr and det2d file
        
        video_paths[c] = [[path for path in data_paths if os.path.split(path)[1][:-4]==name and path[-4:] in ('.mp4', '.mkv')][0] for name in names] # Final list of video paths in the order of names
        vipr_paths[c] = [[path for path in data_paths if os.path.split(path)[1]==name+'.vipr.json'][0] for name in names] # Final list of vipr paths in the order of names
        poslims_paths[c] = [[path for path in data_paths if os.path.split(path)[1]==name+'.poslims.json'][0] for name in names] # Final list of poslims paths in the order of names
        det2d_paths[c] = [[path for path in results_paths if os.path.split(path)[1]==name+'.det2d.json'][0] for name in names] # Final list of det2d paths in the order of names
        
        ## Load output paths
        LOG.info("Preparing visualization paths")
        
        visualization_paths[c] = [os.path.join(config.get('PATHS', 'visualization'), name+(f'.{args.format}' if args.format in ('mp4', 'mkv') else '')) for name in names] # Video or directory depending on format
        if not os.path.isdir(config.get('PATHS', 'visualization')): os.makedirs(config.get('PATHS', 'visualization'))
        try: # Try writing video output fps
            visualization_fpss[c] = (vis.getint('VISUALIZATION', 'fps'),)*len(visualization_paths[c])
        except:
            visualization_fpss[c] = [None,]*len(visualization_paths[c])
        
        ## Check paths to ignore
        ignore_inds = set()
        
        # Check if inputs have duplicates within this or other configurations
        for i, input_paths in enumerate(zip(video_paths[c], vipr_paths[c], det2d_paths[c], visualization_paths[c], visualization_fpss[c])):
            if i in ignore_inds: continue # This input was already ignored
            for c2,_ in enumerate(args.configs[:c+1]):
                for i2, input_paths2 in enumerate(zip(video_paths[c2][:i], vipr_paths[c2][:i], det2d_paths[c2][:i], visualization_paths[c2][:i], visualization_fpss[c2][:i])):
                    if input_paths==input_paths2: # Duplicate found!
                        ignore_inds.add(i)
                        LOG.info(f"Duplicate input video \"{input_paths[0]}\" with keypoints \"{input_paths[2]}\" and visualization path \"{input_paths[3]}\" will be ignored")
                    
        # Ask permission for overwriting
        for i, visualization_path in enumerate(visualization_paths[c]):
            if i in ignore_inds: continue # This input was already ignored
            if args.format not in ('mp4', 'mkv'):
                if os.path.isdir(visualization_path): # Image visualization
                    reply = 'y' if args.overwrite_all else 'n' if args.overwrite_none else ''
                    while reply not in ('y', 'n'):
                        print(f"Output directory \"{visualization_path}\" already exists. Overwrite contents? (y/n)")
                        reply = input()
                    if reply.lower()=='y':
                        LOG.info(f"Existing \"{visualization_path}\" has not been deleted, and new results will replace or add to its contents")
                    else:
                        ignore_inds.add(i)
                        LOG.info(f"Existing \"{visualization_path}\" will be ignored and no new results will be generated")
                else:
                    os.makedirs(visualization_path)
                
            elif args.format in ('mp4', 'mkv') and os.path.isfile(visualization_path):
                reply = 'y' if args.overwrite_all else 'n' if args.overwrite_none else ''
                while reply not in ('y', 'n'):
                    print(f"Output file \"{visualization_path}\" already exists. Overwrite? (y/n)")
                    reply = input()
                if reply.lower()=='y':
                    os.remove(visualization_path)
                    LOG.info(f"Existing \"{visualization_path}\" has been deleted and new results will be generated in its place")
                else:
                    ignore_inds.add(i)
                    LOG.info(f"Existing \"{visualization_path}\" will be ignored and no new results will be generated")
        
        # Remove paths to be ignored
        for ignore_ind in sorted(tuple(ignore_inds), reverse=True):
            for path_list in video_paths[c], vipr_paths[c], poslims_paths[c], det2d_paths[c], visualization_paths[c], visualization_fpss[c]:
                path_list.pop(ignore_ind)
                
        # Check if paths remain for this configuration
        for path_list in video_paths[c], vipr_paths[c], poslims_paths[c], det2d_paths[c], visualization_paths[c], visualization_fpss[c]:
            if len(path_list)==0:
                LOG.info("No input files remain -> skipping configuration")
                config_skip_inds.add(c)
                break
    
    ## Remove skipped configurations
    if len(config_skip_inds)==len(args.configs):
        LOG.info("Skipped all configurations -> quitting")
        quit()
    
    for skip_ind in sorted(tuple(config_skip_inds), reverse=True):
        LOG.info(f"Skipping configuration {skip_ind+1}")
        for path_list in args.configs, video_paths, vipr_paths, poslims_paths, det2d_paths, visualization_paths, visualization_fpss:
            path_list.pop(skip_ind)
    
    # Draw poses
    for c, config_path in enumerate(args.configs):
        config = configparser.ConfigParser()
        config.read(config_path)
    
        ## Setup logger
        LOG = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG if config.getboolean('DEBUG', 'debug') else logging.INFO)
        LOG.info(f"Configuration {c+1} of {len(args.configs)} read")
        LOG.debug("Debug messages enabled")
        
        # Iterate over videos
        LOG.info('Iterating over {} video(s)'.format(len(video_paths[c])))
        LOG.setLevel(logging.CRITICAL) # Suppress logging from here on
        for i, (video_path, vipr_path, poslims_path, det2d_path, visualization_path, visualization_fps) in enumerate(zip(video_paths[c], vipr_paths[c], poslims_paths[c], det2d_paths[c], visualization_paths[c], visualization_fpss[c])):
            # Load video
            with open(vipr_path, 'r') as f:
                vipr = json.load(f) # Load properties
            
            capture = cv2.VideoCapture(video_path)
            
            # Load detections
            LOG.info("Loading tracklets")
            detections = det2d.read_detections(det2d_path)
            tracklets = det2d.detections2tracklets(detections, verbose=True)
            human_tracklets = tracklets[categories.Human]
            
            print("Sorting tracklets")
            human_tracklets = dict(sorted(human_tracklets.items(), key=lambda tracklet: tracklet[1][det2d.Keys.start]))
            
            print("Loading limits")
            movement_limits = Limits.from_file(config.get('PATHS', 'movlims'), MovementLimit)
            
            print("Detecting movements")
            movement_satisfied = [movement_limits(tracklet) for tracklet in human_tracklets.values()]
            movement_satisfied = dict(zip(human_tracklets.keys(), movement_satisfied))
            
            print("Preparing visualization")
            
            # Load figure
            dots_per_pt = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/figure_height_pt
            dpi = dots_per_pt*pt_per_inch
            figure_width_pt = capture.get(cv2.CAP_PROP_FRAME_WIDTH)/dots_per_pt
            
            fig, ax = plt.subplots(dpi=dpi, figsize=(figure_width_pt/pt_per_inch, figure_height_pt/pt_per_inch))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            fig.set_frameon(False)
            renderer = fig.canvas.get_renderer()
            
            # Load writer
            if args.format in ('mp4', 'mkv'):
                if visualization_fps is not None: writer_fps=visualization_fps
                else:
                    ffprobe_command = [
                        'ffprobe',
                        '-loglevel', 'quiet',
                        '-output_format', 'default=noprint_wrappers=1:nokey=1',
                        '-show_entries', 'stream=r_frame_rate',
                        video_path
                    ]
                    print(f"Executing ffprobe on {video_path} (might take a while)")
                    out,_ = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                    numerator, denominator = [float(value) for value in out.decode().split('/')]
                    writer_fps = numerator/denominator
                
                writer = mpl.animation.FFMpegFileWriter(
                    fps=writer_fps,
                    codec='h264'
                )
                writer.setup(fig, visualization_path, dpi)
            
            # Iterate over frames
            frames_config = [] # Unpack frames
            for element in frames:
                if isinstance(element, int): frames_config += [element]
                else: frames_config += [*range(0 if element[0] is None else element[0], vipr['stop']-vipr['start'] if element[1] is None else element[1], 1 if len(element)==2 or element[2] is None else element[2])]
            for frame in frames_config: assert frame>=0 and frame<vipr['stop']-vipr['start'], f"Configuration \"{config_path}\", video \"{video_path}\": requested frame {frame} is out of bounds (0-{vipr['stop']-vipr['start']})"
            frames_config = np.array(frames_config)
            
            det2d_frames = np.array(list(detections.keys())) # Subset detection frames
            # det2d_frames = det2d_frames[np.unique(np.where(np.any(det2d_frames[:,None]==frames_config[None,:], axis=1))[0])]
            det2d_config_frames = np.array([det2d_frame for det2d_frame in det2d_frames if det2d_frame in frames_config])
            
            tracklet_ids_queued = list(human_tracklets.keys())
            tracklet_ids_active = []
            for det2d_frame in tqdm(det2d_config_frames, desc=f"Drawing poses in video {i+1} of {len(video_paths[c])}, configuration {c+1} of {len(args.configs)} ({os.path.split(args.configs[c])[1]})"):
                if capture.get(cv2.CAP_PROP_POS_FRAMES) != det2d_frame+vipr['start']: capture.set(cv2.CAP_PROP_POS_FRAMES, det2d_frame+vipr['start'])
                ret, frame = capture.read()
                if not ret: continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.clear()
                ax.set_axis_off()
                ax.set_frame_on(False)
                ax.set_xlim(frame.shape[0])
                ax.set_ylim(frame.shape[1])
                img = ax.imshow(frame, interpolation='none')
                
                # Initialize figure
                if args.format=='tex':
                    tikz_string =  f'\\begin{{tikzpicture}}[x=1pt,y=-1pt]\n'
                    tikz_string += f'\\node[inner sep=0,anchor=north west] at (0,0) {{\\includegraphics[width={figure_width_pt}pt]{{\\currfiledir {det2d_frame}.jpg}}}};\n'
                
                # Draw poses
                while len(tracklet_ids_queued) and human_tracklets[tracklet_ids_queued[0]][det2d.Keys.start] <= det2d_frame: tracklet_ids_active.append(tracklet_ids_queued.pop(0))
                for tracklet_index, tracklet_id in reversed(list(enumerate(tracklet_ids_active))):
                    if human_tracklets[tracklet_id][det2d.Keys.start]+human_tracklets[tracklet_id][det2d.Keys.keypoints].shape[0] <= det2d_frame: tracklet_ids_active.pop(tracklet_index)
                    
                category = list(cats.keys())[categories.Human]
                limbs = np.array(cats[category]['edges'])
                joint_colour = np.array(cats[category]["keypoint_colour"])/255
                movement_colour = np.array(cats[category]["position_only_colour"])/255
                no_movement_colour = np.array(cats[category]["no_interaction_colour"])/255
                for tracklet_id in tracklet_ids_active: # Iterate over predictions of this category
                    tracklet = human_tracklets[tracklet_id]
                    
                    # Draw limbs and keypoints
                    keypoints = tracklet[det2d.Keys.keypoints][det2d_frame-tracklet[det2d.Keys.start]] # [K,3]
                    
                    if pose_score(keypoints) < vis.getfloat('VISUALIZATION', 'object_confidence_threshold') or pose_score(keypoints) <= 0: continue # Don't draw poses that don't satisfy the threshold
                    
                    keypoint_mask = (keypoints[:,2] >= vis.getfloat('VISUALIZATION', 'keypoint_confidence_threshold')) & (keypoints[:,2] > 0) & (keypoints[:,0] >= 0) & (keypoints[:,1] >= 0) & (keypoints[:,0] < frame.shape[1]) & (keypoints[:,1] < frame.shape[0]) # [K]
                    if np.sum(keypoint_mask)==0: continue # No keypoints to show
                    limb_mask = np.all(keypoint_mask[limbs], axis=1) # [L]
                    if movement_satisfied[tracklet_id][det2d_frame-tracklet[det2d.Keys.start]]: limb_colour = no_movement_colour
                    else: limb_colour = movement_colour
                    
                    if args.format=='tex':
                        tikz_string += '\n'.join(['\\draw[draw={{rgb,1:red,{0};green,{1};blue,{2}}},line width={3}pt, line cap=round] ({4},{5})--({6},{7});'.format(*limb_colour, limb_width_pt, *(keypoints[limb[0],:2]/dots_per_pt), *(keypoints[limb[1],:2]/dots_per_pt)) for limb in limbs[limb_mask]])
                        tikz_string += '\n'
                        tikz_string += '\n'.join(['\\node[circle,draw=none,fill={{rgb,1:red,{0};green,{1};blue,{2}}},inner sep=0,minimum size={3}pt] at ({4},{5}) {{}};'.format(*joint_colour, joint_width_pt, *(keypoint[:2]/dots_per_pt)) for keypoint in keypoints[keypoint_mask]])
                        tikz_string += '\n'
                    else:
                        if np.any(limb_mask):
                            ax.plot(keypoints[limbs[limb_mask],0].T, keypoints[limbs[limb_mask],1].T, color=limb_colour, solid_capstyle='round', linewidth=limb_width_pt, zorder=0)
                        if np.any(keypoint_mask): ax.plot(keypoints[keypoint_mask,0], keypoints[keypoint_mask,1], linestyle='', marker='o', markersize=joint_width_pt, markeredgewidth=0, markerfacecolor=joint_colour, zorder=1)
                                            
                    # Draw confidence and ID
                    textbox_width = 0
                    textbox_height = 0
                    textbox_center = np.array(pose_bbox(keypoints, vis.getfloat('VISUALIZATION', 'keypoint_confidence_threshold'))[:2]) # box center coordinates at pose bounding box corner
                    
                    conf_val = pose_score(keypoints)
                    conf_text = '{{:.{d}f}}'.format(d=args.decimals).format(round(conf_val, args.decimals))
                    conf_text_object = ax.text(0, 0, conf_text, conf_fontdict)
                    conf_text_bbox = conf_text_object.get_window_extent().transformed(ax.transData.inverted()) # Exact text size in data units
                    conf_text_bbox.y0, conf_text_bbox.y1 = conf_text_bbox.y1, conf_text_bbox.y0
                    textbox_width = max(textbox_width, conf_text_bbox.width) # Exact text width in px
                    textbox_height += conf_text_bbox.height # Exact text height
                    
                    id_text = str(tracklet_id)
                    id_text_object = ax.text(0, 0, id_text, id_fontdict)
                    id_text_bbox = id_text_object.get_window_extent(renderer=renderer).transformed(ax.transData.inverted()) # Exact text size in data units
                    id_text_bbox.y0, id_text_bbox.y1 = id_text_bbox.y1, id_text_bbox.y0
                    textbox_width = max(textbox_width, id_text_bbox.width) # Exact text width
                    textbox_height += id_text_bbox.height # Exact text height
                    
                    textbox_width  += 2*textbox_margin_pt*dots_per_pt # Add margins -> box is wider than text
                    textbox_height += 2*textbox_margin_pt*dots_per_pt # Add margins -> box is higher than text
                    textbox_height += textbox_margin_pt*dots_per_pt
                    
                    textbox_center[0] = min(max(textbox_center[0], textbox_width/2), frame.shape[1]-textbox_width/2-1)
                    textbox_center[1] = min(max(textbox_center[1], textbox_height/2), frame.shape[0]-textbox_height/2-1)
                    
                    if args.format=='tex':
                        tikz_string += '\\node[fill={{rgb,1:red,{0};green,{1};blue,{2}}},opacity={3},inner sep=0pt,minimum width={4}pt,minimum height={5}pt,align=center] at ({6},{7}) {{}};\n'.format(*textbox_colour[:3], textbox_colour[3], textbox_width/dots_per_pt, textbox_height/dots_per_pt, *(textbox_center/dots_per_pt))
                        tikz_string += '\\node[text={{rgb,1:red,{0};green,{1};blue,{2}}},align=center,text depth=0pt,font=\\fontseries{{{3}}}\\fontsize{{{4}pt}}{{0pt}}\\selectfont] at ({5},{6}) {{{7}}};\n'.format(*conf_fontdict['color'], conf_fontdict_latex['weight'], conf_fontdict['size'], textbox_center[0]/dots_per_pt, (textbox_center[1]+.5*textbox_height-.5*conf_text_bbox.height)/dots_per_pt-textbox_margin_pt, conf_text)
                        tikz_string += '\\node[text={{rgb,1:red,{0};green,{1};blue,{2}}},align=center,text depth=0pt,font=\\fontseries{{{3}}}\\fontsize{{{4}pt}}{{0pt}}\\selectfont] at ({5},{6}) {{{7}}};\n'.format(*id_fontdict['color'], id_fontdict_latex['weight'], id_fontdict['size'], textbox_center[0]/dots_per_pt, (textbox_center[1]-.5*textbox_height+.5*id_text_bbox.height)/dots_per_pt+textbox_margin_pt, id_text)
                    else:
                        ax.add_patch(mpl.patches.Rectangle(textbox_center-.5*np.array((textbox_width, textbox_height)), textbox_width, textbox_height, color=textbox_colour, fill=True, linestyle='', linewidth=0, zorder=2))
                        conf_text_object.set_position((textbox_center[0], textbox_center[1]+textbox_height/2 - textbox_margin_pt*dots_per_pt - .5*conf_text_bbox.height))
                        id_text_object.set_position((textbox_center[0], textbox_center[1]-textbox_height/2 + textbox_margin_pt*dots_per_pt + .5*id_text_bbox.height))
                    
                # Write frame
                match args.format:
                    case 'png' | 'jpg' | 'pdf' | 'eps':
                        write_path = os.path.join(visualization_path, f'{det2d_frame}.{args.format}') # Always write pdf, convert later with ghostscript
                        fig.savefig(write_path, bbox_inches='tight', pad_inches=0)
                    case 'tex':
                        tikz_string += f'\\end{{tikzpicture}}'
                        write_path_base = os.path.join(visualization_path, str(det2d_frame))
                        cv2.imwrite(write_path_base+'.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        with open(write_path_base+'.tex', 'w') as f:
                            f.write(tikz_string)
                    case 'mp4' | 'mkv':
                        writer.grab_frame()
                
                # Optimize
                if args.optimize and args.format in ('pdf', 'eps'):
                    write_path_temp = write_path[:-4]+'.tmp.'+args.format
                    shutil.copyfile(write_path, write_path_temp)
                    try:
                        os.system(f'gswin64c -q -dBATCH -dSAFER -dNOPAUSE -sDEVICE={"pdfwrite" if args.format=="pdf" else "eps2write"} -dAutoRotatePages=/None -dPDFSETTINGS=/default -dCompressPages=true {"-sUseOCR=never" if args.format=="pdf" else ""} -sOutputFile="{write_path}" "{write_path_temp}"')
                    except:
                        LOG.info(f'Failed to compress {args.format}: ghostscript not found')
                    os.remove(write_path_temp)
            
            capture.release()
            plt.close(fig)
            
            if args.format not in ('mp4', 'mkv'): continue
            
            writer.finish()
            
            if not args.optimize: continue
            
            # Optimize
            results_file_path_temp = '{0}.tmp.{1}'.format(*visualization_path.rsplit('.', 1))
            os.rename(visualization_path, results_file_path_temp)
            print(f"Compressing pose video {i+1} of {len(video_paths[c])}, configuration {c+1} of {len(args.configs)}")
            (
                ffmpeg
                .input(results_file_path_temp)
                .output(visualization_path, acodec='copy', vcodec='libx265')
                .run()
            )
            os.remove(results_file_path_temp)
            
    LOG.info('Done!')