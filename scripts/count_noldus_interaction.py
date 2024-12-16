if __name__ == '__main__':
    import os
    import json
    from types import SimpleNamespace
    import argparse
    from datetime import datetime
    from matplotlib import pyplot as plt, patches as mpatches
    import numpy as np
    import math
    from wcmatch import glob
    
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
        parser = argparse.ArgumentParser("From Noldus annotations, count interaction over time")
        parser.add_argument('--noldus-path', required=True, type=str, help="Extglob path to the Noldus file(s) to count")
        parser.add_argument('--video-summary-path', required=True, type=str, help="Path to summary txt file to write procedure- & video results to")
        return parser.parse_args()
    
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
    
    def weight_activity(start_seconds: np.ndarray, phase_sequence: np.ndarray, active_sequence: np.ndarray, idle_sequence: np.ndarray, absent_sequence: np.ndarray) -> np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray:
        """
        From a range of timestamps and corresponding annotations, weight annotation by duration
        
        Inputs:
        - start_seconds: timestamps in seconds where annotations start
        - phase_sequence: sequence of annotated phase indices corresponding to each entry in start_seconds
        - active_sequence: sequence of active person counts corresponding to each entry in start_seconds
        - idle_sequence: sequence of idle person counts corresponding to each entry in start_seconds
        - absent_sequence: sequence of absent person counts corresponding to each entry in start_seconds
        
        Outputs:
        - Array of durations in seconds per annotation
        - Phase annotations corresponding to each entry in the array of durations
        - Annotated active person counts, multiplied by their duration in seconds
        - Annotated idle person counts, multiplied by their duration in seconds
        - Annotated absent person counts, multiplied by their duration in seconds
        """
        duration_seconds = np.diff(start_seconds)
        sampled_phase_sequence, active_sequence, idle_sequence, absent_sequence = phase_sequence[:-1], active_sequence[:-1], idle_sequence[:-1], absent_sequence[:-1]
        weighted_active_sequence  = duration_seconds*active_sequence
        weighted_idle_sequence    = duration_seconds*idle_sequence
        weighted_absent_sequence  = duration_seconds*absent_sequence
        
        return duration_seconds, sampled_phase_sequence, weighted_active_sequence, weighted_idle_sequence, weighted_absent_sequence
    
    args = cli()
    
    if not os.path.isdir(os.path.split(args.video_summary_path)[0]): os.makedirs(os.path.split(args.video_summary_path)[0])
    summary_file = open(args.video_summary_path, 'w')
    
    total_time_per_phase = np.zeros(shape=len(phases), dtype=float)
    total_active_time_per_phase = np.zeros_like(total_time_per_phase)
    total_present_time_per_phase = np.zeros_like(total_time_per_phase)
    file_names = glob.glob(args.noldus_path.replace('\\', '/'), flags=glob.EXTGLOB)
    for file_index, file_name in enumerate(file_names):
        file_path = os.path.join(args.noldus_path, file_name)
        
        start_seconds, phase_sequence, active_sequence, idle_sequence, absent_sequence = get_noldus_phases(file_path)
        duration_seconds, sampled_phase_sequence, weighted_active_sequence, weighted_idle_sequence, weighted_absent_sequence = weight_activity(start_seconds, phase_sequence, active_sequence, idle_sequence, absent_sequence)
        
        for phase_index, phase in enumerate(phases):
            timeline_mask = sampled_phase_sequence == phase
            active_time   = np.sum(weighted_active_sequence[timeline_mask])
            present_time  = np.sum(weighted_active_sequence[timeline_mask] + weighted_idle_sequence[timeline_mask])
            summary_file.write(f"Procedure_interaction ({phase}): True on {active_time} of {present_time} seconds ({active_time/present_time*100 if present_time > 0 else 0:.1f}%)\n")
            total_active_time_per_phase[phase_index] += active_time
            total_present_time_per_phase[phase_index] += present_time
        
        active_time   = np.sum(weighted_active_sequence)
        present_time  = np.sum(weighted_active_sequence + weighted_idle_sequence)
        summary_file.write(f"Procedure interaction (Total): True on {active_time} of {present_time} seconds ({active_time/present_time*100 if present_time > 0 else 0:.1f}%)\n")
        
        for phase_index, phase in enumerate(phases):
            timeline_mask = sampled_phase_sequence == phase
            phase_time    = np.sum(duration_seconds[timeline_mask])
            summary_file.write(f"Procedure duration ({phase}): {phase_time} seconds\n")
            total_time_per_phase[phase_index] += phase_time
        
        phase_time = np.sum(duration_seconds)
        summary_file.write(f"Procedure duration (Total): {phase_time} seconds\n")
        summary_file.write(f"\n")
        
    summary_file.write(f"\n")
    
    summary_file.write(f"Total:\n")
    for phase_index, phase in enumerate(phases):
        active_time = total_active_time_per_phase[phase_index]
        present_time = total_present_time_per_phase[phase_index]
        summary_file.write(f"Total interaction ({phase}): True on {active_time} of {present_time} seconds ({active_time/present_time*100 if present_time > 0 else 0:.1f}%)\n")
    
    active_time = np.sum(total_active_time_per_phase)
    present_time = np.sum(total_present_time_per_phase)
    summary_file.write(f"Total Interaction (Total): True on {active_time} of {present_time} seconds ({active_time/present_time*100 if present_time > 0 else 0:.1f}%)\n")
    
    for phase_index, phase in enumerate(phases):
        phase_time = total_time_per_phase[phase_index]
        summary_file.write(f"Total duration ({phase}): {phase_time} seconds\n")
    
    phase_time = np.sum(total_time_per_phase)
    summary_file.write(f"Total duration (Total): {phase_time} seconds\n")
    
    summary_file.close()