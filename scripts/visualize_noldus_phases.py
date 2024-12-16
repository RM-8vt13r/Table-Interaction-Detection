if __name__ == '__main__':
    import os
    import json
    from types import SimpleNamespace
    import argparse
    from datetime import datetime
    import numpy as np
    from matplotlib import pyplot as plt, patches as mpatches
    
    event_types = SimpleNamespace(**{
        'START': 'State start',
        'STOP': 'State stop'
    })
    
    subjects = SimpleNamespace(**{
        'ENVIRONMENT': 'Omgeving',
        'STAFF': 'Staflid',
        'SCRUBNURSE': 'Scrub Nurse'
    })
    
    phase_colours = {
        None: 'k',
        'Inleiding anesthesie': 'r',
        'Chirurgische voorbereidingen': 'g',
        'Snijtijd': 'b',
        'Uitleiding anesthesie': 'c',
        'Patient kamer uit': 'm'
    }
    
    def cli():
        parser = argparse.ArgumentParser("From Noldus annotations, visualize the phases over time")
        parser.add_argument('--path', required=True, type=str, help="Path to the Noldus files root")
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
        
    args = cli()
    
    file_names = [name for name in os.listdir(args.path) if name[-4:]=='.txt']
    phases_fig, phases_ax = plt.subplots()
    active_fig, active_ax = plt.subplots()
    idle_fig, idle_ax     = plt.subplots()
    absent_fig, absent_ax = plt.subplots()
    present_fig, present_ax = plt.subplots()
    total_fig, total_ax   = plt.subplots()
    current_active_y = 0
    current_idle_y = 0
    current_absent_y = 0
    current_present_y = 0
    current_total_y = 0
    for file_index, file_name in enumerate(file_names):
        start_seconds, phase_sequence, active_sequence, idle_sequence, absent_sequence = get_noldus_phases(os.path.join(args.path, file_name))
        
        for index, (start_second, end_second, phase, active_count, idle_count, absent_count) in enumerate(zip(start_seconds[:-1], start_seconds[1:], phase_sequence[:-1], active_sequence[:-1], idle_sequence[:-1], absent_sequence[:-1])):
            phase_colour = phase_colours[phase_sequence]
            phases_ax.plot([start_second, end_second], 2*[file_index], phase_colour, linewidth=7, solid_capstyle='butt')
            active_ax.plot([start_second, end_second], 2*[current_activity_y + active_count], phase_colour, linewidth=7, solid_capstyle='butt')
            idle_ax.plot([start_second, end_second], 2*[current_idle_y + idle_count], phase_colour, linewidth=7, solid_capstyle='butt')
            absent_ax.plot([start_second, end_second], 2*[current_absent_y + absent_count], phase_colour, linewidth=7, solid_capstyle='butt')
            present_ax.plot([start_second, end_second], 2*[current_present_y + active_count + idle_count], phase_colour, linewidth=7, solid_capstyle='butt')
            total_ax.plot([start_second, end_second], 2*[current_total_y + active_count + idle_count + absent_count], phase_colour, linewidth=7, solid_capstyle='butt')
            
        current_active_y += 2 + np.max(active_sequence[:-1])
        current_idle_y += 2 + np.max(idle_sequence[:-1])
        current_absent_y += 2 + np.max(absent_sequence[:-1])
        current_present_y += 2 + np.max(active_sequence[:-1] + idle_sequence[:-1])
        total_present_y += 2 + np.max(active_sequence[:-1] + idle_sequence[:-1] + absent_sequence[:-1])
            
        # ax.plot([timestamp_seconds+duration_seconds, line_dict['Time_Relative_sf']], [file_index, file_index], phase_colours[None], linewidth=7, solid_capstyle='butt')
        # ax.text(line_dict['Time_Relative_sf'], file_index, file_name, verticalalignment='center')
    
    phases_ax.legend(handles=[mpatches.Patch(color=colour, label=phase) for phase, colour in phase_colours.items() if phase is not None])
    phases_ax.grid(True)
    phases_ax.set_ylim([0-1,file_index+1])
    phases_ax.set_xlabel('Procedure duration [s]')
    phases_ax.set_title('Annotated procedure phases')
    
    active_ax.grid(True)
    active_ax.set_ylim([0-1,current_active_y-1])
    active_ax.set_xlabel('Procedure duration [s]')
    active_ax.set_title('Annotated number of active persons')
    
    idle_ax.grid(True)
    idle_ax.set_ylim([0-1,current_idle_y-1])
    idle_ax.set_xlabel('Procedure duration [s]')
    idle_ax.set_title('Annotated number of idle persons')
    
    absent_ax.grid(True)
    absent_ax.set_ylim([0-1,current_absent_y-1])
    absent_ax.set_xlabel('Procedure duration [s]')
    absent_ax.set_title('Annotated number of absent persons')
    
    present_ax.grid(True)
    present_ax.set_ylim([0-1,current_present_y-1])
    present_ax.set_xlabel('Procedure duration [s]')
    present_ax.set_title('Annotated number of present (active + idle) persons')
    
    total_ax.grid(True)
    total_ax.set_ylim([0-1,current_total_y-1])
    total_ax.set_xlabel('Procedure duration [s]')
    total_ax.set_title('Annotated number of total (active + idle + absent) persons')
    
    plt.show(block=True)