import argparse
import json

def cli():
    """
    Return command line arguments
    
    Outputs:
    - arguments namespace
    """
    parser = argparse.ArgumentParser("Convert 'CVAT for images 1.1'-style polygon annotations to a .poslims.json file, writing results to the same directory")
    parser.add_argument('--cvat', required=True, type=str, nargs='+', help="Path(s) to the CVAT .xml annotations")
    parser.add_argument('--n-limits-required', type=int, default=2, help="Number of position limits that need to be satisfied for interaction")
    parser.add_argument('--n-keypoints-required', type=int, nargs='+', default=[1,1,2], help="Number of keypoints that should be within their respective area, three values in order Wrists, Shoulders, Head")
    parser.add_argument('--confidence-thresholds', type=float, nargs='+', default=[.3,.3,.15], help="Confidence threshold before a keypoint can be considered")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = cli()
    
    assert len(args.n_keypoints_required)==3, "--n-keypoints-required expects 3 values"
    assert len(args.confidence_thresholds)==3, "--confidence-thresholds expects 3 values"
    for confidence_threshold in args.confidence_thresholds: assert confidence_threshold >= 0 and confidence_threshold <= 1, "--confidence_thresholds must be at least 0 and at most 1"
    
    keypoint_group_indices = ([9,10], [5,6], [0,1,2,3,4])
    
    for cvat_path in args.cvat:
        with open(cvat_path, 'r') as f:
            cvat_lines = f.read()
        
        wrists_area_coordinates = cvat_lines.split('<polygon label="Patient Area (Wrists)" source="manual" occluded="0" points="')[1].split('" z_order')[0]
        shoulders_area_coordinates = cvat_lines.split('<polygon label="Patient Area (Shoulders)" source="manual" occluded="0" points="')[1].split('" z_order')[0]
        head_area_coordinates = cvat_lines.split('<polygon label="Patient Area (Head)" source="manual" occluded="0" points="')[1].split('" z_order')[0]
        area_coordinates = (wrists_area_coordinates, shoulders_area_coordinates, head_area_coordinates)
        
        with open(cvat_path.replace('.xml', '.poslims.json'), 'w') as f:
            json.dump({
                'n_limits_required': args.n_limits_required,
                'limits': [
                    {
                        'keypoint_classes': keypoint_classes,
                        'n_keypoints_required': n_keypoints_required,
                        'confidence_threshold': confidence_threshold,
                        'vertices': vertices
                    }
                    for keypoint_classes, n_keypoints_required, confidence_threshold, vertices in zip(keypoint_group_indices, args.n_keypoints_required, args.confidence_thresholds, area_coordinates)
                ]
            }, f, indent=4, separators=(',',': '))