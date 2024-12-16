# Patient_Interaction_Detection
A repository to detect whether medical staff in the OR is interacting with the patient

Modules:
- interaction: calculate personnel-patient interaction
  - positioning.py: analyze personnel positioning relative to the patient
  - movement.py: analyze personnel movement
  - interaction.py: analyze personnel-patient interaction
  - summarize.py: summarize personnel-patient interaction
  - keypoints.py: list of human COCO keypoints

- scripts: detect patient interaction using the previous modules
  - interaction_from_det2d.py: given pose tracklets and annotated patient area, summarize patient interaction time per individual

Planned:
- Visualizing patient interaction

Usage:
```python
from interaction import PositionLimit, MovementLimit, Limits, InteractionDetector, summarize
import det2d

# Create limits
position_limits = Limits.from_file(poslims_path, PositionLimit)
movement_limits = Limits.from_file(movlims_path, MovementLimit)

# Create detector
interaction_detector = InteractionDetector(position_limits, movement_limits)

# Load poses
categories = det2d.read_categories(cats_path)
tracklets  = det2d.read_tracklets(det2d_path)
human_tracklets = tracklets[categories.Human]

# Detect and summarize interaction
for human_id,human_tracklet in human_tracklets.items():
    interaction = interaction_detector(human_tracklet)
    print(f"{human_id}: {summarize(interaction)}")
```

Test:
`pytest testing/`