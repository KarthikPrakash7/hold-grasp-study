# Climbing Hold Taxonomy

Read before annotating in Roboflow. All annotators must use these definitions.

## Classes

### jug
Large incut hold. The whole hand fits inside the cavity. Easy to grip. Often at route start/top.
**Annotation:** Box tightly around the entire hold body including the incut lip.

### crimp
Small horizontal edge, typically 5–20 mm depth. Only fingertips contact it.
**Annotation:** Box around the edge surface only, not surrounding plastic.

### sloper
Smooth, rounded hold with no positive edge. Requires open-hand friction. Surface is convex outward.
**Annotation:** Box around the rounded surface.

### pinch
Hold designed to be squeezed between thumb (one side) and fingers (opposite side). Typically a knob or rib.
**Annotation:** Box around the full pinch body including both contact sides.

### pocket
Hold with one or more finger-depth holes. Fingers insert into the pocket.
**Annotation:** Box around the entire pocket body including the hole opening.

## Ambiguous Cases

- Hold could be jug or sloper: choose by grip type most climbers would use.
- Dual-texture holds: label by the dominant grip type.
- Tiny jugs that look like crimps: label by incut size (thumb-width incut → jug).
- Footholds too small to classify: skip.
