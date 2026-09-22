# Camera Navigation Update

This document details the newly introduced camera controls in the KinectFusionQt application, mimicking the Blender-like navigation while extending the system to support free-flight. 

## Features Added & Rationality

### 1. Unified `Camera` Class supporting both Orbit and Free modes
- **What was changed:** 
  The existing `OrbitCamera` was renamed to `Camera` and expanded to track not only the target and spherical coordinates (azimuth, elevation) but also explicit position and Euler angles (pitch, yaw, roll). An internal mode toggle (`Mode::Orbit` vs `Mode::Free`) dictates how inputs are processed and how the `viewMatrix()` is calculated.
- **Rationality:**
  Supporting two distinct paradigms of navigation (revolving vs. flying) within a single UI view requires shared state or smooth transitions. By embedding both modes inside a single `Camera` class, the viewport seamlessly recalculates coordinates (e.g., inferring target position from free position, or vice versa) whenever the user switches context without breaking the view matrix.

### 2. Blender-like Orbiting & Target Focus
- **What was changed:**
  Pressing the `F` key forces the application into Orbit Mode and centers the camera's target back onto the object/origin (`[0, 0, 1]`). Orbit mode remains the default, where dragging the mouse revolves the viewpoint spherically around this target.
- **Rationality:**
  When checking the quality of a recreation, the focus is the 3D volume (the reconstructed mesh). By revolving around a static point (Target), the user can inspect the generated geometry from all angles intuitively without drifting out into empty space. 

### 3. Axis Locks for X, Y, and Z Panning
- **What was changed:**
  Added constraint handling in `OpenGLWidget` that listens for the `X`, `Y`, or `Z` keys. Holding one of these keys while dragging with the Right Mouse Button locks the panning to that specific local axis (Z acts as an explicit zoom lock). 
- **Rationality:**
  Modeled after Blender's transform shortcuts (G + X/Y/Z), giving the user precise, predictable translations when they are trying to align their view or inspect a specific cross-section of the reconstructed mesh.

### 4. XYZ UI Sliders (Control Panel)
- **What was changed:**
  Added three `QSlider` widgets inside the `ControlPanel` corresponding to Pitch (X), Yaw (Y), and Roll (Z). These UI elements are bi-directionally bound to the `OpenGLWidget`.
- **Rationality:**
  Providing dedicated visual sliders fulfills the requirement of discrete UI interaction for rotation. It makes exact alignments accessible for users preferring UI constraints over free-hand mouse drag, particularly useful for inspecting misalignments in the 3D scan on an explicit axis. Sliders are generally more intuitive and standard in 3D applications compared to dials.
- **Status correction (2026-09-22, big-fix Todo 32 / audit `gui:GL-29`):** this is history, not the current UI. The three `QSlider` rotation widgets were dead weight (nothing drove them bidirectionally) and have since been **removed** from `ControlPanel`; the CPU gate now fails if the name reappears in `ControlPanel.*`. Rotation is still bi-directionally bound through `ControlPanel::setCameraRotation()` / `cameraRotationChanged()` and the on-canvas navigation gizmo — read the paragraph above as the design rationale that survived, not as a widget inventory.

### 5. Fly / Free Mode (WASD)
- **What was changed:**
  A QTimer-driven physics loop was added to `OpenGLWidget` operating at ~60fps. Pressing `Tab` deselects the origin and switches the camera into Free Mode. The `W, A, S, D, Q, E` keys traverse the 3D space relative to the camera's current viewing angle (First Person control), and Left-Click + Drag rotates the head (yaw/pitch).
- **Rationality:**
  Orbiting is counter-intuitive if the user accidentally pans too far away from the model or needs to explore interior spaces (like a reconstructed room). The First Person / Free flight navigation solves this by anchoring traversal to the camera's origin rather than an external target point.

---

## Adversarial Review: Status Update (2026-04-26)

The following critiques from the initial implementation have been addressed to ensure system stability and consistency:

1. **Gimbal Lock & Numerical Stability:**
   - **Update:** The dependency on `std::asin` (prone to `NaN`) has been replaced with direct trigonometric projections in the mode transition logic. `pitch_` is strictly clamped, preventing zenith flips.

2. **UI Sliders Desync:**
   - **Update:** **RESOLVED**. The `Camera` class now uses a unified `yaw_/pitch_` state shared between both modes. Sliders in the `ControlPanel` are bi-directionally bound and update in real-time during WASD flight and mouse-look.

3. **Input Handling:**
   - **Update:** **RESOLVED**. `OpenGLWidget` correctly manages focus, and keys like `W, A, S, D` are only captured when the viewport has focus, preventing interference with other UI elements.

4. **Hardcoded Frame Rates:**
   - **Update:** **RESOLVED**. Traversal speed is now frame-rate independent. Movement is calculated using a high-resolution `deltaTime` (`frame_timer_.restart()`), ensuring consistent speed across different monitor refresh rates and CPU loads.

5. **Axis Panning Constraints:**
   - **Update:** **RESOLVED**. The 'Z' key now correctly translates the camera along the world Z-axis during panning, rather than incorrectly hijacking the 'zoom' (distance) parameter.

6. **Build System Regression (NavigationGizmo):**
   - **Update:** **RESOLVED**. The build system was modernized to use `CMAKE_AUTOMOC` and explicit header tracking. This ensures that the `NavigationGizmo` and other Qt-dependent components are always correctly compiled and linked, even after complex repository updates or `git pull` operations.

