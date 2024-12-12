# Notes on Event-based Vision

🦖 ".. _sssh_ they can't see us if we don't move"

<!-- mtoc-start -->

* [**1. Working Principles of Event-Based Cameras**](#1-working-principles-of-event-based-cameras)
  * [**2. Types of Event Cameras**](#2-types-of-event-cameras)
  * [**3. Event Representations**](#3-event-representations)
  * [**4. Processing Strategies**](#4-processing-strategies)
  * [**5. Applications**](#5-applications)
  * [**6. Mathematical Formulations**](#6-mathematical-formulations)
  * [**7. Challenges**](#7-challenges)

<!-- mtoc-end -->

### **1. Working Principles of Event-Based Cameras**

- **Fact**: DVS pixels respond independently and asynchronously to changes in relative light intensity.
  - \( e = (x, y, t, p) \): Event representation with spatial (\(x, y\)) and temporal (\(t\)) coordinates, and polarity (\(p\)).
- **Advantages**:
  - **Low Power Consumption**: Sparse data reduces computational and storage costs.
  - **High Temporal Resolution**: Enables low latency and minimizes motion blur.
  - **High Dynamic Range (HDR)**: Independent response of pixels ensures high contrast sensitivity.
- **Challenges**:
  - No absolute intensity; brightness changes are in logarithmic scale.
  - Requires novel algorithms for space-time processing.

### **2. Types of Event Cameras**

- **Dynamic Vision Sensor (DVS)**:
  - Captures brightness changes as events.
  - Models transient visual pathways ("where").
  - Evolution from \(128 \times 128\) pixels (2008) to 1 Mpixel versions.
- **Asynchronous Time-based Image Sensor (ATIS)**:
  - Outputs both change detection (CD) and exposure measurement (EM) events.
  - Asynchronous grayscale intensity encoding.
- **Dynamic Pixel and Active Vision Sensor (DAVIS)**:
  - Combines DVS events with standard camera frames.
  - Frames resemble information in sustained visual pathways ("what").
  - Includes inertial measurement unit (IMU) data.

### **3. Event Representations**

- **Point Sets**:
  - Events represented in space-time as \(e_k = (\mathbf{x}\_k, t_k, p_k)\).
- **Event Frames**:
  - Aggregated events converted to 2D histograms, brightness increment images, or time surfaces.
  - Histograms of polarities: \(\Delta L(\mathbf{x}) = \text{sum of event polarities per pixel}\).
- **Voxel Grids**:
  - Events represented in 3D histograms discretized over space and time.
- **Motion Compensation**:
  - Aligns events based on motion hypothesis, creating sharper edge maps.
- **Reconstructed Intensity Images**:
  - Approximation of scene intensity by accumulating event data.

### **4. Processing Strategies**

- **Event-by-Event**:
  - Process individual events using filters or spiking neural networks (SNNs).
  - Advantage: Minimal latency.
  - Disadvantage: High computational cost at high event rates.
- **Event Packets**:
  - Aggregate groups of events, suitable for CPU or GPU processing.
  - Example Applications:
    - Feature tracking (e.g., Harris, KLT).
    - Stereo depth estimation.
    - Optical flow estimation.

### **5. Applications**

- **Intensity Reconstruction**:
  - High-speed and HDR video from events.
- **Object Recognition**:
  - Enhanced using color event cameras.
- **SLAM and Tracking**:
  - Event-based simultaneous localization and mapping (SLAM).
- **Computational Photography**:
  - Achieves HDR and low-latency imaging.

### **6. Mathematical Formulations**

- **Event Generation**:
  - Polarity change $\( p = \text{sign}(\Delta \log I) \)$, where $\( \Delta \log I \)$ is the logarithmic intensity change.
- **Histogram of Events**:
  $$
  \[
  H(x, y) = \sum_k \delta(x - x_k, y - y_k)
  \]
  $$
  - $\(H(x, y)\)$: Event count per pixel.

### **7. Challenges**

- Noise and Refractory Period:
  - Real-world sensors introduce trailing events and sensor-specific noise.
- Processing Bottlenecks:
  - High event rates demand novel algorithms and optimized hardware.
