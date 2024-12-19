# Notes on Event-based Vision

🦖 ".. _sssh_ they can't see us if we don't move"

<!-- mtoc-start -->

* [📸 Summary of Event-Based Vision Notes](#-summary-of-event-based-vision-notes)
  * [Key Features of Event-Based Cameras](#key-features-of-event-based-cameras)
  * [Mathematical Model for Event Generation](#mathematical-model-for-event-generation)
  * [Types of Event-Based Sensors](#types-of-event-based-sensors)
  * [Representations of Event Data](#representations-of-event-data)
  * [Event Processing Techniques](#event-processing-techniques)
  * [Applications](#applications)
  * [Advantages and Challenges](#advantages-and-challenges)
  * [Future Directions](#future-directions)
* [🕶️ Awesome EBV](#-awesome-ebv)
  * [Essential Reading](#essential-reading)
  * [Papers of Interest](#papers-of-interest)
  * [Datasets](#datasets)
  * [Videos / Teachings](#videos--teachings)
* [🗺️ Research / Learnings Roadmap](#-research--learnings-roadmap)

<!-- mtoc-end -->

## 📸 Summary of Event-Based Vision Notes

### Key Features of Event-Based Cameras

- **Core Principle**:
  - Detects changes in light intensity asynchronously for each pixel.
  - Unlike frame-based cameras, outputs data only when a change occurs.
- **Event Data**: Outputs events $e = (x, y, t, p)$, where:
  - $(x, y)$ Pixel coordinates.
  - $t$: Timestamp with $\mu s$ precision.
  - $p$: Polarity ($+1$ for intensity increase, $-1$ for decrease).
- **Advantages**:
  - High temporal resolution ($\sim 1 \mu s$).
  - Low latency and minimal motion blur.
  - High dynamic range (HDR) $\sim 140 \, dB$.
  - Sparse output with reduced data redundancy.
  - Energy-efficient (low power consumption).
  - Real-time capabilities ideal for high-speed applications.

### Mathematical Model for Event Generation

1. **Trigger Condition**:

```math
\Delta L = \log I(x, t) - \log I(x, t - \Delta t) = \pm C
```

- $\Delta L$: Log intensity change.
- $C$: Contrast threshold triggering the event.

2. **Brightness Constancy Equation**:
   - Events align with gradients of intensity changes over time:

```math
\frac{dL}{dt} = \nabla L \cdot \mathbf{v} + \frac{\partial L}{\partial t} = 0,
```

where $\mathbf{v}$ Apparent velocity of intensity changes on the image plane.

3. **Noise Model**:
   - Event generation is affected by probabilistic noise:

```math
\Delta L \sim \mathcal{N}(C, \sigma^2),
```

where $\sigma^2$ represents noise variance.

### Types of Event-Based Sensors

1. **Dynamic Vision Sensor (DVS)**:
   - Specialized in detecting temporal contrast events.
   - Resolution ranges from $128 \times 128$ (early models) to $1 \, MP$ (modern sensors).
2. **Asynchronous Time-based Image Sensor (ATIS)**:
   - Combines DVS-like events with grayscale exposure measurements (EM).
   - Provides temporal and intensity data simultaneously.
3. **Dynamic and Active Pixel Vision Sensor (DAVIS)**:
   - Merges DVS events with standard frame-based imaging.
   - Outputs: DVS events, grayscale frames, and IMU data.

### Representations of Event Data

1. **Point-Based Representations**:
   - **Space-Time Point Clouds**: Represent events in $x, y, t$ space.
   - Retains spatial and temporal structure.
2. **Event Frames**:
   - Converts events into 2D histograms or edge maps:
     - Histograms of event counts.
     - Polarity-based accumulations.
     - Time surfaces ( $T(x, y) = f(\text{time of last event})$ ).
3. **Voxel Grids**:
   - Constructs 3D histograms of events in a space-time volume.
   - Balances memory usage and event fidelity.
4. **Motion-Compensated Frames**:
   - Aligns events based on motion hypotheses for sharper edge representation.
5. **Reconstructed Intensity Images**:
   - Integrates event streams to approximate scene brightness:

```math
\log \hat{I}(x, t) = \log I(x, 0) + \sum_k p_k C \delta(x - x_k, t - t_k).
```

- Applications:
  - HDR video recovery.
  - Scene understanding.

### Event Processing Techniques

1. **Event-by-Event Processing**:
   - Processes each event individually, maintaining microsecond latency.
   - Example Methods:
     - Spiking Neural Networks (SNNs).
     - Bayesian filters for noise reduction and inference.
   - **Pros**: Minimal latency, high temporal precision.
   - **Cons**: Computationally expensive for high event rates.
2. **Event Packets**:
   - Groups $N$ events into packets for batch processing.
   - Aggregation strategies include fixed time intervals or adaptive thresholds.
   - Suitable for real-time tasks with reduced computational overhead.
3. **Hybrid Representations**:
   - Time surfaces and voxel grids combine advantages of event-by-event and packet processing.

### Applications

1. **SLAM (Simultaneous Localization and Mapping)**:
   - Combines event data with inertial measurement units (IMUs) for high-speed, HDR mapping.
2. **Autonomous Systems**:
   - Low-latency, robust detection in dynamic environments (e.g., drones, self-driving cars).
3. **Object Recognition and Tracking**:
   - High-speed motion tracking and gesture recognition.
4. **3D Reconstruction**:
   - Stereo event cameras generate depth maps and high-precision reconstructions.
5. **Microscopy**:
   - Tracks rapid biological dynamics, such as neural activity.

### Advantages and Challenges

**Advantages**:

- Real-time operation with minimal motion blur.
- Wide HDR enables use in low-light or high-contrast scenarios.
- Low data redundancy reduces memory and bandwidth requirements.

**Challenges**:

- **Algorithm Design**:
  - Asynchronous data necessitates novel approaches incompatible with traditional vision algorithms.
- **Event Noise**:
  - Variability in event generation requires denoising techniques.
- **Integration with Frame-Based Systems**:
  - Hybrid systems face challenges in synchronizing event-based and frame-based data.
- **Hardware Limitations**:
  - Achieving high resolution while retaining low power consumption.

### Future Directions

1. **Algorithmic Development**:
   - Focus on neuromorphic designs mimicking biological systems.
   - Improved reconstruction and feature extraction techniques.
2. **Enhanced Hardware**:
   - Higher resolutions with efficient processing pipelines.
3. **Applications in Computational Imaging**:
   - Integrating event cameras with deep learning for end-to-end system optimization.

## 🕶️ Awesome EBV

See these links for a more extensive list of resources:

- [UZH's Robotics and Perception Group](https://github.com/uzh-rpg/event-based_vision_resources)
- https://github.com/chakravarthi589/Event-based-Vision_Resources
- https://github.com/vlislab22/Deep-Learning-for-Event-based-Vision
- [PROPHESEE White Paper](https://www.prophesee.ai/wp-content/uploads/2024/01/White_Paper_EN_January_2024.pdf)

### Essential Reading

- [Event-based Vision: A Survey](https://arxiv.org/pdf/1904.08405)
- [Recent Event Camera Innovations: A Survey](https://arxiv.org/pdf/2408.13627)
- [Deep Learning for Event-based Vision: A Comprehensive Survey and Benchmarks](https://arxiv.org/pdf/2302.08890)

  > We first scrutinize the typical event representations with quality enhancement methods as they play a pivotal role as inputs to the DL models. We then provide a comprehensive survey of existing DL-based methods by structurally grouping them into two major categories: 1) image/video reconstruction and restoration; 2) event-based scene understanding and 3D vision. We conduct benchmark experiments for the existing methods in some representative research directions i.e., image reconstruction, deblurring, and object recognition, to identify some critical insights and problems. Finally, we have discussions regarding the challenges and provide new perspectives for inspiring more research studies.

  <details>

  **Paper Title:** _Deep Learning for Event-based Vision: A Comprehensive Survey and Benchmarks_ (arXiv:2302.08890)

  **Context & Motivation:**

  - Event-based cameras (DVS, ATIS, etc.) operate differently than traditional frame-based cameras, recording asynchronous changes in brightness at individual pixels.
  - These event streams are sparse, high temporal resolution signals with low latency, high dynamic range, and no motion blur.
  - Deep learning techniques, which have revolutionized conventional vision tasks, need adaptation to handle these fundamentally different data formats.
  - This paper provides a comprehensive survey of deep learning methods developed for event-based vision, alongside benchmarks and frameworks for evaluation.

  **Key Contributions:**

  1. **Comprehensive Survey:**

     - Reviews a wide range of deep learning models for event-based data, including their architectures, representations, and learning paradigms.
     - Covers tasks such as object recognition, tracking, optical flow estimation, SLAM, HDR imaging, and classification.

  2. **Taxonomy of Event Representations & Methods:**

     - Systematically classifies input representations from raw event streams to frame-like event grids, voxel grids, event surfaces, and graph-based representations.
     - Details how these representations interface with popular deep neural network architectures (CNNs, RNNs, Transformers, SNNs).

  3. **Benchmarking & Datasets:**

     - Introduces standardized benchmarks for comparing methods.
     - Reviews existing event-based datasets and their characteristics, guiding the selection of appropriate training and testing sets.

  4. **Performance Analysis & Open Challenges:**

     - Identifies strengths and weaknesses of current state-of-the-art methods.
     - Outlines open research questions such as how to leverage temporal sparsity more effectively, how to combine event-based and frame-based modalities, and how to develop large-scale pretrained models for events.

  **Event-based Vision Fundamentals:**

  - **Event Generation Model:**
    Event cameras output a stream of _events_, each indicating a brightness change at a specific pixel location. An event \( e_i \) is typically defined as:
    \[
    e_i = (x_i, y_i, t_i, p_i)
    \]
    where:

    - \( x_i, y_i \): Pixel coordinates.
    - \( t_i \): Timestamp of the event.
    - \( p_i \in \{+1, -1\} \): Polarity of the change (increase or decrease in brightness).

  - **Brightness Change Trigger:**
    If \(L(x,y,t)\) represents the logarithm of pixel brightness, an event is triggered when:
    \[
    L(x,y,t_i) - L(x,y,t_i - \Delta t) \geq \theta
    \]
    Here, \(\theta\) is a threshold that determines the sensitivity of the event camera. When this threshold is exceeded (i.e., a sufficient change in log-intensity occurs), an event is produced.

  **Representations for Deep Learning:**

  - **Event Frames / Histograms:**
    Aggregating events over a fixed time interval \(\Delta T\):
    \[
    F(x,y) = \sum\_{t_i \in [T, T+\Delta T]} \mathbb{I}(x_i = x, y_i = y) p_i
    \]
    where \(\mathbb{I}(\cdot)\) is an indicator function selecting events at a given pixel, and \(p_i\) is often added or used to form separate positive/negative channels.

  - **Voxel Grids:**
    Partitioning a time interval into \(B\) bins creates a 3D volume (x,y,b):
    \[
    V(x,y,b) = \sum\_{t_i \in T_b} f(e_i)
    \]
    where \(T_b\) is the set of events in the \(b\)-th time bin, and \(f(e_i)\) could encode polarity or event count. This yields a spatiotemporal voxel representation that can be fed into 3D CNNs.

  - **Event Surfaces:**
    Using the last timestamp of an event at each pixel:
    \[
    S(x,y,t) = t - t\_{\text{last event at }(x,y)}
    \]
    Normalized or transformed variants of these surfaces serve as continuous-time feature maps.

  **Deep Learning Architectures & Approaches:**

  - **Convolutional Neural Networks (CNNs):**
    Applied on event frames or voxel grids. CNNs can handle stacked "event images" as input channels.

  - **Spiking Neural Networks (SNNs):**
    Naturally well-suited to event streams due to their asynchronous and spike-based computations. They often process event sequences without explicit frame conversion.

  - **Recurrent Neural Networks (RNNs)/LSTMs/Transformers:**
    Used to capture temporal dependencies directly from event streams. Transformers leverage attention to handle the sparse and asynchronous nature of events.

  - **Hybrid Architectures:**
    Combining event and frame-based modalities or integrating event-specific preprocessing with standard deep architectures to improve efficiency and performance.

  **Applications Covered:**

  - **Object Classification & Detection:**
    Models that outperform frame-based methods under challenging lighting or high-speed scenarios.

  - **Optical Flow & Motion Estimation:**
    Exploiting the high temporal resolution of events to estimate flow at microsecond precision.

  - **Visual SLAM & 3D Reconstruction:**
    Leveraging events for robust SLAM in low-light and high-speed environments, addressing challenges where frame-based systems fail.

  - **HDR Imaging & Depth Estimation:**
    Using event streams to reconstruct high-dynamic-range images or estimate depth information that traditional frames cannot capture reliably.

  **Benchmarking & Evaluation:**

  - The paper discusses standardizing evaluation protocols:

    - Clear definitions of training/testing splits in event-based datasets.
    - Metrics that reflect both accuracy and temporal resolution advantages, such as event-based error measures.

  - Provides guidance on selecting datasets and tasks for fair comparison:
    - N-Caltech101, N-CARS for classification.
    - DSEC, EVIMO for depth and motion tasks.
    - Event-based optical flow datasets like EV-FlowNet’s benchmarks.

  **Open Challenges & Future Directions:**

  - **Data Scarcity & Pretraining:**
    Limited labeled event datasets motivate development of self-supervised, semi-supervised, or synthetic data generation approaches.

  - **Computational Efficiency & Hardware Integration:**
    Efficient architectures that can run in real-time on specialized event-based processors or neuromorphic chips.

  - **Bridging Modalities:**
    Combining event-based and frame-based data, or integrating events with other sensor modalities (LiDAR, radar) for robust perception.

  - **Unified Benchmarks & Metrics:**
    Establishing agreed-upon benchmarks, metrics, and protocols to fairly evaluate and accelerate progress in this rapidly evolving field.

  **Equations at a Glance:**

  1. **Event Definition:**
     \[
     e_i = (x_i, y_i, t_i, p_i)
     \]

  2. **Event Trigger Condition:**
     \[
     L(x,y,t_i) - L(x,y,t_i - \Delta t) \geq \theta
     \]

  3. **Event Frame Accumulation:**
     \[
     F(x,y) = \sum\_{t_i \in [T, T+\Delta T]} p_i \mathbb{I}(x_i = x, y_i = y)
     \]

  4. **Voxel Grid Representation:**
     \[
     V(x,y,b) = \sum\_{t_i \in T_b} f(e_i)
     \]

  **Overall Summary:**
  This survey provides a thorough overview of how deep learning techniques can be adapted and applied to event-based vision. It clarifies the unique properties of event data, discusses a wide range of representations, deep network architectures, and application-specific methods. It highlights available datasets, benchmarks, and existing methodologies, while identifying open challenges and future directions to guide the development of more powerful, efficient, and generalizable event-based deep learning systems.
    </details>

- [Low-latency automotive vision with event cameras](https://www.nature.com/articles/s41586-024-07409-w)
- [Video to Events: Recycling Video Datasets for Event Cameras](https://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf)

### Papers of Interest

- [End-to-End Edge Neuromorphic Object Detection System](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10595906)
- [A Recurrent YOLOv8-Based Framework for Event-based Object Detection](https://arxiv.org/pdf/2408.05321)
- [Time Aggregation based Lossless Video Encoding for Neuromorphic Vision Sensor Data](https://eprints.kingston.ac.uk/id/eprint/45886/6/Khan-N-45886-AAM.pdf)
  - Algorithm used [here for generating superframes](https://www.kaggle.com/code/gogo827jz/generating-superframes-from-dvs-event-data)
  - See. `notebooks/generating-superframes-from-dvs-event-data.ipynb`
- [PEDRo: an Event-based Dataset for Person Detection in Robotics](https://tub-rip.github.io/eventvision2023/papers/2023CVPRW_PEDRo_An_Event-based_Dataset_for_Person_Detection_in_Robotics.pdf)
  - [Github](https://github.com/SSIGPRO/PEDRo-Event-Based-Dataset)
- [MS-EVS: Multispectral event-based vision for deep learning based face detection](https://openaccess.thecvf.com/content/WACV2024/papers/Himmi_MS-EVS_Multispectral_Event-Based_Vision_for_Deep_Learning_Based_Face_Detection_WACV_2024_paper.pdf)
- [Low-power, Continuous Remote Behavioral Localization with Event Cameras](https://arxiv.org/pdf/2312.03799)
- [A 240 × 180 130 dB 3 µs Latency Global Shutter Spatiotemporal Vision Sensor](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6889103)
- [How Many Events Make an Object? Improving Single-frame Object Detection on the 1 Mpx Dataset](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Kugele_How_Many_Events_Make_an_Object_Improving_Single-Frame_Object_Detection_CVPRW_2023_paper.pdf)
- [EVREAL: Towards a Comprehensive Benchmark and Analysis Suite for Event-based Video Reconstruction](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Ercan_EVREAL_Towards_a_Comprehensive_Benchmark_and_Analysis_Suite_for_Event-Based_CVPRW_2023_paper.pdf)
- [Entropy Coding-based Lossless Compression of Asynchronous Event Sequences](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Schiopu_Entropy_Coding-Based_Lossless_Compression_of_Asynchronous_Event_Sequences_CVPRW_2023_paper.pdf)
- [MoveEnet: Online High-Frequency Human Pose Estimation with an Event Camera](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Goyal_MoveEnet_Online_High-Frequency_Human_Pose_Estimation_With_an_Event_Camera_CVPRW_2023_paper.pdf)
- [Sparse-E2VID: A Sparse Convolutional Model for Event-Based Video Reconstruction Trained with Real Event Noise](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Cadena_Sparse-E2VID_A_Sparse_Convolutional_Model_for_Event-Based_Video_Reconstruction_Trained_CVPRW_2023_paper.pdf)
- [Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution](https://arxiv.org/pdf/2303.13767)
- [Recurrent Vision Transformers for Object Detection with Event Cameras](https://arxiv.org/pdf/2212.05598)
- [Neuromorphic Drone Detection: an Event-RGB Multimodal Approach](https://arxiv.org/pdf/2409.16099)
- [EventSleep: Sleep Activity Recognition with Event Cameras](https://arxiv.org/pdf/2404.01801)
- [High Speed and High Dynamic Range Video with an Event Camera](https://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf)
  - https://rpg.ifi.uzh.ch/E2VID.html

### Datasets

- [DAVIS UZH: The Event-Camera Dataset and Simulator](https://rpg.ifi.uzh.ch/davis_data.html)
- [PEDRo-Event-Based-Dataset]()
- [N-Youtube]()

### Videos / Teachings

- [TU Berlin course](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision):
- [CVPR 2023 Workshop: Dr. Christoph Posch (Prophesee). Event sensors for embedded edge AI vision applications](https://www.youtube.com/watch?v=wRWCwJBF534&ab_channel=RPGWorkshops)
- [CVPR 2023 Workshop: Edge Vision: it's about Time](https://www.youtube.com/watch?v=aZr_F-Ne75k&ab_channel=RPGWorkshops)
- [CVPR 2023 Workshop: Dr. Ryad Benosman (Meta) The Interplay Between Events and Frames: A Comprehensive Explanation.](https://www.youtube.com/watch?v=skPW4igKWOo&ab_channel=RPGWorkshops)

## 🗺️ Research / Learnings Roadmap

1. Complete exercises and review notes for [TU Berlin course](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision):
   - https://github.com/tallamjr/tuberlin-ebrv/tree/master
2. Download sample dataset and run simple event-to-image algorithms
3. Complete re-write of video-to-event simulator to better understand data.
