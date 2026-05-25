# Simulation and Digital Testing: CARLA, SUMO, and Scenario Generation

## Abstract

Simulation-based testing is the cornerstone of autonomous vehicle validation, providing the scale, controllability, and safety that physical testing alone cannot achieve. This paper provides a comprehensive survey of simulation and digital testing methodologies for autonomous vehicle systems, focusing on the CARLA and SUMO simulation platforms, scenario generation techniques, and the broader simulation ecosystem. We analyze CARLA's photorealistic rendering, sensor simulation, and flexible actor control, contrasting it with SUMO's microscopic traffic flow modeling and large-scale network simulation capabilities. The paper examines scenario generation approaches—from knowledge-based and search-based methods to data-driven and adversarial generation—and their role in systematically covering the operational design domain. Co-simulation architectures that combine multiple simulators (e.g., CARLA for ego-vehicle dynamics and perception, SUMO for background traffic) are analyzed for their ability to create comprehensive testing environments. We further discuss the challenges of simulation-to-reality (sim2real) transfer, simulation fidelity, computational cost, and regulatory acceptance of simulation-based evidence. The findings demonstrate that simulation is not merely a development tool but a critical component of the AV safety assurance case, requiring the same rigor as the vehicle system itself.

---

## Table of Contents

1. Introduction
2. CARLA Simulator
3. SUMO Traffic Simulator
4. Co-Simulation Architectures
5. Scenario Generation
6. Key Concepts
7. Methodologies
8. Challenges
9. Key References
10. Future Directions
11. Relevance to AVCS

---

## 1. Introduction

Autonomous vehicles must be validated across billions of driving scenarios, far exceeding what is feasible through physical road testing alone. RAND Corporation estimates that demonstrating AV safety at a 95% confidence level would require 8.8 billion miles of driving—a practical impossibility. Simulation-based testing addresses this challenge by enabling:

- **Scale**: Testing thousands of scenarios per hour on compute clusters.
- **Control**: Precisely specifying test conditions, actor behaviors, and environmental factors.
- **Safety**: Testing dangerous scenarios (e.g., pedestrian dart-outs, head-on collisions) without risk.
- **Repeatability**: Running identical scenarios to verify fixes and perform regression testing.
- **Coverage**: Systematically exploring the space of possible scenarios to identify weaknesses.

However, simulation is only as valuable as its fidelity to reality. A simulator that produces unrealistic sensor data, traffic behavior, or vehicle dynamics may generate false confidence in system performance. Understanding the capabilities, limitations, and appropriate use of different simulation platforms is essential for effective AV validation.

---

## 2. CARLA Simulator

### 2.1 Overview

CARLA (Car Learning to Act) is an open-source autonomous driving simulator developed by the Computer Vision Center at Universitat Autònoma de Barcelona, built on Unreal Engine 4. It provides:

- **Photorealistic rendering**: Urban, suburban, and highway environments with dynamic weather and lighting.
- **Sensor simulation**: RGB cameras, depth cameras, semantic segmentation, LiDAR (with configurable beam count, rotation frequency, and noise models), radar, GNSS, and IMU.
- **Flexible actor control**: Python and C++ APIs for controlling ego vehicles, background traffic, pedestrians, and traffic signals.
- **Scenario runner**: Built-in support for the NHTSA pre-crash scenario typology and custom scenario definitions.
- **ROS integration**: Native ROS and ROS2 bridges for connecting autonomous driving stacks.

### 2.2 Towns and Maps

CARLA provides multiple built-in maps (Town01 through Town10HD) representing:

- Dense urban cores with multi-lane intersections
- Suburban residential areas
- Highway segments with on/off ramps
- Open roads with varying curvature

Maps include annotated waypoints, lane markings, traffic signal positions, and speed limits, enabling both high-fidelity rendering and semantic-level navigation.

### 2.3 Weather and Lighting

Dynamic weather control includes:

- Cloudiness, precipitation, wind intensity
- Fog density and distance
- Sun position (time of day) and orientation
- Wetness of road surfaces (affecting reflections and grip)

These parameters can be varied continuously, enabling systematic testing of perception robustness under adverse conditions.

### 2.4 Sensor Simulation Fidelity

CARLA's sensor models range from idealized to realistic:

- **Camera**: Based on Unreal Engine rendering with configurable FOV, exposure, and lens effects. Post-processing adds motion blur, bloom, and lens flares.
- **LiDAR**: Ray-casting with configurable noise models (drop-off at distance, cross-talk, atmospheric attenuation). Limited by Unreal Engine's geometry resolution.
- **Radar**: Simplified Doppler model with configurable field of view and angular resolution. Less physically accurate than dedicated radar simulators.

### 2.5 Traffic and Pedestrian Behavior

CARLA's traffic manager provides configurable vehicle and pedestrian behaviors:

- Percentage of vehicles running red lights or speeding
- Pedestrian crossing probability and speed
- Inter-vehicle gap settings
- Lane-change behavior

However, CARLA's traffic behavior is scripted rather than learned, limiting its realism for testing interaction-aware planning algorithms.

---

## 3. SUMO Traffic Simulator

### 3.1 Overview

SUMO (Simulation of Urban MObility) is an open-source microscopic traffic simulation package developed by the German Aerospace Center (DLR). Unlike CARLA's focus on ego-vehicle perception and dynamics, SUMO specializes in:

- **Large-scale traffic flow**: Simulating thousands of vehicles across city-sized road networks.
- **Microscopic vehicle models**: Car-following (Krauss, IDM, Wiedemann) and lane-changing (LC2013, SL2015) models.
- **Traffic signal control**: Pre-timed, actuated, and custom signal plans with full phase timing control.
- **Route demand modeling**: Origin-destination matrices, turn probabilities, and flow definitions.

### 3.2 Network Models

SUMO networks are defined in a custom XML format (.net.xml) that includes:

- Lane-level road geometry
- Junction topology and right-of-way rules
- Traffic signal plans and timing
- Speed limits and road classes

Networks can be imported from OpenStreetMap using SUMO's netconvert tool, enabling rapid creation of realistic road networks for any city.

### 3.3 Traffic Demand

Traffic demand in SUMO is specified through:

- **Route files**: Explicit vehicle departure times and routes.
- **Flow definitions**: Continuous vehicle generation with specified origins, destinations, and rates.
- **Origin-destination matrices**: Aggregate trip patterns distributed across the network.
- **Activity-based models**: Sophisticated demand generation based on population demographics and land use.

### 3.4 Vehicle Dynamics Models

SUMO implements several car-following models:

- **Krauss model**: The default; based on safe following distance with stochastic driver imperfection.
- **Intelligent Driver Model (IDM)**: Smooth acceleration/deceleration with desired speed and time-gap parameters.
- **Wiedemann model**: Psycho-physical model used in VISSIM, calibrated from real driving data.
- **ACC/CACC models**: Adaptive Cruise Control models for simulating connected and automated vehicles.

### 3.5 SUMO-GUI and Analysis Tools

SUMO provides a graphical interface (SUMO-GUI) for visualization and a comprehensive analysis toolkit:

- **edgeData/laneData**: Aggregate traffic metrics (flow, speed, occupancy) by road segment.
- **tripinfo**: Per-vehicle trip statistics (travel time, waiting time, route length).
- **emission modeling**: Fuel consumption and emissions based on vehicle type and speed profile.

---

## 4. Co-Simulation Architectures

### 4.1 CARLA-SUMO Co-Simulation

The CARLA-SUMO co-simulation bridge (CARLA's built-in feature) combines:

- **CARLA's strengths**: Photorealistic sensor simulation, ego-vehicle dynamics, pedestrian simulation.
- **SUMO's strengths**: Large-scale traffic flow, realistic traffic signal control, network-level demand modeling.

The bridge synchronizes vehicle states between simulators: SUMO manages background traffic while CARLA renders the visual scene and simulates sensors. This combination enables testing AV perception in realistic traffic contexts.

### 4.2 Functional Mockup Interface (FMI)

The FMI standard enables co-simulation of heterogeneous models:

- Vehicle dynamics models (CarMaker, veDYNA) for high-fidelity chassis simulation.
- Traffic models (SUMO, VISSIM) for realistic traffic flow.
- Sensor models (SensorSim, VTD) for high-fidelity sensor simulation.
- Control models (Simulink, Modelica) for controller testing.

FMI-compliant models can be orchestrated through co-simulation masters (e.g., Maestro, FMI Go) that manage time synchronization and data exchange.

### 4.3 Cloud-Based Simulation

Large-scale AV testing requires running thousands of simulation instances in parallel:

- **AWS/Azure/GCP simulation farms**: Running CARLA instances on GPU-equipped cloud VMs.
- **Containerized simulation**: Docker images of CARLA and SUMO for reproducible, scalable deployment.
- **Continuous integration**: Automated simulation runs triggered by code changes, providing rapid feedback.

---

## 5. Scenario Generation

### 5.1 Knowledge-Based Generation

Scenarios are derived from domain knowledge and safety standards:

- **NHTSA pre-crash scenarios**: 37 pre-crash scenario types covering the most common crash configurations.
- **Euro NCAP test protocols**: Standardized test procedures for AEB, LKA, and other ADAS functions.
- **PEGASUS project**: German initiative that defined a six-layer scenario model (road, traffic, environment, communication, temporal, behavioral).

### 5.2 Search-Based Generation

Optimization and search algorithms explore the scenario space to find failure-inducing conditions:

- **Adaptive stress testing**: Using reinforcement learning to find the most likely paths to system failure.
- **Falsification**: Searching for inputs that violate temporal logic specifications.
- **Bayesian optimization**: Efficiently exploring high-dimensional parameter spaces to find worst-case scenarios.

### 5.3 Data-Driven Generation

Real-world driving data is used to generate realistic test scenarios:

- **Scenario mining**: Extracting scenarios from naturalistic driving datasets (e.g., NDD, Waymo Open Dataset).
- **Generative models**: Using VAEs or diffusion models to generate novel scenarios with realistic traffic patterns.
- **Augmentation**: Modifying real scenarios by varying parameters (speeds, distances, timings) to increase coverage.

### 5.4 Adversarial Generation

Scenarios are specifically designed to challenge the AV system:

- **Adversarial perturbations**: Adding subtle modifications to sensor inputs that cause misclassification.
- **Edge case generation**: Creating rare but critical scenarios (e.g., pedestrian appearing from behind a truck).
- **Behavioral adversarial agents**: Simulating road users that actively try to cause the AV to fail.

---

## 6. Key Concepts

| Concept | Description |
|---------|-------------|
| Simulation Fidelity | The degree to which a simulation matches real-world behavior |
| Sim2Real Gap | The discrepancy between simulation performance and real-world performance |
| Scenario Generation | Systematic creation of test scenarios for AV validation |
| Co-Simulation | Running multiple simulators in a synchronized, integrated manner |
| Microscopic Traffic Simulation | Modeling individual vehicle movements and interactions |
| Photorealistic Rendering | Generating synthetic images that closely resemble real photographs |
| Scenario Mining | Extracting test scenarios from recorded real-world driving data |
| Adaptive Stress Testing | Using RL to find the most likely failure scenarios |
| Falsification | Searching for inputs that violate formal safety specifications |
| Operational Design Domain (ODD) | The conditions under which an AV is designed to operate |

---

## 7. Methodologies

### 7.1 Simulation Fidelity Assessment

- **Sensor fidelity comparison**: Comparing synthetic sensor data against real-world recordings using SSIM, FID, and task-specific metrics.
- **Traffic behavior validation**: Comparing simulated traffic flow patterns against real-world loop detector data.
- **Vehicle dynamics validation**: Comparing simulated vehicle responses against proving ground measurements.

### 7.2 Scenario Coverage Metrics

- **Parameter space coverage**: Measuring the fraction of the ODD parameter space covered by test scenarios.
- **Functional coverage**: Tracking which AV functions (lane keeping, AEB, intersection navigation) have been tested.
- **Risk-weighted coverage**: Prioritizing coverage of high-risk scenario categories based on crash statistics.

### 7.3 Regression Testing

- **Baseline comparison**: Running the same scenario suite before and after code changes to detect regressions.
- **Golden data comparison**: Comparing AV outputs against reference outputs for fixed scenarios.
- **Statistical change detection**: Identifying statistically significant performance changes across scenario distributions.

### 7.4 Regulatory Evidence Generation

- **Scenario-based test reports**: Documented test results for each regulatory-required scenario.
- **Statistical evidence packages**: Aggregated pass/fail statistics across scenario suites.
- **Simulation qualification**: Evidence that the simulation platform itself produces valid results for the intended use.

---

## 8. Challenges

### 8.1 Sim2Real Gap

The most fundamental challenge: AVs that perform well in simulation may fail in the real world due to:

- **Sensor model limitations**: Synthetic images may lack the noise, artifacts, and variability of real cameras.
- **Behavioral gap**: Simulated pedestrians and drivers are less unpredictable than real humans.
- **Dynamic model simplifications**: Vehicle dynamics models may not capture tire slip, suspension effects, or road surface variations.
- **Environmental variability**: Real weather, lighting, and road conditions are more diverse than simulation parameters.

### 8.2 Computational Cost

High-fidelity simulation is computationally expensive:

- CARLA with full sensor suites requires a high-end GPU per instance.
- Running millions of scenarios for statistical validation requires massive compute clusters.
- Simulation cost can exceed physical testing cost if not carefully managed.

### 8.3 Scenario Space Explosion

The number of possible driving scenarios is effectively infinite. Even with systematic generation, determining when "enough" testing has been done remains an open question.

### 8.4 Traffic Behavior Realism

SUMO's car-following models produce reasonable aggregate traffic flow but individual vehicle behaviors may be unrealistic (e.g., no aggressive driving, no distracted driving, no cultural driving norms).

### 8.5 Multi-Agent Scenario Testing

Testing scenarios involving multiple interacting AVs (platooning, cooperative merging) requires simulating all AVs simultaneously, increasing computational cost and scenario complexity.

### 8.6 Simulation Qualification

For simulation results to be accepted as regulatory evidence, the simulation platform itself must be qualified—demonstrated to produce valid results for the specific use case. Simulation qualification methodology is still developing.

### 8.7 Data Privacy in Scenario Mining

Mining scenarios from real-world data raises privacy concerns (vehicle trajectories, pedestrian identities). Anonymization and privacy-preserving scenario extraction methods are needed.

---

## 9. Key References

1. Dosovitskiy, A., et al. (2017). "CARLA: An Open Urban Driving Simulator." *CoRL*.
2. Lopez, P. A., et al. (2018). "Microscopic Traffic Simulation using SUMO." *IEEE ITSC*.
3. Kalra, N., & Paddock, S. M. (2016). "Driving to Safety." *RAND Corporation*.
4. PEGASUS Project. (2019). "Final Report of the PEGASUS Project." *German Federal Ministry for Economic Affairs*.
5. Koren, M., et al. (2018). "Adaptive Stress Testing of Autonomous Vehicles." *IEEE IV*.
6. Fremont, D. J., et al. (2020). "Scenic: A Language for Scenario Specification and Scene Generation." *PLDI*.
7. Feng, D., et al. (2021). "Deep Active Learning for Efficient Training of LiDAR Detection Systems." *IEEE RA-L*.
8. Grewal, G. S., & Song, B. (2021). "Survey of Scenario-Based Testing for Autonomous Vehicles." *SAE Technical Paper*.
9. Riedmaier, S., et al. (2020). "Survey on Scenario-Based Safety Assessment of Automated Vehicles." *IEEE Access*.
10. O'Kelly, M., et al. (2020). "Scalable End-to-End Autonomous Vehicle Testing via Rare-Event Simulation." *NeurIPS*.

---

## 10. Future Directions

### 10.1 Neural Rendering

Using neural radiance fields (NeRF) and Gaussian splatting to create photorealistic simulated environments from real-world image data, dramatically improving sim2real alignment for visual perception.

### 10.2 Learned Traffic Behavior

Replacing SUMO's rule-based car-following models with learned behavior models trained on real driving data, producing more realistic and diverse traffic patterns.

### 10.3 Digital Twin Simulation

Real-time digital twins of actual road networks, continuously updated from sensor data, enabling simulation of specific real-world locations with current conditions.

### 10.4 Generative AI for Scenario Creation

Using large language models and diffusion models to generate scenario descriptions and visual scenes from natural language specifications.

### 10.5 Regulatory Simulation Standards

Developing industry standards for simulation platform qualification, scenario quality metrics, and simulation evidence packages that regulators will accept as valid safety evidence.

---

## 11. Relevance to AVCS

Simulation and digital testing are integral to the AVCS development lifecycle:

- **Continuous Integration Testing**: Every AVCS code change triggers automated CARLA-SUMO co-simulation runs across a regression scenario suite, catching defects before they reach the vehicle.
- **Perception Validation**: AVCS perception modules are tested against CARLA-generated sensor data with systematic weather, lighting, and occlusion variations to ensure robustness.
- **Planning Scenario Coverage**: AVCS planning algorithms are stress-tested through adversarial scenario generation, identifying failure modes before deployment.
- **Co-Simulation Framework**: AVCS connects to the CARLA-SUMO co-simulation bridge through its ROS2 interface, enabling seamless transition from simulation to vehicle deployment.
- **Safety Case Evidence**: Simulation test results form a critical component of the AVCS safety case, providing documented evidence of systematic scenario coverage.
- **Hardware-in-the-Loop**: AVCS runs on target hardware connected to CARLA for SIL/HIL testing, validating that real-time performance constraints are met.
- **Fleet Learning Loop**: Scenarios where real AVCS-equipped vehicles encounter difficulties are replayed in simulation for root cause analysis and algorithm improvement.
- **Scenario Regression Suite**: A curated library of critical scenarios maintained and expanded throughout AVCS development, ensuring that improvements do not introduce regressions.

Without comprehensive simulation, AVCS cannot achieve the safety assurance required for public deployment. Simulation is the validation backbone of the entire autonomous driving system.

---

*Document Version: 1.0 | Last Updated: 2025-03-04 | Classification: Research Paper*
