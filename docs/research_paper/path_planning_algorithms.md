# Path Planning Algorithms for Autonomous Vehicles: A*, RRT, Lattice Planners, Hybrid A*, and Anytime Repair A*

## Abstract

Path planning is a fundamental capability of autonomous vehicles, generating safe, comfortable, and efficient trajectories from origin to destination while avoiding obstacles and respecting traffic rules. This research summary provides a comprehensive examination of path planning algorithms used in autonomous driving, from classical search-based methods (A*, Hybrid A*, Anytime Repair A*) through sampling-based approaches (RRT, RRT*, PRM) to optimization-based and learning-based methods. We analyze lattice-based planners that operate on discretized motion primitives, discuss the trade-offs between completeness, optimality, and computational efficiency, and examine recent advances in learning-augmented planning. The hierarchical nature of autonomous driving planning—mission planning, behavioral planning, and motion planning—is discussed, with emphasis on how different algorithms are suited to different planning levels. The relevance of these algorithms to the Autonomous Vehicle Control System (AVCS) is examined throughout.

## Key Concepts

### Planning Hierarchy
Autonomous driving planning is typically decomposed into three levels:
- **Mission planning (routing)**: Finding the road-level route from origin to destination using road network graphs (Dijkstra, A*)
- **Behavioral planning (decision making)**: Determining high-level driving maneuvers (lane change, merge, yield) using finite state machines, decision trees, or POMDPs
- **Motion planning (trajectory generation)**: Generating smooth, collision-free, dynamically feasible trajectories using search, sampling, or optimization

### Configuration Space
Path planning operates in configuration space (C-space):
- **Configuration**: A complete specification of the robot's pose (x, y, θ for a car)
- **C-space obstacles**: Regions of C-space where the vehicle collides with obstacles
- **Free space**: Regions of C-space that are collision-free
- **C-space construction**: Computing the Minkowski sum of obstacle and vehicle geometries

### Completeness and Optimality
Important theoretical properties of planning algorithms:
- **Completeness**: Finding a path if one exists, reporting failure otherwise
- **Resolution completeness**: Finding a path if one exists, given sufficient resolution
- **Probabilistic completeness**: Finding a path with probability approaching 1 given enough time
- **Optimality**: Finding the minimum-cost path
- **Asymptotic optimality**: Converging to the optimal path as computation time increases

### Vehicle Kinematics and Dynamics
Planning must respect vehicle motion constraints:
- **Kinematic bicycle model**: Simplified vehicle model with non-holonomic constraints
- **Ackermann steering**: Realistic front-wheel steering geometry
- **Dubins car**: Minimum turning radius without reverse
- **Reeds-Shepp car**: Minimum turning radius with reverse
- **Dynamic constraints**: Acceleration, jerk, and curvature rate limits

### Cost Functions
Planning objectives encoded as cost functions:
- **Safety**: Distance to obstacles, collision risk
- **Comfort**: Smooth acceleration, jerk, curvature
- **Efficiency**: Travel time, energy consumption
- **Compliance**: Traffic rules, lane discipline
- **Progress**: Advancement toward the goal
- **Social compliance**: Courtesy to other road users

## State of the Art

### Search-Based Planning

#### A* Algorithm
A* is the foundational search algorithm for path planning:
- **Best-first search**: Expanding the node with lowest f(n) = g(n) + h(n)
- **Admissible heuristic**: h(n) never overestimates the true cost to the goal
- **Optimality**: Guaranteed to find the shortest path with an admissible heuristic
- **Applications in AV**: Road-level routing, grid-based planning
- **Limitations**: Discretization, memory for large spaces, not directly handling kinematics

#### Hybrid A*
Hybrid A* extends A* to respect vehicle kinematics:
- **Continuous state space**: States are continuous (x, y, θ) rather than grid cells
- **Motion primitives**: Expanding nodes using kinematically feasible motions
- **Heuristic**: Non-holonomic-without-obstacles heuristic (Reeds-Shepp or Dubins distance)
- **Analytic expansion**: Attempting direct Reeds-Shepp connection to the goal
- **Applications**: Parking, low-speed maneuvering, narrow passages
- **Limitations**: Resolution suboptimal, computational cost in cluttered environments

#### Anytime Repairing A* (ARA*)
ARA* provides anytime search with provable suboptimality bounds:
- **Inflated heuristic**: Using w*h(n) for faster initial solution
- **Iterative refinement**: Decreasing inflation factor for better solutions
- **Suboptimality bound**: Initial solution is within w* of optimal
- **Anytime property**: Can be interrupted at any time with a valid solution
- **Applications**: Real-time planning where computation time is limited

### Sampling-Based Planning

#### Rapidly-exploring Random Trees (RRT)
RRT builds a tree of feasible paths through random sampling:
- **Random expansion**: Growing the tree toward random samples
- **Kinematic feasibility**: Steering toward samples using vehicle model
- **Probabilistic completeness**: Guaranteed to find a path with enough samples
- **Limitations**: Non-optimal, jagged paths requiring post-processing
- **Variants**:
  - RRT-Connect: Bidirectional RRT for faster connection
  - RRT*: Asymptotically optimal version with rewiring
  - Informed RRT*: Focusing sampling in promising regions
  - anytime RRT*: Improving solution quality over time

#### Probabilistic Roadmaps (PRM)
PRM builds a roadmap of the free configuration space:
- **Learning phase**: Sampling random configurations and connecting nearby ones
- **Query phase**: Connecting start and goal to the roadmap and searching
- **Multi-query**: Roadmap can be reused for multiple planning queries
- **Limitations**: Not suitable for dynamic environments, narrow passage problem

### Lattice Planners
Lattice planners search over a discretized set of motion primitives:
- **Motion primitives**: Pre-computed, kinematically feasible short trajectories
- **Lattice structure**: Primitives connect a regular grid of states
- **Graph search**: A* or Dijkstra search over the lattice graph
- **Completeness**: Resolution complete given sufficient lattice resolution
- **Applications**: Highway driving, structured road environments
- **Limitations**: Discretization effects, lattice design complexity

### Optimization-Based Planning
Formulating planning as a numerical optimization problem:
- **Trajectory optimization**: Minimizing cost function over trajectory parameters
- **CHOMP (Covariant Hamiltonian Optimization for Motion Planning)**: Gradient-based optimization with collision costs
- **STOMP (Stochastic Trajectory Optimization for Motion Planning)**: Gradient-free optimization with stochastic sampling
- **Model Predictive Control (MPC)**: Receding-horizon optimization
- **Quadratic programming (QP)**: Convex optimization for polynomial trajectories
- **Sequential convex optimization**: Iteratively solving convex approximations

### Learning-Based Planning
Using machine learning to improve planning:
- **Neural motion planning**: Learning cost functions or planning policies
- **Neural A*": Learning heuristic functions for A* search
- **Imitation learning for planning**: Learning from expert driving trajectories
- **RL for planning**: Learning planning policies through interaction
- **Motion prediction integration**: Joint prediction and planning

## Methodologies

### Trajectory Representation
Different ways to represent planned trajectories:
- **Polynomial trajectories**: Minimum-jerk or minimum-snap polynomials
- **Spline trajectories**: Piecewise polynomial with continuity constraints
- **Bezier curves**: Parametric curves with intuitive control
- **Clothoid curves**: Curves with linearly varying curvature (Euler spirals)
- **Lattice connections**: Pre-computed motion primitive trajectories
- **Waypoint sequences**: Discrete points with interpolation

### Collision Checking
Efficient collision detection for planning:
- **Geometric checking**: Point-in-polygon, separating axis theorem
- **Occupancy grids**: Checking grid cells along the trajectory
- **Signed distance fields**: Pre-computed distance to nearest obstacle
- **Bounding volume hierarchies**: Hierarchical collision checking
- **Continuous collision detection**: Checking entire trajectory segments

### Speed Planning
Generating the speed profile along the path:
- **Path-speed decomposition**: First plan path, then plan speed
- **Spatio-temporal planning**: Joint path and speed planning
- **Quadratic programming**: Convex speed optimization
- **Graph-based speed search**: DP or A* search over speed profiles
- **Speed optimization objectives**: Safety, comfort, efficiency, compliance

### Behavioral Planning
High-level decision making for driving scenarios:
- **Finite state machines**: Transition between driving states
- **Decision trees**: Hierarchical decision logic
- **POMDPs**: Planning under uncertainty with partial observability
- **Rule-based systems**: Expert-coded driving rules
- **Learning-based approaches**: RL or IL for behavioral decisions

### Planning Under Uncertainty
Handling uncertainty in the planning process:
- **Uncertain predictions**: Planning over multiple prediction hypotheses
- **Robust planning**: Optimizing worst-case performance
- **Chance-constrained planning**: Probabilistic safety constraints
- **Contingency planning**: Planning multiple branches for different futures
- **Belief space planning**: Planning in the space of belief states

## Challenges

### Real-Time Performance
Planning must operate within strict time budgets:
- **Highway driving**: 100ms planning cycle
- **Urban driving**: 200ms planning cycle with more complexity
- **Parking**: Up to 1s for complex maneuvers
- **Emergency maneuvers**: < 50ms for collision avoidance

### Dynamic Environments
Planning around moving obstacles:
- **Prediction uncertainty**: Future trajectories are uncertain
- **Interactive agents**: Other road users react to the ego vehicle
- **Multi-agent coordination**: Multiple vehicles negotiating right-of-way
- **Temporal constraints**: Time-dependent obstacles and traffic signals

### Complex Driving Scenarios
Challenging scenarios for planning:
- **Unprotected left turns**: Crossing oncoming traffic
- **Merging in heavy traffic**: Finding gaps in dense traffic flow
- **Roundabouts**: Negotiating entry and exit with multiple agents
- **Narrow passages**: Constrained spaces with limited maneuverability
- **Construction zones**: Temporary changes to road layout

### Comfort and Naturalness
Generating comfortable and natural driving behavior:
- **Acceleration/jerk limits**: Smooth acceleration profiles
- **Lane change smoothness**: Gradual lateral transitions
- **Speed profile naturalness**: Human-like speed modulation
- **Social compliance**: Courteous driving behavior

### Multi-Objective Optimization
Balancing competing planning objectives:
- **Safety vs. progress**: Conservative vs. assertive driving
- **Comfort vs. efficiency**: Smooth vs. fast trajectories
- **Rule compliance vs. naturalness**: Strict vs. context-aware rule following
- **Pareto-optimal solutions**: No single optimal trade-off

## Recent Advances

### Learning-Augmented Search
Combining learning with classical search:
- **Neural A***: Learning better heuristics for A* search
- **Learned motion primitives**: Data-driven primitive generation
- **Learning to prune**: Neural network-based branch pruning
- **Attention-based planning**: Transformer models for trajectory generation

### Joint Prediction and Planning
Integrating prediction into the planning loop:
- **Game-theoretic planning**: Modeling agent interactions as games
- **Socially-aware planning**: Accounting for social norms and expectations
- **Interactive prediction**: Predictions that account for the ego vehicle's actions
- **Closed-loop planning**: Planning with closed-loop prediction

### Optimization with Neural Cost Maps
Using learned cost functions for trajectory optimization:
- **Neural cost maps**: Learned representations of planning costs
- **Differentiable planning**: End-to-end optimization of planning costs
- **Imitation learning costs**: Learning costs from expert demonstrations
- **RL-optimized costs**: Optimizing costs through reinforcement learning

### Safe Planning with Formal Guarantees
Planning with provable safety properties:
- **Control barrier functions**: Guaranteeing safety constraints
- **Hamilton-Jacobi reachability**: Computing reachable sets for verification
- **Satisficing planning**: Finding good-enough plans with safety guarantees
- **Runtime assurance**: Safety monitors that override unsafe plans

### Planning for Large Language Model Integration
Using LLMs for high-level planning:
- **Natural language navigation**: Interpreting natural language route instructions
- **LLM-based scene reasoning**: Understanding complex traffic situations
- **Explainable planning**: Generating natural language plan explanations
- **LLM-guided search**: Using LLM reasoning to guide search-based planning

## Key Papers/References

1. Dolgov, D., et al. (2010). "Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments." IJRR (Hybrid A*).
2. LaValle, S. (1998). "Rapidly-Exploring Random Trees: A New Tool for Path Planning." TR (RRT).
3. Karaman, S., & Frazzoli, E. (2011). "Sampling-based Algorithms for Optimal Motion Planning." IJRR (RRT*).
4. Likhachev, M., et al. (2003). "ARA*: Anytime A* with Provable Bounds on Sub-Optimality." NIPS.
5. Pivtoraiko, M., et al. (2009). "Differentially Constrained Mobile Robot Motion Planning in State Lattices." JFR.
6. Zucker, M., et al. (2013). "CHOMP: Covariant Hamiltonian Optimization for Motion Planning." IJRR.
7. Kalakrishnan, M., et al. (2011). "STOMP: Stochastic Trajectory Optimization for Motion Planning." ICRA.
8. McNaughton, M., et al. (2011). "Motion Planning for Autonomous Driving with a Conformal Spatiotemporal Lattice." ICRA.
9. Xu, W., et al. (2012). "Optimization-based Autonomous Driving in Complex Urban Environments." IROS.
10. Zeng, W., et al. (2019). "End-to-End Interpretable Neural Motion Planner." CVPR.
11. Sadat, A., et al. (2020). "Jointly Learnable Behavior and Trajectory Planning for Self-Driving." IV.
12. Huang, Z., et al. (2023). "Differentiable Integrated Motion Prediction and Planning." ICRA.
13. Le, T., et al. (2023). "Neural A*: A Neural Network-based A* Algorithm." ICLR.
14. Ross, S., et al. (2008). "AISTAR: Anytime Repairing A* for Real-Time Planning." ICAPS.
15. Fan, T., et al. (2021). "Context-Aware Motion Planning for Autonomous Driving." T-RO.

## Future Directions

### Foundation Models for Planning
Leveraging foundation models for zero-shot planning in novel scenarios, using pre-trained world knowledge for common-sense reasoning about driving situations.

### Real-Time Global Optimality
Developing algorithms that can find globally optimal or near-optimal paths in real-time for complex urban driving scenarios.

### Multi-Agent Cooperative Planning
Scalable algorithms for cooperative planning among multiple autonomous vehicles, enabling traffic flow optimization and conflict resolution.

### Continual Learning for Planning
Planning systems that improve from experience, learning better cost functions, heuristics, and behavioral models over time.

### Certified Safe Planning
Planning algorithms with formal safety certificates, proving that generated trajectories satisfy safety constraints under bounded uncertainty.

### Cognitive Planning
Planning algorithms that model and reason about the mental states of other road users (theory of mind), enabling more natural and cooperative driving behavior.

### Planning with Large Language Models
Integrating LLMs into the planning pipeline for high-level reasoning, natural language interaction, and common-sense decision making in novel scenarios.

## Relevance to AVCS

Path planning is central to the AVCS's navigation capability:

1. **Mission Planning**: A*-based routing provides the AVCS with efficient road-level route planning using real-time traffic information.

2. **Behavioral Planning**: The AVCS uses decision-theoretic approaches for high-level driving decisions such as lane changes, merges, and intersection navigation.

3. **Motion Planning**: Lattice-based and optimization-based planners generate smooth, collision-free trajectories that respect the AVCS's vehicle dynamics.

4. **Real-Time Operation**: Anytime algorithms like ARA* enable the AVCS to produce valid plans within its real-time planning budget, with quality improving as time allows.

5. **Parking and Maneuvering**: Hybrid A* enables the AVCS to plan complex low-speed maneuvers in tight spaces such as parking lots.

6. **Safety Guarantees**: Control barrier function-based safety monitors ensure the AVCS's planned trajectories never violate safety constraints.

7. **Multi-Objective Optimization**: The AVCS balances safety, comfort, efficiency, and compliance through configurable cost functions in its planning optimization.

8. **Adaptive Planning**: Learning-augmented planning enables the AVCS to improve its planning behavior from experience, adapting to local driving norms and conditions.
