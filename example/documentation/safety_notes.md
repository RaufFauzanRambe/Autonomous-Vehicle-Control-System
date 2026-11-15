# Safety Notes

Scope, limitations and responsible-use guidance for the example suite.
Read this before using any output in a publication, benchmark or
derivative system.

## Simulation-only scope

Every module in this repository is a **teaching simulation**. Nothing
here drives hardware, publishes to a vehicle network, or should be
treated as a safety-certified component. There is no fail-operational
design, no watchdog, no redundancy, and no validation evidence of the
kind required for real autonomous systems (ISO 26262, ISO 21448 / SOTIF,
or regional equivalents).

## Model limitations

* **Dynamics (01, 06, 08).** The kinematic bicycle model ignores tire
  slip, load transfer, suspension, pitch/roll and actuator latency. It
  is valid qualitatively for low-acceleration passenger-car maneuvers
  and breaks down near handling limits, at high steering angles at
  speed, and for heavy vehicles. Euler integration adds discretization
  error proportional to `dt`.
* **Sensors (02).** Synthetic noise is white and Gaussian; real sensors
  exhibit correlated noise, dropout, miscalibration, and structured
  artifacts (motion blur, multipath, lidar grazing angles). Target
  models are cartoon stand-ins, not scene simulations.
* **Perception (03).** Classical CV heuristics (thresholds, Hough
  lines, color dominance) are brittle to lighting, weather and occlusion.
  Confidence values are heuristics, not calibrated probabilities.
* **Localization (04).** The complementary filter assumes small
  headings, unbiased-after-correction IMU data and continuous GPS.
  During outages the estimate drifts quadratically; the shipped dataset
  is too short to expose long-run divergence.
* **Planning (05).** A* on a coarse graph returns a route, not a
  kinematically feasible trajectory; no lane boundaries, no dynamic
  obstacles, no traffic rules beyond node connectivity.
* **Control (06).** Gains are hand-tuned for the shipped scenarios only.
  There is no stability proof, no constraint handling and no actuator
  saturation model beyond clipping.
* **Decisions (07).** Four ordered rules over a curated benchmark;
  behavior outside the covered cases is undefined. Rule order embodies
  a priority judgment (signals > vulnerable road users > car-following)
  that must be reconsidered, not reused, in any other context.

## Responsible use in research

* Treat all reported numbers (RMSE, cross-track error, decision matches)
  as illustrative of the *method*, not as performance claims about any
  real system or sensor.
* When comparing algorithms, fix `--seed` and report the seed; a single
  seed is not a statistical result. The suite is deliberately cheap to
  run across many seeds.
* Cite the model class (kinematic bicycle, complementary filter, A*)
  rather than implying validated autonomy.
* If you connect any derived code to physical actuators or test
  vehicles, you are solely responsible for a full safety validation
  that this repository does not provide.

## Data privacy and licensing hygiene

The shipped datasets are synthetic and contain no personal data, no
real license plates, no real map imagery. When you substitute real
recordings (camera frames, GNSS logs, maps), apply your dataset's
license and privacy obligations before redistribution — this
repository's structure and docs impose no extra terms of their own.

## Failure behaviour to expect

Modules raise exceptions on invalid input (missing files, unknown node
ids, malformed configs) rather than silently degrading. In an
integration of your own, add explicit degradation policy: the demo
components in 08 intentionally keep running on GPS noise and empty
perception lists, which real systems must not do unconditionally.

## Reporting issues

If a shipped scenario produces a physically implausible result (e.g.
negative speeds, unbounded cross-track growth), check `dt` first, then
the parameter comments in the module YAML; the defaults are chosen so
every shipped scenario converges. Document any parameter change
alongside your experiment logs so runs stay comparable.
