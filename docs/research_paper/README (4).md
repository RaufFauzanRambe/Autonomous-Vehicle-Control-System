# Diagrams

This directory contains system diagrams for the Autonomous Vehicle Control System.

## Diagram Types

| Diagram | Format | Description |
|---------|--------|-------------|
| System Architecture | Mermaid / PlantUML | Top-level module architecture |
| Data Flow | Mermaid | Sensor-to-actuator data pipeline |
| State Machines | Mermaid | FSM diagrams for behavior planning |
| Sequence Diagrams | Mermaid | Inter-module communication sequences |
| Class Diagrams | PlantUML | C++ class hierarchy and relationships |
| Deployment Diagram | PlantUML | Hardware and software deployment |

## Generating Diagrams

```bash
# Generate all Mermaid diagrams to PNG
npx @mermaid-js/mermaid-cli -i diagrams/ -o diagrams/output/

# Generate PlantUML diagrams
java -jar plantuml.jar diagrams/*.puml

# Generate all using Makefile
make docs-diagrams
```

## Diagram Files

- `system_architecture.mmd` - Top-level system architecture
- `perception_pipeline.mmd` - Perception module data flow
- `localization_pipeline.mmd` - Localization module data flow
- `behavior_fsm.mmd` - Behavior planning state machine
- `control_loop.mmd` - Real-time control loop sequence
- `safety_monitor.mmd` - Safety monitoring and escalation

> **Note**: Mermaid diagram source files use the `.mmd` extension.
> PlantUML diagram source files use the `.puml` extension.
