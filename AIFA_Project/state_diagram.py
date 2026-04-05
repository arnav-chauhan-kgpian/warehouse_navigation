"""
state_diagram.py — Robot Finite State Machine & State Diagrams
================================================================
Defines the robot FSM states, transitions, and generates Mermaid
state diagrams for documentation and debugging.
"""

from enum import Enum, auto
from collections import defaultdict


class RobotState(Enum):
    """States in the robot pick-and-place lifecycle."""
    IDLE = auto()
    MOVING_TO_PICKUP = auto()
    PICKING_UP = auto()
    MOVING_TO_DROPOFF = auto()
    DROPPING_OFF = auto()

    def label(self):
        return self.name.replace("_", " ").title()


# ── Valid transitions ────────────────────────────────────────────────────
TRANSITIONS = {
    RobotState.IDLE:              [RobotState.MOVING_TO_PICKUP],
    RobotState.MOVING_TO_PICKUP:  [RobotState.PICKING_UP],
    RobotState.PICKING_UP:        [RobotState.MOVING_TO_DROPOFF],
    RobotState.MOVING_TO_DROPOFF: [RobotState.DROPPING_OFF],
    RobotState.DROPPING_OFF:      [RobotState.IDLE],
}

STATE_COLORS = {
    RobotState.IDLE:              (100, 149, 237),   # Cornflower blue
    RobotState.MOVING_TO_PICKUP:  (34, 197, 94),     # Green
    RobotState.PICKING_UP:        (251, 191, 36),     # Amber
    RobotState.MOVING_TO_DROPOFF: (168, 85, 247),     # Purple
    RobotState.DROPPING_OFF:      (239, 68, 68),      # Red
}


class RobotFSM:
    """
    Finite State Machine for a single warehouse robot.
    Validates transitions and logs state history.
    """

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.state = RobotState.IDLE
        self.history = [(0, RobotState.IDLE)]  # (timestep, state)

    def transition(self, new_state, timestep=None):
        """
        Transition to *new_state*.  Raises ValueError if invalid.
        """
        if new_state not in TRANSITIONS.get(self.state, []):
            raise ValueError(
                f"Robot {self.robot_id}: invalid transition "
                f"{self.state.name} → {new_state.name}"
            )
        self.state = new_state
        self.history.append((timestep, new_state))

    def can_transition(self, new_state):
        return new_state in TRANSITIONS.get(self.state, [])

    @property
    def color(self):
        return STATE_COLORS.get(self.state, (200, 200, 200))


# ── Mermaid diagram generation ───────────────────────────────────────────

def generate_mermaid_fsm():
    """Return a Mermaid stateDiagram-v2 string for the robot FSM."""
    lines = [
        "stateDiagram-v2",
        "    [*] --> IDLE",
        "",
    ]
    for src, dsts in TRANSITIONS.items():
        for dst in dsts:
            lines.append(f"    {src.name} --> {dst.name}")
    lines.append("")
    lines.append("    DROPPING_OFF --> [*]")
    lines.append("")

    # State descriptions
    descriptions = {
        "IDLE":              "Robot waiting\\nfor task assignment",
        "MOVING_TO_PICKUP":  "Navigating to\\npickup location",
        "PICKING_UP":        "Picking up item\\n(1 timestep)",
        "MOVING_TO_DROPOFF": "Carrying item\\nto drop-off",
        "DROPPING_OFF":      "Placing item\\n(1 timestep)",
    }
    for state_name, desc in descriptions.items():
        lines.append(f'    {state_name} : {desc}')

    return "\n".join(lines)


def generate_snapshot_diagram(robots_fsm, timestep):
    """
    Generate a Mermaid diagram showing current state of all robots
    at a specific timestep — a 'state diagram at a crucial turn'.
    """
    lines = [
        f"stateDiagram-v2",
        f"    direction LR",
        f"    note right of [*] : Timestep {timestep}",
        "",
    ]
    for rid, fsm in sorted(robots_fsm.items()):
        state_name = fsm.state.name
        lines.append(f"    state Robot_{rid} {{")
        lines.append(f"        [*] --> {state_name}")
        lines.append(f"        {state_name} : {fsm.state.label()}")
        lines.append(f"    }}")
        lines.append("")

    return "\n".join(lines)


def print_state_diagram():
    """Print the FSM state diagram to console."""
    print("\n" + "=" * 60)
    print("  ROBOT FINITE STATE MACHINE — State Diagram (Mermaid)")
    print("=" * 60)
    print()
    print("```mermaid")
    print(generate_mermaid_fsm())
    print("```")
    print()
    print("=" * 60 + "\n")


def print_snapshot(robots_fsm, timestep):
    """Print a state snapshot at a crucial timestep."""
    print(f"\n--- State Snapshot at Timestep {timestep} ---")
    for rid, fsm in sorted(robots_fsm.items()):
        print(f"  Robot {rid}: {fsm.state.label()}")
    print()
