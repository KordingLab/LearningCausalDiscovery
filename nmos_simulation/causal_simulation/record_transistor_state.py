import numpy as np
from sim2600 import sim2600Console
from sim2600 import params, sim6502, simTIA


def record_regular_transistor_state(rom=params.ROMS_DONKEY_KONG, num_iterations=1000):
    """
    Record the transistor state for a regular simulation
    """
    sim = sim2600Console.Sim2600Console(rom, simTIAfactory=simTIA.MySimTIA,
                                        sim6502factory=sim6502.MySim6502)
    transistor_history = []

    sim.sim6507.setRecording(True)
    for step in range(num_iterations):
        # Run the simulation for one iteration
        sim.advanceOneHalfClock()

        transistor_history.append(sim.sim6507.getFullTransistorState())

    transistor_history = transistor_history[0] if len(transistor_history) == 1 else np.concatenate(transistor_history, axis=1)
    return transistor_history


def single_transistor_perturbation(tidx=None, perturb_step=None, perturb_type=None, rom=params.ROMS_DONKEY_KONG, num_iterations=1000):
    """
    Perturb a single transistor and record the transistor states
    """
    assert tidx is not None and perturb_step is not None and perturb_type is not None, \
        "tidx and perturb_step and perturb_type must be specified"
    assert perturb_step < num_iterations, "perturb_step must be smaller than num_iterations"

    sim = sim2600Console.Sim2600Console(rom, simTIAfactory=simTIA.MySimTIA,
                                        sim6502factory=sim6502.MySim6502)

    # perturbation happens between perturb_step - 1 and perturb_step
    for step in range(num_iterations):
        # Run the simulation for one iteration
        sim.advanceOneHalfClock()

        # Record the transistor states, only consider the next half-clock
        if step == perturb_step:
            state = sim.sim6507.getFullTransistorState()
            break

        # Perturb on a transistor if needed
        if step == perturb_step - 1:
            sim.sim6507.intervenTransistor(tidx, perturb_type)
            sim.sim6507.setRecording(True)

    return state