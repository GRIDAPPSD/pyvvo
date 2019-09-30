# Tasks
The intent of this file is to document outstanding tasks for PyVVO.
This document is still in a draft state.

## Monitor topology changes, update model, kill optimization
### Summary
When the system topology changes, PyVVO's internal model needs
to be updated, and the optimization may need re-triggered.

### Details
The detection of topology changes can be done via an EquipmentManager
that manages all switches in the system. Maybe the update_state method
simply needs augmented to call some sort of callback when equipment
changes. Since the individual EquipmentSinglePhase objects track their
state and previous state, this should be pretty simple.

Once the EquipmentManager instance has reported a change, the "master"
GLMManager should go through and update the switches in PyVVO's internal
model of the system. 

The solution here might be to create a higher level class which takes
in all the EquipmentManagers (e.g. one for regulators, one for
capacitors, one for switches, etc.) and the "master" GLMManager and 
keeps the model updated. This higher level class could also report 
when equipment changes state for use in managing the running
optimization.

The final piece here is deciding when to kill the optimization and start
again. Some faults will be cleared quickly, so there's no sense in 
immediately killing any running optimization. So really, this final
piece is about tracking status changes over time, and if switches have
not returned to their pre-event state, we need to re-trigger
optimization.

## Command Tracking
### Summary
One event that the application needs to handle is unresponsive
equipment. E.g., you send a tap position command to a regulator but it
doesn't change it's position.

### Details
There are a few pieces to this puzzle. First is, of course, detection.
We need a mechanism for monitoring equipment that has recently had a 
command sent out. This could involve augmenting the EquipmentManager to
be the one that actually sends out commands related to its equipment,
and then watches for the change. What I don't like about this is the 
EquipmentManager then needs to be capable of querying the platform.

Perhaps a better approach would be to make some sort of class which has
both a PlatformManager and an EquipmentManager as attributes, and
ensures that those changes actually happen.

After detection has occurred, we need to flag the equipment as out of 
service. This could be as simple as adding some sort of flag to
EquipmentSinglePhase tracking if the object is currently operable or 
not. The EquipmentSinglePhase class already has a "controllable"
attribute, but I don't think we want to overload that one, as that 
currently indicates whether or not the equipment has a controller, not
whether or not the control is working.

After detection and flagging, the optimization needs to be halted and 
re-triggered. The genetic algorithm will need to be modified so that it
only attempts to control equipment which is "operable"/"working."

## Inverters/General distributed generation (DG)
equipment.py needs to be extended so that it has a class for
inverters/DG. Mainly, we need to be able to keep our internal model up
to date with what's happening in the system with regards to P/Q output.

If done right, this should also be a good set up for future augmentation
of PyVVO - specifically, the genetic algorithm (GA) should be able to
manage DG set points, and later on we would want a solar forecast so
that the GA uses expected upcoming weather rather than current weather. 

## Communication Outage
This can probably be handled similarly to command tracking/managing 
stuck equipment, but the detection is different. Presently, the 
SimOutRouter's `_filter_output_by_mrid` method throws an error if there
is a missing measurement. Maybe instead of throwing an error it could 
warn in the log and then place `None` in the correct spot in the output.
Then, the corresponding EquipmentManager could interpret receiving
`None` as presently inoperable equipment.

## Flow Control for Genetic Algorithm (GA)
Some of the other pieces here are inevitably going to involve stopping
the GA. At present, the GA doesn't really have flow control. It'll need
some methods of gracefully stopping - e.g. stop GridLAB-D models, 
drop all related tables (maybe, unless we drop the table up front,
making dropping tables in cleanup unnecessary), etc.