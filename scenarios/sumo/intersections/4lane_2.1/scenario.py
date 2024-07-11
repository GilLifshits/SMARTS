"""
This example demonstrates the creation of a multi-agent scenario for autonomous driving using SMARTS.
The scenario features four ego vehicles, including a bus and three passenger cars,
each with predefined routes and missions. The vehicles traverse an intersection with varying start points and paths,
providing a complex environment for testing multi-agent interactions.
The scenario is generated using the SMARTS SStudio tools, specifying map details, ego missions,
and simulation parameters to create a realistic and challenging simulation for autonomous driving research.
"""

from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.sstypes import (
    EndlessMission,
    Flow,
    JunctionEdgeIDResolver,
    MapSpec,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
)

ego1 = TrafficActor(
    name="car1",
    vehicle_type="bus",
)

ego2 = TrafficActor(
    name="car2",
    vehicle_type="passenger",
)

ego3 = TrafficActor(
    name="car2",
    vehicle_type="passenger",
)

ego4 = TrafficActor(
    name="car2",
    vehicle_type="passenger",
)

# Define routes for ego agents
route1 = Route(begin=("edge-south-SN", 0, 0), end=("edge-west-EW", 0, "max"))
route2 = Route(begin=("edge-west-WE", 0, 0), end=("edge-east-WE", 0, "max"))
route3 = Route(begin=("edge-south-SN", 0, 30), end=("edge-west-EW", 0, "max"))
route4 = Route(begin=("edge-north-NS", 0, 0), end=("edge-west-EW", 0, "max"))

# Define missions for ego agents
mission1 = Mission(route=route1)
mission2 = Mission(route=route2)
mission3 = Mission(route=route3)
mission4 = Mission(route=route4)

ego_missions = [mission1, mission2, mission3, mission4]


scenario = Scenario(
    map_spec=MapSpec(
        source=Path(__file__).parent.absolute(),
        shift_to_origin=True,
        lanepoint_spacing=1.0,
    ),
    ego_missions=ego_missions
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)
