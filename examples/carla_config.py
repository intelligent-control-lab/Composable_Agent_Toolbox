import random
import time
import carla
from numpy import array

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()
client.load_world('Town07')

settings = world.get_settings()
# settings.synchronous_mode = True # Enables synchronous mode
# settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)

spawn_points = world.get_map().get_spawn_points()
for pt in spawn_points:
    print(pt)

blueprint_library = world.get_blueprint_library()
car_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
# car_bp = blueprint_library.find('vehicle.yamaha.yzf')
ped_bp = blueprint_library.find('walker.pedestrian.0017')

car_trans = []

# # FRAME 1
# car_trans.append(carla.Transform(carla.Location(x=-15, y=0, z=0.1), carla.Rotation(yaw=0))) # NPC
# car_trans.append(carla.Transform(carla.Location(x=-5, y=-10, z=0.1), carla.Rotation(yaw=90)))
# car_trans.append(carla.Transform(carla.Location(x=8, y=-3, z=0.1), carla.Rotation(yaw=180)))
# car_trans.append(carla.Transform(carla.Location(x=-2, y=8, z=0.1), carla.Rotation(yaw=270))) # NPC

# approx 18 across field

# # FRAME 2
# car_trans.append(carla.Transform(carla.Location(x=-15+2*1.5**2, y=0, z=0.1), carla.Rotation(yaw=0))) # NPC
# car_trans.append(carla.Transform(carla.Location(x=-6, y=-6, z=0.1), carla.Rotation(yaw=100)))
# car_trans.append(carla.Transform(carla.Location(x=8, y=-3, z=0.1), carla.Rotation(yaw=190)))
# car_trans.append(carla.Transform(carla.Location(x=-2, y=8-2*1.5**2, z=0.1), carla.Rotation(yaw=270))) # NPC

# FRAME 3
car_trans.append(carla.Transform(carla.Location(x=-15+2*2**2, y=0, z=0.1), carla.Rotation(yaw=0))) # NPC
car_trans.append(carla.Transform(carla.Location(x=-6, y=-3, z=0.1), carla.Rotation(yaw=95)))
# car_trans.append(carla.Transform(carla.Location(x=2, y=-10, z=0.1), carla.Rotation(yaw=200)))
car_trans.append(carla.Transform(carla.Location(x=-2, y=8-2*2**2, z=0.1), carla.Rotation(yaw=270))) # NPC

# # FRAME 4
# car_trans.append(carla.Transform(carla.Location(x=-15+2*3**2, y=0, z=0.1), carla.Rotation(yaw=0))) # NPC
# car_trans.append(carla.Transform(carla.Location(x=-5, y=4, z=0.1), carla.Rotation(yaw=80)))
# car_trans.append(carla.Transform(carla.Location(x=-10, y=-4, z=0.1), carla.Rotation(yaw=170)))
# car_trans.append(carla.Transform(carla.Location(x=-2, y=8-2*3**2, z=0.1), carla.Rotation(yaw=270))) # NPC

ped_trans = []
# ped_trans.append(carla.Transform(carla.Location(x=-57, y=11, z=0.01), carla.Rotation(yaw=0)))

cars = [world.spawn_actor(car_bp, t) for t in car_trans]
peds = [world.spawn_actor(ped_bp, t) for t in ped_trans]

if input("ENTER to destroy") == "":
    for car in cars:
        car.destroy()
    for ped in peds:
        ped.destroy()
    