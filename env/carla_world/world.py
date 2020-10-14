import random

import carla
import numpy as np
from progressbar import progressbar
import pygame

from env.base_world.world import World
from env.carla_world.hud import HUD
from env.carla_world.camera_manager import CameraManager


class CarlaWorld(World):
    def __init__(self, spec, render_window=True):
        super(CarlaWorld, self).__init__(spec)
        self.recording_enabled = False
        self.recording_start = 0

        # Carla Client setup
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)
        self.world = client.load_world(spec.get('world', 'Town01'))
        self.settings = self.world.get_settings()
        self._set_timestep(0.1)
        self.map = self.world.get_map()

        # Rendering setup
        self.clock = None
        self.camera_manager = None  # Camera tracking ego vehicle
        self.display = None
        self.hud = None
        self.display_resolution = (1280, 720)
        self._render_window = render_window
        if self._render_window:
            self._setup_display()

        # Ego vehicle setup
        self.ego = None
        self.ego_bp = self._create_vehicle_blueprint(spec.get('ego_vehicle_filter', 'vehicle.*'), color='49,8,8')

        self.reset()

    def _setup_display(self):
        """Setup pygame display for rendering."""
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(self.display_resolution, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(*self.display_resolution)
        self.clock = pygame.time.Clock()
        self.world.on_tick(self.hud.on_world_tick)

    def _set_timestep(self, timestep):
        """Set and clamp timestep if necessary."""
        self.settings.fixed_delta_seconds = timestep
        if self.settings.fixed_delta_seconds > 0.1:
            print('Carla should not use a timestep greater than 0.1. Clamping timestep')
            print('See https://github.com/carla-simulator/carla/issues/695 for more info')
            self.settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self.settings)

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=4):
        """Create the blueprint for a specific actor type.
        See https://carla.readthedocs.io/en/latest/bp_library/#vehicle

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
          color: vehicle color
          number_of_wheels: limit vehicles by number of wheels
        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        blueprint_library += [
            x for x in blueprints if int(x.get_attribute('number_of_wheels')) == number_of_wheels
        ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_actor_polygons(self, f):
        """Get the bounding box polygon of actors.
        Args:
          f: the filter indicating what type of actors we'll look at.
        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(f):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            length = bb.extent.x
            width = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]).transpose()
            # Get rotation matrix to transform to global coordinate
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(rotation, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at a specific transform.
        Args:
          transform: the Carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        vehicle_polygons = self._get_actor_polygons('vehicle.*')
        for poly in vehicle_polygons.items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def reset(self):
        """Reset the environment."""
        # TODO(piraka9011) Return sensor observations
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Spawn the ego vehicle
        if self.ego is not None:
            spawn_point = self.ego.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.ego = self.world.try_spawn_actor(self.ego_bp, spawn_point)
        while self.ego is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.ego = self.world.try_spawn_actor(self.ego_bp, spawn_point)
        # Set up the sensors.
        self.camera_manager = CameraManager(self.ego, self.hud, 2.2)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

    def set_render_window(self, render_window):
        self._render_window = render_window

    def render(self):
        self.camera_manager.render(self.display)
        pygame.display.flip()

    def _destroy_camera_manager(self):
        if self.camera_manager is not None:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

    def destroy(self):
        """Destroy all known actors and agents."""
        self._destroy_camera_manager()
        # TODO(piraka9011) Destroy all added agents
        actors = [self.ego]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def simulate(self, actions, dt=0.1):
        self._set_timestep(dt)
        acceleration = actions[0]
        steer = actions[1]
        # Convert acceleration to throttle and brake
        if acceleration > 0:
            throttle = np.clip(acceleration, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acceleration, 0, 1)

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)

        if self._render_window:
            self.render()


def main():
    carla_world_spec = {
        "world": "Town01",
        "dt": 0.1
    }
    carla_env = CarlaWorld(carla_world_spec)
    actions = [0.2, 0.0]
    for _ in progressbar(range(200)):
        carla_env.simulate(actions)


if __name__ == '__main__':
    main()
