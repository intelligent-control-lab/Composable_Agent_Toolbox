import sys
import os
import pygame as pg
import pygame.camera
from pygame.locals import *

class VideoWrapper:
    def __init__(self, env):
        self.env = env
        pygame.camera.init()
        cam = pygame.camera.Camera("/dev/video0", (640, 480))
        cam.start()

        file_num = 0
        done_capturing = False

    def start_capture(self):
        while not done_capturing:
            file_num = file_num + 1
            image = cam.get_image()
            screen.blit(image, (0,0))
            pygame.display.update()

            # Save every frame
            filename = "Snaps/%04d.png" % file_num
            pygame.image.save(image, filename)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done_capturing = True

    def create_video(self):
        # Combine frames to make video
        os.system("avconv -r 8 -f image2 -i Snaps/%04d.png -y -qscale 0 -s 640x480 -aspect 4:3 result.avi")