import os
import pygame
from pygame.locals import *
from sys import exit

current_dir=os.path.split(os.path.realpath(__file__))[0]
#print(current_dir)
plane_file=current_dir+"/pic/plane.png"
wind_file=current_dir+"/pic/up.png"
target_file=current_dir+"/pic/target.png"
background_file=current_dir+"/pic/background.png"
def load_plane():
    plane=pygame.image.load(plane_file).convert_alpha()
    return plane
def load_wind():
    wind=pygame.image.load(wind_file).convert_alpha()
    return wind
def load_target():
    target=pygame.image.load(target_file).convert_alpha()
    return target
def load_background():
    background=pygame.image.load(background_file).convert()
    return background

pass
