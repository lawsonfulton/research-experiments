import autograd.numpy as numpy
import pygame
import sys

# Display parameters
width = height = 800
max_framerate = 60

# Color constants
background = (255, 255, 255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
white = (255,255,255)
black = (0,0,0)

# Initialization
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('', 30)
screen = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
frame_count = 0

def render(q, springs, save_frames=False, color=black):
    screen.fill(background)
    
    points = to_screen(numpy.reshape(q, (len(q)//2, 2)))

    for point in points:
        pygame.draw.circle(screen, red, point, 3)

    for spring in springs:
        pygame.draw.aaline(screen, black, points[spring[0]], points[spring[1]])

    update(save_frames)
    
def to_screen(world_points):
    """Takes points in [0, 1]x[0,1] and maps them to [0,width]x[0,height] with the y coordinate reversed for screen space"""
    screen_points = []
    for world_point in world_points:
        wx = world_point[0]
        wy = world_point[1]

        sx = int(wx * width)
        sy = int((1 - wy) * height)

        screen_points.append((sx, sy))

    return screen_points

def update(save_frames):
    global frame_count

    clock.tick(max_framerate)
    text_surf = font.render(str(int(clock.get_fps())), True, (0,0,0))
    screen.blit(text_surf,(0,0))

    pygame.display.update()

    if save_frames:
        frame_count += 1
        str_num = "000" + str(frame_count)
        file_name = "frame_capture/image" + str_num[-4:] + ".jpg"
        pygame.image.save(screen, file_name)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
