import pygame
import sys

import numpy

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

def render(grid, save_frames=False, color=black):
    screen.fill(background)
    
    n_cells = len(grid)
    cell_size = width // n_cells
    for y_i, x_i in occupied_cells(grid):
        if grid[y_i][x_i] > 0:
            pygame.draw.rect(screen, color, (int(x_i/n_cells * width), int((1.0 - y_i/n_cells) * width - cell_size), cell_size, cell_size))

    update(save_frames)
    
def to_screen(world_points):
    """Takes points in [0, 1]x[0,1] and maps them to [0,width]x[0,height] with the y coordinate reversed for screen space"""
    screen_points = []
    for world_point in world_points:
        wx = world_point[0] * 0.05 + 0.5
        wy = world_point[1] * 0.05 + 0.5

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

####
grid_res = 200
growth_rate = 0.05

def occupied_cells(grid):
    return numpy.array(numpy.transpose(numpy.nonzero(grid)))



def indices_to_points(indices):
    return numpy.array([indices[:,1], indices[:,0]]).T / grid_res


def vector_field(p):
    return numpy.array([numpy.random.uniform(-1,1), (1.0 +  p[1])**2])

def growth(grid):
    occ_cells = occupied_cells(grid)
    n_occ = len(occ_cells)

    points = indices_to_points(occ_cells)
    
    # TODO combine magnitude with probability

    samples = numpy.random.random_sample(n_occ)
    growth_point_indices = numpy.where(samples < growth_rate)[0]

    if len(growth_point_indices) == 0:
        return

    growth_points = points[growth_point_indices]
    growth_cells = occ_cells[growth_point_indices]

    direction = numpy.apply_along_axis(vector_field, 1, growth_points)
    #angles = numpy.atan2(direction[:,1], direction[:,0])
    offsets = numpy.flip(numpy.round(direction), 1) ## TODO should normalize

    neighbors = (growth_cells + offsets).astype(int)
    xs = neighbors[:,1]
    ys = neighbors[:,0]
    condition = (xs >= 0) & (ys >=0) & (xs < grid_res) & (ys < grid_res)
    neighbors = neighbors[condition]

    grid[neighbors[:,0], neighbors[:,1]] = True


def main():
    grid = numpy.array([numpy.zeros(grid_res)] * grid_res, dtype=bool)
    
    grid[0] += True

    while True:
        growth(grid)
        render(grid)


if __name__ == "__main__":
    main()