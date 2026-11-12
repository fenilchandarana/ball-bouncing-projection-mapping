import math
import random
import pygame
import pymunk

DISPLAY_INDEX = 1     # 1 = secondary screen/projector, 0 = main screen
BALL_RADIUS = 20
SPAWN_X = 0.8         # 0 = left edge, 0.5 = center, 1 = right edge
SPAWN_PER_SEC = 1    # balls created per second
MAX_BALLS = 200
GRAVITY = 900         # pixels/sec^2, positive = down on screen
BOUNCE = 0.7          # 0 = no bounce, 1 = perfect bounce
ROTATE_STEP = 5       # degrees per mouse wheel tick
MIN_SIZE = 6          # ignore tiny accidental boxes (pixels)

pygame.init()
displays = pygame.display.get_num_displays()
idx = DISPLAY_INDEX if DISPLAY_INDEX < displays else 0
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=idx)
W, H = screen.get_size()
pygame.mouse.set_visible(True)
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0, GRAVITY)

balls = []
boxes = []  # (shape, size, angle_deg) drawn with the mouse

drag_start = None
angle = 0.0  # degrees, changed with the mouse wheel


def spawn_ball():
    x = int(W * SPAWN_X)
    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, BALL_RADIUS))
    body.position = (x, -BALL_RADIUS)
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.elasticity = BOUNCE
    shape.friction = 0.4
    space.add(body, shape)
    balls.append(shape)


def add_box(center, size, angle_deg):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = center
    body.angle = math.radians(angle_deg)
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = BOUNCE
    shape.friction = 0.4
    space.add(body, shape)
    boxes.append(shape)


def corners(center, size, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    hw, hh = size[0] / 2, size[1] / 2
    pts = []
    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        pts.append((center[0] + x * c - y * s, center[1] + x * s + y * c))
    return pts


def current_drag():
    mx, my = pygame.mouse.get_pos()
    sx, sy = drag_start
    w, h = abs(mx - sx), abs(my - sy)
    center = ((mx + sx) / 2, (my + sy) / 2)
    return center, (w, h)


spawn_timer = 0.0
running = True
while running:
    dt = 1 / 60
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif e.key == pygame.K_c:  # clear all boxes
                for s in boxes:
                    space.remove(s, s.body)
                boxes.clear()
            elif e.key == pygame.K_z and boxes:  # undo last box
                s = boxes.pop()
                space.remove(s, s.body)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            drag_start = e.pos
            angle = 0.0
        elif e.type == pygame.MOUSEWHEEL:
            angle += e.y * ROTATE_STEP
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and drag_start:
            center, size = current_drag()
            if size[0] > MIN_SIZE and size[1] > MIN_SIZE:
                add_box(center, size, angle)
            drag_start = None

    spawn_timer += dt
    while spawn_timer > 1 / SPAWN_PER_SEC:
        spawn_timer -= 1 / SPAWN_PER_SEC
        if len(balls) < MAX_BALLS:
            spawn_ball()

    space.step(dt)

    for s in balls[:]:
        if s.body.position.y > H + 50:
            space.remove(s, s.body)
            balls.remove(s)

    screen.fill((0, 0, 0))
    for s in balls:
        x, y = s.body.position
        pygame.draw.circle(screen, (220, 30, 30), (int(x), int(y)), BALL_RADIUS)
    for s in boxes:
        pts = [s.body.local_to_world(v) for v in s.get_vertices()]
        pygame.draw.polygon(screen, (230, 210, 40), pts)

    # live preview while dragging
    if drag_start:
        center, size = current_drag()
        if size[0] > 1 and size[1] > 1:
            pygame.draw.polygon(screen, (0, 120, 255), corners(center, size, angle), 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()