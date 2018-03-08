import os
import time
import random
import numpy
import sys
import math
import pygame
import numpy.linalg
import scipy.linalg

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800


pygame.init()
pygame.font.init()

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

pygame.display.set_caption("Strech Test")

FPS = 60
currentTime = 0
lastFrameTime = 0


class Point:
    def __init__(self, x_val, y_val, mass):
        self.last_x_pos = x_val
        self.last_y_pos = y_val
        self.x_pos = x_val
        self.y_pos = y_val
        self.mass = mass
        self.x_vel = 0
        self.y_vel = 0

    def minus(self, other):
        new_x_val = self.x_pos - other.x_val
        new_y_val = self.y_pos - other.y_val
        return Point(new_x_val, new_y_val, self.mass)

    def to_vec(self):
        return numpy.matrix([self.x_pos, self.y_pos]).T


fixed_cluster_configuration = [Point(.5, 0, 1), Point(0, -.5, 1), Point(-.5, 0, 1), Point(0, .5, 1), Point(0, 0, 1)]

current_cluster_configuration = [Point(1, 0, 1), Point(.5, -.1, 1), Point(-.75, 0, 1), Point(0, .75, 1), Point(0, 0, 1)]


def get_center_of_mass(list_of_points):
    x_pos = 0
    y_pos = 0
    total_mass = 0

    for curr_point in list_of_points:
        x_pos = x_pos + curr_point.x_pos * curr_point.mass
        y_pos = y_pos + curr_point.y_pos * curr_point.mass
        total_mass = total_mass + curr_point.mass

    x_pos = x_pos / total_mass
    y_pos = y_pos / total_mass

    return numpy.matrix([x_pos, y_pos]).T


# init

def draw_loop(dt):
    for curr_point in current_cluster_configuration:
        pygame.draw.circle(window,
                           (255, 255, 255),
                           (math.floor((curr_point.x_pos * WINDOW_WIDTH / 2) + WINDOW_WIDTH / 2),
                            math.floor(WINDOW_HEIGHT - ((curr_point.y_pos * WINDOW_HEIGHT / 2) + WINDOW_HEIGHT / 2))),
                           10,
                           0)
        # print("x = " + str(math.floor((curr_point.x_val * WINDOW_WIDTH/2) + WINDOW_WIDTH / 2)) + " y = " + str(math.floor(WINDOW_HEIGHT - ((curr_point.y_val * WINDOW_HEIGHT/2) + WINDOW_HEIGHT / 2))))
        # pygame.draw.circle(window, (255, 255, 255), (100, 100), 50, 0)
        #label = my_font.render(str(1 / dt), False, (0, 0, 0))
        #window.blit(label, (0, 0))
        pygame.display.set_caption("Stretch Test | FPS: " + str(math.floor(1/dt)))


def semi_implicit_euler_step(dt):
    for v in current_cluster_configuration:
        v.x_vel = v.x_vel  # ext force in x
        v.y_vel = v.y_vel - (dt * (9.81 / v.mass))  # gravity
        v.x_pos = v.x_pos + dt * v.x_vel
        v.y_pos = v.y_pos + dt * v.y_vel


def get_pos_vec(list_p):
    p = []
    for curr_point in list_p:
        p.append(curr_point.to_vec())
    return p


def get_mass_vec(list_p):
    m = []
    for curr_point in list_p:
        m.append(curr_point.mass)
    return m


def compute_goal_pos(x0, x1):
    # TODO need to refactor to make it shorter and more readable
    # qi = x0 - xcm
    q = get_pos_vec(x0)
    m = get_mass_vec(x0)

    xcm0 = get_center_of_mass(x0)

    for curr_point in q:
        x_val = curr_point.item(0) - xcm0.item(0)
        y_val = curr_point.item(1) - xcm0.item(1)
        curr_point = numpy.matrix([x_val, y_val]).T

    # pi = x1- xcm1
    p = get_pos_vec(x1)

    xcm1 = get_center_of_mass(x1)
    for curr_point in p:
        x_val = curr_point.item(0) - xcm1.item(0)
        y_val = curr_point.item(1) - xcm1.item(1)
        curr_point = numpy.matrix([x_val, y_val]).T

    # solve Sum of mi*(A*qi-pi)^2 for a so that Eq minimal
    # 1
    # A = (sum of mi*(pi*qi^t)) * (sum of mi*(pi*qi^t))^-1 = Apq * Aqq

    mat_apq = numpy.zeros((2, 2))
    for i in range(0, len(q)):
        mat_apq = mat_apq + (m[i] * p[i] * q[i].T)

    # Aqq symmetric -> no rotational part -> find rotational part of Apq
    # polar decomposition: Apq = R*S -> R = Apq* S^-1
    # R = Apq * sqrt(Apq^t*Apq)^-1
    #if numpy.linalg.matrix_rank(mat_apq.getH() * mat_apq != (mat_apq.getH() * mat_apq).shape):
    #    print(p)
    #    print(mat_apq)
    #    print(mat_apq.T * mat_apq)
    mat_r = mat_apq * scipy.linalg.inv(scipy.linalg.sqrtm(mat_apq.getH() * mat_apq))
    # gi = R*qi + xcm
    g = []
    for i in range(0, len(q)):
        g.append(mat_r * q[i] + xcm1)

    return g


def init_scene():
    testvar = 1


def project_shape_constraints(k, positions):
    goal_pos_vec = compute_goal_pos(fixed_cluster_configuration, current_cluster_configuration)
    curr_pos_vec = get_pos_vec(positions)
    constraint_gradient = []
    for iter in range(0, len(positions)):
        constraint_gradient.append(goal_pos_vec[iter] - curr_pos_vec[iter])

    i = 0
    for curr_pos in positions:
        # print(str(curr_pos.to_vec()) + "before shape" + str(i))
        # print(curr_pos.x_pos)
        # print((k))
        curr_pos.x_pos = curr_pos.x_pos + (k * constraint_gradient[i].item(0))
        # print(curr_pos.x_pos)
        # print(constraint_gradient[i].item(0))
        curr_pos.y_pos = curr_pos.y_pos + (k * constraint_gradient[i].item(1))
        # print(str(curr_pos.to_vec()) + "after shape" + str(i))
        i = i + 1


def project_collision_constraints(k, positions):
    # goal pos > 0 in y
    curr_pos_vec = get_pos_vec(positions)
    constraint_gradient = []
    for k in range(0, len(positions)):

        y_val = curr_pos_vec[k].item(1)
        # print(curr_pos_vec[k])
        if y_val < -1:
            constraint_gradient.append(numpy.matrix([0, -1 * (y_val + 1)]).T)
        else:
            constraint_gradient.append(numpy.matrix([0, 0]).T)

    for i in range(0, len(positions)):
        # print(str(positions[i].to_vec()) + "before" + str(i))
        positions[i].y_pos = positions[i].y_pos + (1 * constraint_gradient[i].item(1)) #assume k = 1
        # print(str(positions[i].to_vec()) + "after" + str(i))
        # print(str(positions[i].y_vel) + " vel y" + str(i))


def project_constraints(k, positions):
    project_shape_constraints(k, positions)

    project_collision_constraints(k, positions)


solverIterations = 10
# in [0,1]
stiffness = .01
# correct stiffness, so that it is linear to k (stiffness)
corrected_stiffness = 1 - ((1 - stiffness) ** solverIterations)
# print(str(corrected_stiffness) + "lul")


def simulation_step(dt):
    # compute_goal_pos(fixed_cluster_configuration,current_cluster_configuration)
    semi_implicit_euler_step(dt)
    # generate collision constraints
    for i in range(0, solverIterations):
        project_constraints(corrected_stiffness, current_cluster_configuration)
    for curr_point in current_cluster_configuration:
        curr_point.x_vel = (curr_point.x_pos - curr_point.last_x_pos) / dt
        curr_point.last_x_pos = curr_point.x_pos
        curr_point.y_vel = min((curr_point.y_pos - curr_point.last_y_pos) / dt, 0) # hack
        curr_point.last_y_pos = curr_point.y_pos
    # print(get_pos_vec(current_cluster_configuration))


currentTime = time.time()
lastFrameTime = time.time()
running = True
init_scene()

while running:

    sleepTime = 1. / FPS - (currentTime - lastFrameTime)
    if sleepTime > 0:
        time.sleep(sleepTime)
    currentTime = time.time()
    dt = (currentTime - lastFrameTime)
    lastFrameTime = currentTime

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    simulation_step(dt)

    window.fill((0, 0, 0))

    draw_loop(dt)
    # pygame.draw.circle(window, (255, 255, 255), (xpos, 100), 50, 0)

    pygame.display.flip()

pygame.quit()
