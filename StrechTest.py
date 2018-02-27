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

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

pygame.display.set_caption("Strech Test")

FPS = 60
currentTime = 0
lastFrameTime = 0


class Point:
    def __init__(self, x_val, y_val, mass):
        self.x_val = x_val
        self.y_val = y_val
        self.mass = mass

    def minus(self, other):
        new_x_val = self.x_val - other.x_val
        new_y_val = self.y_val - other.y_val
        return Point(new_x_val, new_y_val, self.mass)

    def to_vec(self):
        return numpy.matrix([self.x_val, self.y_val]).T


fixed_cluster_configuration = [Point(.5, 0, 10), Point(0, -.5, 10), Point(-.5, 0, 10), Point(0, .5, 10)]

current_cluster_configuration = [Point(.75, 0, 10), Point(0, -.75, 10), Point(-.75, 0, 10), Point(0, .75, 10)]


def get_center_of_mass(list_of_points):
    x_pos = 0
    y_pos = 0
    total_mass = 0

    for curr_point in list_of_points:
        x_pos = x_pos + curr_point.x_val * curr_point.mass
        y_pos = y_pos + curr_point.y_val * curr_point.mass
        total_mass = total_mass + curr_point.mass

    x_pos = x_pos / total_mass
    y_pos = y_pos / total_mass

    return numpy.matrix([x_pos, y_pos]).T


# init

def draw_loop():
    for curr_point in current_cluster_configuration:
        pygame.draw.circle(window,
                           (255, 255, 255),
                           (math.floor((curr_point.x_val * WINDOW_WIDTH / 2) + WINDOW_WIDTH / 2),
                            math.floor(WINDOW_HEIGHT - ((curr_point.y_val * WINDOW_HEIGHT / 2) + WINDOW_HEIGHT / 2))),
                           10,
                           0)
        # print("x = " + str(math.floor((curr_point.x_val * WINDOW_WIDTH/2) + WINDOW_WIDTH / 2)) + " y = " + str(math.floor(WINDOW_HEIGHT - ((curr_point.y_val * WINDOW_HEIGHT/2) + WINDOW_HEIGHT / 2))))
        # pygame.draw.circle(window, (255, 255, 255), (100, 100), 50, 0)


def compute_goal_pos(x0, x1):
    # qi = x0 - xcm
    q = []
    m = []
    for curr_point in x0:
        q.append(curr_point.to_vec())
        m.append(curr_point.mass)

    xcm0 = get_center_of_mass(x0)
    for curr_point in q:
        x_val = curr_point.item(0) - xcm0.item(0)
        y_val = curr_point.item(1) - xcm0.item(1)
        curr_point = numpy.matrix([x_val,y_val]).T

    # pi = x1- xcm1
    p = []
    for curr_point in x1:
        p.append(curr_point.to_vec())

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
    mat_r = mat_apq * scipy.linalg.inv(scipy.linalg.sqrtm(mat_apq.T * mat_apq))
    # gi = R*qi + xcm
    g = []
    for i in range(0,len(q)):
        g.append(mat_r*q[i] + xcm1)
    print(g)


def simulation_step(dt):
    compute_goal_pos(fixed_cluster_configuration,current_cluster_configuration)


x_pos = 0
running = True
while running:

    sleepTime = 1. / FPS - (currentTime - lastFrameTime)
    if sleepTime > 0:
        time.sleep(sleepTime)
    currentTime = time.time()
    dt = currentTime - lastFrameTime
    lastFrameTime = currentTime

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    simulation_step(dt)

    window.fill((0, 0, 0))

    draw_loop()
    # pygame.draw.circle(window, (255, 255, 255), (xpos, 100), 50, 0)

    x_pos = x_pos + 1

    pygame.display.flip()

pygame.quit()
