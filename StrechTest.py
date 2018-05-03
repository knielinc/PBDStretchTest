import os
import time
import random
import numpy
import sys
import math
import pygame
import numpy.linalg
import scipy.linalg
from scipy.spatial import Delaunay

# TODO shape -> points, parameter // FLEX wo particle /materialien, selber scene machen, verschiebung/dehnung von einfachem würfel sei flex oder 2d

USE_XPBD = True
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800

gravity = 9.81

pygame.init()
pygame.font.init()

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF, 32)

pygame.display.set_caption("Strech Test")

FPS = 30
clock = pygame.time.Clock()
currentTime = 0
lastFrameTime = 0

# lamé parameters
lame_youngs_modulus = 10 ** 5
lame_poisson_ratio = 0.3  # apparently poisson ratio
lame_mu = lame_youngs_modulus / (2 * (1 + lame_poisson_ratio))
lame_lambda = lame_youngs_modulus * lame_poisson_ratio / ((1 + lame_poisson_ratio) * (1 - 2 * lame_poisson_ratio))  #

stiffness_matrix = numpy.matrix([[lame_lambda + 2 * lame_mu, lame_lambda, 0],
                                 [lame_lambda, lame_lambda + 2 * lame_mu, 0],
                                 [0, 0, 2 * lame_mu]])
compliance_matrix = stiffness_matrix.I

lagrange_multipliers = []


class Point:
    def __init__(self, x_val, y_val, invmass):
        self.last_x_pos = x_val
        self.last_y_pos = y_val
        self.x_pos = x_val
        self.y_pos = y_val
        self.invmass = invmass
        self.x_vel = 0
        self.y_vel = 0

    def minus(self, other):
        new_x_val = self.x_pos - other.x_val
        new_y_val = self.y_pos - other.y_val
        return Point(new_x_val, new_y_val, self.invmass)

    def to_vec(self):
        return numpy.matrix([self.x_pos, self.y_pos]).T


"""
fixed_cluster_configuration_0 = [Point(.5, 0, 1), Point(0, -.5, 1), Point(-.75, 0, 1), Point(0, .5, 1), Point(0, 0, 1),
                                 Point(.75, .75, 1), Point(-.5, .75, 1)]

current_cluster_configuration_0 = [Point(.5, 0, 1), Point(0, -.5, 1), Point(-.75, 0, 1), Point(0, .5, 1),
                                   Point(0, 0, 1), Point(.75, .75, 1), Point(-.5, .75, 1)]

clusters = [[0, 1, 2, 3, 4], [2, 3, 5, 6]]
"""

h = 0.1
meshgrid_array = numpy.linspace(0, 1, math.ceil(1 / h))
[x_vals, y_vals] = numpy.meshgrid(meshgrid_array, meshgrid_array)
x_vals_vec = numpy.reshape(x_vals.T, (100, 1))
y_vals_vec = numpy.reshape(y_vals.T, (100, 1))

sequence_of_tuples = numpy.block([x_vals_vec, y_vals_vec])
sequence_of_tuples = numpy.asmatrix(sequence_of_tuples)

triplets = Delaunay(sequence_of_tuples)

clusters = triplets.simplices.tolist()


def pointFromTuple(tuple):
    if (tuple[0] == 0 or tuple[0] == 1):
        return Point(tuple[0], tuple[1], 0)
    return Point(tuple[0], tuple[1], .1)


def deformedPointFromTuple(tuple):
    if (tuple[0] == 0):
        return Point(tuple[0] - 0, tuple[1], 0)
    if (tuple[0] == 1):
        return Point(tuple[0] + 0, tuple[1], 0)

    return Point(tuple[0], tuple[1], .1)


fixed_cluster_configuration_0 = list(map(pointFromTuple, sequence_of_tuples.tolist()))

current_cluster_configuration_0 = list(map(deformedPointFromTuple, sequence_of_tuples.tolist()))

'''

fixed_cluster_configuration_0 = [Point(-1, -.5, 1), Point(-1, .25, 1), Point(-.5, -.25, 1), Point(-.5, .25, 1),
                                 Point(0, -.25, 1), Point(0, .25, 1), Point(.5, -.5, 1), Point(.5, .25, 1)]

current_cluster_configuration_0 = [Point(-1, -.5, 1), Point(-1, .25, 1), Point(-.5, -.25, 1), Point(-.5, .25, 1),
                                   Point(0, -.25, 1), Point(0, .25, 1), Point(.5, -.5, 1), Point(.5, .25, 1)]

clusters = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]  # triangles

'''

'''
fixed_cluster_configuration_0 = [Point(-.5, -.5, 1), Point(0, .25, 1), Point(.5, -.5, 1)]

current_cluster_configuration_0 = [Point(-.5, -.5, 1), Point(0, 22, 1), Point(.5, -.5, 1)]

clusters = [[0, 1, 2]]  # triangles
'''
precalculated_Qis = []


def precalculate_Qis():
    for item in clusters:
        p0 = fixed_cluster_configuration_0[item[0]]
        p1 = fixed_cluster_configuration_0[item[1]]
        p2 = fixed_cluster_configuration_0[item[2]]
        q_inverse = numpy.matrix(
            [[p1.x_pos - p0.x_pos, p2.x_pos - p0.x_pos], [p1.y_pos - p0.y_pos, p2.y_pos - p0.y_pos]]).I
        precalculated_Qis.append(q_inverse)


def get_Ps(cluster_index):
    item = clusters[cluster_index]
    p0 = current_cluster_configuration_0[item[0]]
    p1 = current_cluster_configuration_0[item[1]]
    p2 = current_cluster_configuration_0[item[2]]
    p = numpy.matrix([[p1.x_pos - p0.x_pos, p2.x_pos - p0.x_pos], [p1.y_pos - p0.y_pos, p2.y_pos - p0.y_pos]])
    return p


'''
    startx, starty, endx, endy, density, connectiveness

    connectiveness: how many clusters in each axis are being built? connectiveness ^2 clusters in total
'''


def createClusterBox(startx, starty, endx, endy, weight, density, connectiveness):
    startx_ = min(startx, endx)
    endx_ = max(startx, endx)

    starty_ = min(starty, endy)
    endy_ = max(starty, endy)

    curr_pos = [startx_, starty_]
    fixed_cluster_configuration_0.clear()
    current_cluster_configuration_0.clear()
    # fixed_cluster_configuration_0.append(Point(curr_pos[0], curr_pos[1], weight))

    nrInXDir = 0
    nrInYDir = 0
    i = 0
    j = 0
    while (starty_ + (j) * density) <= endy_:

        while (startx_ + (i) * density) <= endx_:
            curr_pos[0] = startx_ + i * density
            fixed_cluster_configuration_0.append(Point(curr_pos[0], curr_pos[1], 1 / weight))
            current_cluster_configuration_0.append(Point(curr_pos[0], curr_pos[1], 1 / weight))
            i = i + 1

        j = j + 1
        curr_pos[1] = starty_ + j * density
        curr_pos[0] = startx_
        nrInXDir = i
        i = 0
    nrInYDir = j
    currCluster = []

    clusterSizeX = math.floor(nrInXDir / connectiveness) + 1
    clusterSizeY = math.floor(nrInYDir / connectiveness) + 1

    global clusters
    clusters = []

    for y in range(0, connectiveness):
        for x in range(0, connectiveness):
            for j_ in range(-1, clusterSizeY):
                for i_ in range(-1, clusterSizeX):
                    currIndex = (i_) + (j_) * nrInXDir + (x * clusterSizeX) + (y * clusterSizeY * nrInXDir)
                    if not (((x == 0 and i_ == -1) or (y == 0 and j_ == -1)) or (
                            (x == connectiveness - 1 and i_ == clusterSizeX - 1) or (
                            y == connectiveness - 1 and j_ == clusterSizeY - 1))):
                        if currIndex < (j_ + 1 + y * clusterSizeY) * nrInXDir:
                            if currIndex < nrInYDir * nrInXDir:
                                currCluster.append(currIndex)
            clusters.append(currCluster)
            currCluster = []


def get_center_of_mass(list_of_points):
    x_pos = 0
    y_pos = 0
    total_mass = 0

    for curr_point in list_of_points:
        x_pos = x_pos + curr_point.x_pos * 1. / curr_point.invmass
        y_pos = y_pos + curr_point.y_pos * 1. / curr_point.invmass
        total_mass = total_mass + 1. / curr_point.invmass

    x_pos = x_pos / total_mass
    y_pos = y_pos / total_mass

    return numpy.matrix([x_pos, y_pos]).T


# init

def draw_loop(dt):

    if USE_XPBD:
        for curr_triangle in clusters:
            pos1 = [math.floor(
                current_cluster_configuration_0[curr_triangle[0]].x_pos * WINDOW_WIDTH / 2) + WINDOW_WIDTH / 2,
                    math.floor(WINDOW_HEIGHT - ((current_cluster_configuration_0[
                                                     curr_triangle[0]].y_pos * WINDOW_HEIGHT / 2) + WINDOW_HEIGHT / 2))]
            pos2 = [math.floor(
                current_cluster_configuration_0[curr_triangle[1]].x_pos * WINDOW_WIDTH / 2) + WINDOW_WIDTH / 2,
                    math.floor(WINDOW_HEIGHT - ((current_cluster_configuration_0[
                                                     curr_triangle[1]].y_pos * WINDOW_HEIGHT / 2) + WINDOW_HEIGHT / 2))]
            pos3 = [math.floor(
                current_cluster_configuration_0[curr_triangle[2]].x_pos * WINDOW_WIDTH / 2) + WINDOW_WIDTH / 2,
                    math.floor(WINDOW_HEIGHT - ((current_cluster_configuration_0[
                                                     curr_triangle[2]].y_pos * WINDOW_HEIGHT / 2) + WINDOW_HEIGHT / 2))]

            pygame.draw.lines(window, (255, 255, 255), True, [pos1, pos2, pos3], 2)

    for curr_point in current_cluster_configuration_0:
        pygame.draw.circle(window,
                           (255, 100, 255),
                           (math.floor((curr_point.x_pos * WINDOW_WIDTH / 2) + WINDOW_WIDTH / 2),
                            math.floor(WINDOW_HEIGHT - ((curr_point.y_pos * WINDOW_HEIGHT / 2) + WINDOW_HEIGHT / 2))),
                           5,
                           0)
        # print("x = " + str(math.floor((curr_point.x_val * WINDOW_WIDTH/2) + WINDOW_WIDTH / 2)) + " y = " + str(math.floor(WINDOW_HEIGHT - ((curr_point.y_val * WINDOW_HEIGHT/2) + WINDOW_HEIGHT / 2))))
        # pygame.draw.circle(window, (255, 255, 255), (100, 100), 50, 0)
        # label = my_font.render(str(1 / dt), False, (0, 0, 0))
        # window.blit(label, (0, 0))
def semi_implicit_euler_step(dt):
    for v in current_cluster_configuration_0:
        v.x_vel = v.x_vel  # ext force in x
        v.y_vel = v.y_vel - (dt * (gravity * v.invmass))  # gravity
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
        m.append(curr_point.invmass)
    return m


def compute_goal_pos(x0, x1):
    # TODO need to refactor to make it shorter and more readable
    # qi = x0 - xcm
    q = get_pos_vec(x0)
    m = get_mass_vec(x0)

    xcm0 = get_center_of_mass(x0)

    for i in range(0, len(q)):
        x_val = q[i].item(0) - xcm0.item(0)
        y_val = q[i].item(1) - xcm0.item(1)
        q[i] = numpy.matrix([x_val, y_val]).T

    # pi = x1- xcm1
    p = get_pos_vec(x1)

    xcm1 = get_center_of_mass(x1)
    for i in range(0, len(p)):
        x_val = p[i].item(0) - xcm1.item(0)
        y_val = p[i].item(1) - xcm1.item(1)
        p[i] = numpy.matrix([x_val, y_val]).T

    # solve Sum of mi*(A*qi-pi)^2 for a so that Eq minimal
    # 1
    # A = (sum of mi*(pi*qi^t)) * (sum of mi*(pi*qi^t))^-1 = Apq * Aqq

    mat_apq = numpy.zeros((2, 2))
    for i in range(0, len(q)):
        mat_apq = mat_apq + (m[i] * p[i] * q[i].T)

    # Aqq symmetric -> no rotational part -> find rotational part of Apq
    # polar decomposition: Apq = R*S -> R = Apq* S^-1
    # R = Apq * sqrt(Apq^t*Apq)^-1
    # if numpy.linalg.matrix_rank(mat_apq.getH() * mat_apq != (mat_apq.getH() * mat_apq).shape):
    #    print(p)
    #    print(mat_apq)
    #    print(mat_apq.T * mat_apq)
    mat_test = mat_apq.getH() * mat_apq
    U, s, V = numpy.linalg.svd(mat_apq.getH() * mat_apq, full_matrices=True)
    s[s < 0] = 0
    S = numpy.diag(s)
    # mat_test_2 = numpy.dot(U, numpy.dot(S, V))
    mat_r = mat_apq * scipy.linalg.inv(numpy.dot(U, numpy.dot(numpy.diag(numpy.sqrt(s)), V)))

    # mat_r = mat_apq * scipy.linalg.inv(scipy.linalg.sqrtm(mat_apq.getH() * mat_apq))
    # gi = R*qi + xcm
    g = []
    for i in range(len(q)):
        current_g = (mat_r * q[i] + xcm1)
        current_g_conj = current_g.conj()
        if not numpy.array_equal(current_g, current_g_conj):
            print("CARE")
        g.append(mat_r * q[i] + xcm1)

    return g


def init_scene():
    createClusterBox(-.75, -.75, .75, .75, .75, .25, 2)


def project_shape_constraints(k, positions, cluster, curr_cluster_index):
    curr_positions = []
    fixed_positions = []

    for v in cluster:
        curr_positions.append(current_cluster_configuration_0[v])
        fixed_positions.append(fixed_cluster_configuration_0[v])

    goal_pos_vec = compute_goal_pos(fixed_positions, curr_positions)
    curr_pos_vec = get_pos_vec(curr_positions)
    constraint_gradient = []

    for iter in range(len(cluster)):
        constraint_gradient.append(goal_pos_vec[iter] - curr_pos_vec[iter])

    if USE_XPBD:
        constraint_gradient = numpy.asarray(constraint_gradient)

    i = 0
    # print("positions before shape constraint: " + str(get_pos_vec(positions)))

    if not USE_XPBD:
        for v in cluster:
            # print(str(curr_pos.to_vec()) + "before shape" + str(i))
            # print(curr_pos.x_pos)
            # print((k))
            positions[v].x_pos = positions[v].x_pos + (k * constraint_gradient[i].item(0))
            # print(curr_pos.x_pos)
            # print(constraint_gradient[i].item(0))
            positions[v].y_pos = positions[v].y_pos + (k * constraint_gradient[i].item(1))
            # print(str(curr_pos.to_vec()) + "after shape" + str(i))
            i = i + 1
    # print("positions after shape constraint: " + str(get_pos_vec(positions)))


def project_shape_constraints_new(k, positions, cluster, curr_cluster_index, dt):
    curr_positions = []
    fixed_positions = []

    for v in cluster:
        curr_positions.append(current_cluster_configuration_0[v])
        fixed_positions.append(fixed_cluster_configuration_0[v])

    inv_mass_matrix = numpy.asmatrix(numpy.zeros(shape=(len(cluster) * 2, len(cluster) * 2)))

    for i in range(len(curr_positions)):
        inv_mass_matrix[i * 2, i * 2] = curr_positions[i].invmass
        inv_mass_matrix[i * 2 + 1, i * 2 + 1] = curr_positions[i].invmass

    P = get_Ps(curr_cluster_index)
    Qi = precalculated_Qis[curr_cluster_index]
    F = P * Qi
    G = F.T * F - numpy.identity(2)
    constraint_vec = numpy.matrix([[G.item(0)], [G.item(3)], [G.item(1)]])

    f1 = F[:, 0]
    f2 = F[:, 1]

    c1_T = Qi[:, 0].T
    c2_T = Qi[:, 1].T

    # ∇Sij =[∇p1 ,∇p2]Sij =fj ciT +fi cjT

    delta_s_1_1_without_p0 = numpy.reshape((f1 * c1_T * 2).T, 4)
    delta_s_2_2_without_p0 = numpy.reshape((f2 * c2_T * 2).T, 4)
    delta_s_1_2_without_p0 = numpy.reshape((f2 * c1_T + f1 * c2_T).T, 4)

    constraint_gradient_without_p0 = numpy.block(
        [[delta_s_1_1_without_p0], [delta_s_2_2_without_p0], [delta_s_1_2_without_p0]])

    p0_constraint_gradient = - constraint_gradient_without_p0[:, 0:2] - constraint_gradient_without_p0[:, 2:4]

    constraint_gradient = numpy.block([p0_constraint_gradient, constraint_gradient_without_p0])

    '''
    constraint_gradient_2 = numpy.block([[constraint_gradient[0:1, :], constraint_gradient[1:2, :]],
                                       [constraint_gradient[2:3, :], constraint_gradient[3:4, :]],
                                       [constraint_gradient[4:5, :], constraint_gradient[5:6, :]]])
    '''

    alpha_tilde = compliance_matrix / (dt * dt)

    delta_lambda_nominator = ((-1 * constraint_vec) - (alpha_tilde * lagrange_multipliers[curr_cluster_index]))
    delta_lambda_denominator = (constraint_gradient * inv_mass_matrix * constraint_gradient.T + alpha_tilde)

    delta_lambda = delta_lambda_denominator.I * delta_lambda_nominator  # ORDER MATTERS!

    delta_x = (inv_mass_matrix * constraint_gradient.T * delta_lambda)

    lagrange_multipliers[curr_cluster_index] = lagrange_multipliers[curr_cluster_index] + delta_lambda
    i = 0
    for v in cluster:
        positions[v].x_pos = positions[v].x_pos + delta_x.item(0 + 2 * i)
        positions[v].y_pos = positions[v].y_pos + delta_x.item(1 + 2 * i)
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
        positions[i].y_pos = positions[i].y_pos + (1 * constraint_gradient[i].item(
            1))  # assume k = 1, but make it a bit less, so that it won't get stuck in an irreversible position
        # print(str(positions[i].to_vec()) + "after" + str(i))
        # print(str(positions[i].y_vel) + " vel y" + str(i))


def project_velocity_constraints(k, positions, cluster, dt):
    for v in cluster:
        positions[v].x_vel = positions[v].x_vel  # ext force in x
        positions[v].y_vel = k * (positions[v].y_vel - (dt * (9.81 * positions[v].invmass)))  # gravity
        positions[v].x_pos = positions[v].x_pos + dt * k * positions[v].x_vel
        positions[v].y_pos = positions[v].y_pos + dt * k * positions[v].y_vel


def project_constraints(k, positions, cluster_set, dt):
    for i in range(len(cluster_set)):
        if USE_XPBD:
            project_shape_constraints_new(k, positions, cluster_set[i], i, dt)

        if not USE_XPBD:
            project_shape_constraints(k, positions, cluster_set[i], i)

        project_collision_constraints(k, positions)
        # project_velocity_constraints(k, positions, cluster, dt)


solverIterations = 3
# in [0,1]
stiffness = .01
# correct stiffness, so that it is linear to k (stiffness)
corrected_stiffness = 1 - ((1 - stiffness) ** solverIterations)


# print(str(corrected_stiffness) + "lul")


def init_lagrange_multiplier(nr_of_constraints):
    for i in range(nr_of_constraints):
        lagrange_multipliers.append(numpy.matrix([[0], [0], [0]]))


def simulation_step(dt):
    # compute_goal_pos(fixed_cluster_configuration,current_cluster_configuration)
    semi_implicit_euler_step(dt)
    # generate collision constraints
    nr_of_constraints = len(clusters)
    init_lagrange_multiplier(nr_of_constraints)

    for i in range(0, solverIterations):
        project_constraints(corrected_stiffness, current_cluster_configuration_0, clusters, dt)

    for curr_point in current_cluster_configuration_0:
        curr_point.x_vel = (curr_point.x_pos - curr_point.last_x_pos) / dt
        curr_point.last_x_pos = curr_point.x_pos
        curr_point.y_vel = (curr_point.y_pos - curr_point.last_y_pos) / dt  # hack
        curr_point.last_y_pos = curr_point.y_pos
    # print(get_pos_vec(current_cluster_configuration))


frame_count = 0
frame_rate = 0
frame_rate_t0 = time.process_time()
frame_rate_t1 = time.process_time()
currentTime = time.process_time()
lastFrameTime = time.process_time()
running = True
# init_scene()
precalculate_Qis()

while running:
    # sleepTime = 1. / FPS - (currentTime - lastFrameTime)
    # if sleepTime > 0:
    #    time.sleep(sleepTime)
    clock.tick(FPS)

    currentTime = time.process_time()
    dt = 1 / FPS  # (currentTime - lastFrameTime)
    lastFrameTime = currentTime

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    simulation_step(dt)

    window.fill((0, 0, 0))

    draw_loop(dt)

    frame_count = frame_count + 1

    if frame_count % 30 == 0:
        frame_rate_t1 = time.process_time()
        frame_rate = 30 / (frame_rate_t1 - frame_rate_t0)
        pygame.display.set_caption("Stretch Test | FPS: " + str(frame_rate))
        frame_rate_t0 = frame_rate_t1

    # pygame.draw.circle(window, (255, 255, 255), (xpos, 100), 50, 0)

    pygame.display.flip()

pygame.quit()
