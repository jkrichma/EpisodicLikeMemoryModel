###########################################################################
#
# Copyright(c) 2022
# Regents of the University of California. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. The names of its contributors may not be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###########################################################################
# Author: Jeff Krichmar, UC Irvine
# Email: jkrichma@uci.edu
# Date: September 2, 2025
# Description:
#    Python code to run the "Predator Prey Scenario" in the
#    paper, "Episodic-Like Memory in a Simulation of Cuttlefish Behavior",
#    by S. Kandimalla, Q. Wong, K. Zheng and J. Krichmar, bioRxiv, 2025.
###########################################################################

import random

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

ENV_SIZE = 12
DAYS = 200  # (int) (TIME_STEPS / (HOURS * TS_PER_HOUR))
RUNS = 100
HOURS = 6
REGIONS = 9
REGION_SIZE = 4
TS_PER_HOUR = 100
TIME_STEPS = 12000
NUM_CREATURES = 3
PREDATOR_VISION = 4
CUTTLEBOT_VISION = 2

SHRIMP = 0
SHRIMP_RWD = 4
SHRIMP_BEGIN = 2
SHRIMP_END = 5
CRAB = 1
CRAB_RWD = 1
CRAB_BEGIN = 0
CRAB_END = 5
PREDATOR = 2
PRED_PEN = 8
PRED_BEGIN = 1
PRED_END = 3

LR = 0.10
BETA = 1.0
NUM_WHAT_ACTIONS = 4
WHAT_ACTIONS = False  # if true, What query experiments, else Where query experiments
ROAM = 0
HUNT_SHRIMP = 1
HUNT_CRAB = 2
HIDE = 3


# getLocation - converts Cartesian coordinates into a location index
def getLocation(x, y, env_size):
    return y * env_size + x


# distance - returns the Euclidean distance between two Cartesian coordinates
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# roam - returns a new (x,y) location one random step away from the current (x,y)
def roam(x_curr, y_curr, env_size):
    done = False
    while not done:
        x_new = x_curr + np.random.randint(-1, 2)
        y_new = y_curr + np.random.randint(-1, 2)
        if 0 < x_new < env_size and 0 < y_new < env_size:
            done = True
    return x_new, y_new


# roam_in_region - returns a new (x,y) location one random step away from the current (x,y) while staying in the region
def roam_in_region(x_curr, y_curr, region):
    rx, ry = getRegionCoordinates(region)
    done = False
    # make sure coordinates stay within region
    while not done:
        x_new = x_curr + np.random.randint(-1, 2)
        y_new = y_curr + np.random.randint(-1, 2)
        if rx <= x_new < rx+REGION_SIZE and ry <= y_new < ry+REGION_SIZE:
            done = True
    return x_new, y_new


# move_to - moves from (from_x, from_y) one step closer to (to_x, to_y)
def move_to(from_x, from_y, to_x, to_y):
    x_new = from_x + (1 if from_x < to_x else -1 if from_x > to_x else 0)
    y_new = from_y + (1 if from_y < to_y else -1 if from_y > to_y else 0)
    return x_new, y_new


# action_select - applies the softmax function to the values array.
#     The beta temperature controls the exploration/exploitation tradeoff.
#     Returns the choice and probability distribution.
def action_select(values, beta):
    # check for overflow before calling exp function
    for i in range(len(values)):
        if values[i] * beta > 600:
            values[i] = 600 / beta
    exp_values = np.exp(beta * values)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(values), p=probabilities)


# assocMemoryLR - updates the episodic memory given the conjunction of what, when, and where indices.
#     Uses a variation of the delta learning rule.
def assocMemoryLR(epiMem, learning_rate, reward, what, when, where):
    epiMem[what, when, where] += learning_rate * (reward - epiMem[what, when, where])
    return epiMem

# getRegionCoordinates - returns the upper left corner (x,y) coordinates for a given region
def getRegionCoordinates(region):
    if region == 0:
        x = 0
        y = 0
    elif region == 1:
        x = 4
        y = 0
    elif region == 2:
        x = 8
        y = 0
    elif region == 3:
        x = 0
        y = 4
    elif region == 4:
        x = 4
        y = 4
    elif region == 5:
        x = 8
        y = 4
    elif region == 6:
        x = 0
        y = 8
    elif region == 7:
        x = 4
        y = 8
    else:  # region == 8:
        x = 8
        y = 8
    return x, y


# getRegion - returns the region the (x,y) coordinates reside in
def getRegion(x, y):
    if x < 4 and y < 4:
        region = 0
    elif x < 8 and y < 4:
        region = 1
    elif x < 12 and y < 4:
        region = 2
    elif x < 4 and y < 8:
        region = 3
    elif x < 8 and y < 8:
        region = 4
    elif x < 12 and y < 8:
        region = 5
    elif x < 4 and y < 12:
        region = 6
    elif x < 8 and y < 12:
        region = 7
    else:  # x < 12 and y < 12:
        region = 8

    return region


# getWhatWhenValue - given when and where indices, returns sum of the expected values for all "what" objects.
def getWhenWhereValue(epiMem, when, rx_from, rx_to, ry_from, ry_to):
    value = 0
    for x in range(rx_from, rx_to):
        for y in range(ry_from, ry_to):
            value = value + np.sum(epiMem[:, when, getLocation(x, y, ENV_SIZE)])
    return value


# getWhatWhenValue - given what and when indices, returns sum of the expected values for all the "where" regions.
def getWhatWhenValue(epiMem, what, when):
    return np.sum(epiMem[what, when, :])

# getWhatWhenLocationLocation - given what and when indices, returns the (x,y) coordinates of the maximum
#     expected value location.
def getWhatWhenLocation(epiMem, what, when):
    where = np.argmax(epiMem[what, when, :])
    env_size = int(np.sqrt(epiMem.shape[2]))
    y = (where // env_size)
    x = (where % env_size)
    return x, y


# PredatorPrey - class for the different prey (shrimp and crab) and the predator.
class PredatorPrey:
    def __init__(self, prey_type, vis_begin, vis_end, rwd, env_size):
        self.type = prey_type
        self.env_size = env_size
        self.vis_begin = vis_begin
        self.rwd = rwd
        self.vis_end = vis_end
        self.env_size = env_size
        self.x, self.y = self.generate_position()

    def generate_position(self):
        if self.type == SHRIMP:
            rx, ry = getRegionCoordinates(6)
            x = np.random.randint(rx, rx+REGION_SIZE)
            y = np.random.randint(ry, ry+REGION_SIZE)
        elif self.type == CRAB:
            rx, ry = getRegionCoordinates(8)
            x = np.random.randint(rx, rx+REGION_SIZE)
            y = np.random.randint(ry, ry+REGION_SIZE)
        else:  # Predator
            x = np.random.randint(0, self.env_size/2)
            y = np.random.randint(0, self.env_size)
        return x, y

    def respawn(self):
        self.x, self.y = self.generate_position()
        return self.x, self.y


# ceph_epimem - runs the episodic memory model for one agent over the simulated DAYS
def ceph_epimem(lr, beta,
                shrimp_type, rwd_shrimp, shrimp_begin, shrimp_end,
                crab_type, rwd_crab, crab_begin, crab_end,
                predator_type, pen_predator, predator_begin, predator_end):
    HOME_X, HOME_Y = ENV_SIZE / 2, 0
    VAL_ROAM = 0.5
    CLOSE = 2

    epiMem = np.zeros((NUM_CREATURES, HOURS, ENV_SIZE * ENV_SIZE))
    shrimp_count = np.zeros((DAYS, HOURS))
    crab_count = np.zeros((DAYS, HOURS))
    predator_count = np.zeros((DAYS, HOURS))
    actions = np.zeros((DAYS, HOURS))

    # initialize shrimp, crab, and predator
    preys = [PredatorPrey(shrimp_type, shrimp_begin, shrimp_end, rwd_shrimp, ENV_SIZE),
             PredatorPrey(crab_type, crab_begin, crab_end, rwd_crab, ENV_SIZE),
             PredatorPrey(predator_type, predator_begin, predator_end, pen_predator, ENV_SIZE)]

    current_time = 0
    current_day = 0
    ts = 0

    if WHAT_ACTIONS:
        act = ROAM
    else:
        act = np.random.randint(0, REGIONS)
        rx, ry = getRegionCoordinates(act)

    cuttlebot_x = HOME_X
    cuttlebot_y = HOME_Y

    while current_day < DAYS:

        actions[current_day, current_time] = act

        ######################
        #                    #
        # CUTTLEBOT BEHAVIOR #
        #                    #
        ######################
        camouflage = False
        if WHAT_ACTIONS:
            # What query experiments.  Actions are to hunt shrimp, hunt crab, or hide from predator
            if act == HUNT_SHRIMP:
                visible = (preys[SHRIMP].vis_begin <= current_time <= preys[SHRIMP].vis_end)
                x_prey, y_prey = getWhatWhenLocation(epiMem, SHRIMP, current_time)
                prey_region = getRegion(x_prey, y_prey)
                rx, ry = getRegionCoordinates(prey_region)

                # cuttlefish agent can see and approach shrimp
                if visible and distance(cuttlebot_x, cuttlebot_y, preys[SHRIMP].x, preys[SHRIMP].y) < CUTTLEBOT_VISION:
                    cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, preys[SHRIMP].x, preys[SHRIMP].y)
                elif x_prey != 0 or y_prey != 0:
                    # in region, roam within the region
                    if getRegion(cuttlebot_x, cuttlebot_y) == prey_region:
                        cuttlebot_x, cuttlebot_y = roam_in_region(cuttlebot_x, cuttlebot_y, prey_region)
                    # approach region
                    else:
                        cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, rx + (REGION_SIZE / 2), ry + (REGION_SIZE / 2))
                # no value for prey, roam
                else:
                    cuttlebot_x, cuttlebot_y = roam(cuttlebot_x, cuttlebot_y, ENV_SIZE)
            elif act == HUNT_CRAB:
                visible = (preys[CRAB].vis_begin <= current_time <= preys[CRAB].vis_end)
                x_prey, y_prey = getWhatWhenLocation(epiMem, CRAB, current_time)
                prey_region = getRegion(x_prey, y_prey)
                rx, ry = getRegionCoordinates(prey_region)
                # cuttlefish agent can see and approach crab
                if visible and distance(cuttlebot_x, cuttlebot_y, preys[CRAB].x, preys[CRAB].y) < CUTTLEBOT_VISION:
                    cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, preys[CRAB].x, preys[CRAB].y)
                elif x_prey != 0 or y_prey != 0:
                    # in region, roam within the region
                    if getRegion(cuttlebot_x, cuttlebot_y) == prey_region:
                        cuttlebot_x, cuttlebot_y = roam_in_region(cuttlebot_x, cuttlebot_y, prey_region)
                    # approach region
                    else:
                        cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, rx + (REGION_SIZE / 2), ry + (REGION_SIZE / 2))
                # no value for prey, roam
                else:
                    cuttlebot_x, cuttlebot_y = roam(cuttlebot_x, cuttlebot_y, ENV_SIZE)
            elif act == HIDE:
                camouflage = True
            else:  # act == ROAM:
                cuttlebot_x, cuttlebot_y = roam(cuttlebot_x, cuttlebot_y, ENV_SIZE)
        else:  # WHERE ACTIONS
            # check if either shrimp or crab are visible
            crab_visible = (preys[CRAB].vis_begin <= current_time <= preys[CRAB].vis_end)
            shrimp_visible = (preys[SHRIMP].vis_begin <= current_time <= preys[SHRIMP].vis_end)
            # if visible and within vision, approach prey
            if crab_visible and distance(cuttlebot_x, cuttlebot_y, preys[CRAB].x, preys[CRAB].y) < CUTTLEBOT_VISION:
                cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, preys[CRAB].x, preys[CRAB].y)
            elif shrimp_visible and distance(cuttlebot_x, cuttlebot_y, preys[SHRIMP].x, preys[SHRIMP].y) < CUTTLEBOT_VISION:
                cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, preys[SHRIMP].x, preys[SHRIMP].y)
            elif getRegion(cuttlebot_x, cuttlebot_y) == act:
                cuttlebot_x, cuttlebot_y = roam_in_region(cuttlebot_x, cuttlebot_y, act)
            else:
                cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, rx+(REGION_SIZE/2), ry+(REGION_SIZE/2))

        #####################
        #                   #
        # PREDATOR BEHAVIOR #
        #                   #
        #####################

        # If predator sees a cuttlefish, approach the cuttlefish. The predator cannot see the cuttlefish if camouflaging
        if preys[PREDATOR].vis_begin <= current_time <= preys[PREDATOR].vis_end:
            if not camouflage and distance(cuttlebot_x,cuttlebot_y, preys[PREDATOR].x, preys[PREDATOR].y) < PREDATOR_VISION:
                preys[PREDATOR].x, preys[PREDATOR].y = move_to(preys[PREDATOR].x, preys[PREDATOR].y, cuttlebot_x, cuttlebot_y)
            else:
                preys[PREDATOR].x, preys[PREDATOR].y = roam(preys[PREDATOR].x, preys[PREDATOR].y, ENV_SIZE)

        # Randomize the prey list so there are no order effects.
        # The old "for prey in preys" prioritized hunting the first
        # prey in the list
        jump_to_next_hour = False
        for p in np.random.permutation(len(preys)):
            visible = (preys[p].vis_begin <= current_time <= preys[p].vis_end)
            if visible and distance(cuttlebot_x, cuttlebot_y, preys[p].x, preys[p].y) < CLOSE:
                loc = getLocation(preys[p].x, preys[p].y, ENV_SIZE)
                if preys[p].type == PREDATOR and not camouflage:
                    if WHAT_ACTIONS:
                        reward = preys[p].rwd
                    else:
                        reward = preys[p].rwd * -1
                    epiMem = assocMemoryLR(epiMem, lr, reward, PREDATOR, current_time, loc)
                    predator_count[current_day, current_time] += 1
                elif preys[p].type == SHRIMP and distance(cuttlebot_x, cuttlebot_y, preys[p].x, preys[p].y) < CLOSE:
                    epiMem = assocMemoryLR(epiMem, lr, preys[p].rwd, SHRIMP, current_time, loc)
                    shrimp_count[current_day, current_time] += 1
                elif preys[p].type == CRAB and distance(cuttlebot_x, cuttlebot_y, preys[p].x, preys[p].y) < CLOSE:
                    epiMem = assocMemoryLR(epiMem, lr, preys[p].rwd, CRAB, current_time, loc)
                    crab_count[current_day, current_time] += 1
                cuttlebot_x, cuttlebot_y = HOME_X, HOME_Y
                jump_to_next_hour = True

        if jump_to_next_hour:
            ts = TS_PER_HOUR
        else:
            ts = ts + 1

        # if this is the start of a new hour
        #    increment the hour and day if it is the last hour
        if ts % TS_PER_HOUR == 0:
            current_time = current_time + 1
            if current_time == HOURS:
                current_time = 0
                current_day = current_day + 1

            #    get the expected values for hunting each prey type
            #    select an action (hunt shrimp, hunt crab, hide, or roam).
            if WHAT_ACTIONS:
                values = np.zeros(NUM_WHAT_ACTIONS)
                values[ROAM] = VAL_ROAM
                values[HIDE] = getWhatWhenValue(epiMem, PREDATOR, current_time)
                values[HUNT_SHRIMP] = getWhatWhenValue(epiMem, SHRIMP, current_time)
                values[HUNT_CRAB] = getWhatWhenValue(epiMem, CRAB, current_time)
                #    get the expected values for hunting each prey type
                #    select an action (region to search).
            else:  # WHERE_ACTIONS
                values = np.zeros(REGIONS)
                for i in range(REGIONS):
                    rx, ry = getRegionCoordinates(i)
                    values[i] = getWhenWhereValue(epiMem, current_time, rx, rx+REGION_SIZE, ry, ry+REGION_SIZE)

            act = action_select(values, beta)

            if WHAT_ACTIONS:
                if values[act] == 0:
                    act = ROAM
            else:
                rx, ry = getRegionCoordinates(act)
            cuttlebot_x, cuttlebot_y = HOME_X, HOME_Y
            preys[SHRIMP].respawn()
            preys[CRAB].respawn()
            preys[PREDATOR].respawn()

    return epiMem, shrimp_count, crab_count, predator_count, actions


# run_simulations - runs the cuttlefish simulation RUN times
def run_simulations(runs=RUNS):
    shrimps = np.zeros((runs, DAYS, HOURS))
    crabs = np.zeros((runs, DAYS, HOURS))
    predators = np.zeros((runs, DAYS, HOURS))
    actions = np.zeros((runs, DAYS, HOURS))
    epi_memory = np.zeros((runs, NUM_CREATURES, HOURS, ENV_SIZE * ENV_SIZE))

    for i in range(runs):
        print(f"\nRun {i + 1}/{runs}")
        epi_memory[i, :, :, :], shrimps[i, :, :], crabs[i, :, :], predators[i, :, :], actions[i, :, :] = ceph_epimem(
            LR, BETA,
            SHRIMP, SHRIMP_RWD, SHRIMP_BEGIN, SHRIMP_END,
            CRAB, CRAB_RWD, CRAB_BEGIN, CRAB_END,
            PREDATOR, PRED_PEN, PRED_BEGIN, PRED_END
        )

    # time stamp and save results as NPY files
    current_datetime = datetime.now()
    if WHAT_ACTIONS:
        act_str = 'what-'
    else:
        act_str = 'where-'
    formatted_date = current_datetime.strftime("%d%m%Y-%H%M")
    np.save('predprey_shrimps-'+act_str+formatted_date, shrimps)
    np.save('predprey_crabs-'+act_str+formatted_date, crabs)
    np.save('predprey_predators-'+act_str+formatted_date, predators)
    np.save('predprey_episodic_memory-'+act_str+formatted_date, epi_memory)
    np.save('predprey_actions-'+act_str+formatted_date, actions)

    return epi_memory, shrimps, crabs, predators, actions


# Plots for what query experiments
def plot_what_results(p1, p2, p3, p4):

    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d%m%Y-%H%M")

    p1_values = np.zeros((DAYS, RUNS))
    p2_values = np.zeros((DAYS, RUNS))
    p3_values = np.zeros((DAYS, RUNS))
    for i in range(DAYS):
        for j in range(RUNS):
            for k in range(HOURS):
                p1_values[i, j] = p1_values[i, j] + p1[j, i, k]
                p2_values[i, j] = p2_values[i, j] + p2[j, i, k]
                p3_values[i, j] = p3_values[i, j] + p3[j, i, k]
    plt.figure(1) #, figsize=(12, 8))
    x = np.arange(1, DAYS+1)
    offset = 0.15
    plt.errorbar(x, np.mean(p1_values, axis=(1)),
                 yerr=stats.sem(p1_values, axis=(1)),
                 fmt='g*-', label='shrimps', capsize=5)
    plt.errorbar(x, np.mean(p2_values, axis=(1)),
                 yerr=stats.sem(p2_values, axis=(1)),
                 fmt='b*-', label='crabs', capsize=5)
    plt.errorbar(x, np.mean(p3_values, axis=(1)),
                 yerr=stats.sem(p3_values, axis=(1)),
                 fmt='r*-', label='predators', capsize=5)
    plt.title("Cuttlebot Outcomes - What Query")
    plt.xlabel("Day")
    plt.ylabel("Number Eaten")
    plt.ylim(0, 4)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('fig1_what_epimem_predprey-'+formatted_date+'.png', dpi=300)

    day_range = 20

    roam_count = np.zeros((RUNS, HOURS))
    shrimp_count = np.zeros((RUNS, HOURS))
    crab_count = np.zeros((RUNS, HOURS))
    hide_count = np.zeros((RUNS, HOURS))
    for i in range(RUNS):
        for j in range(HOURS):
            for k in range(day_range):
                if p4[i, k, j] == 0:
                    roam_count[i, j] = roam_count[i, j] + 1
                elif p4[i, k, j] == 1:
                    shrimp_count[i, j] = shrimp_count[i, j] + 1
                elif p4[i, k, j] == 2:
                    crab_count[i, j] = crab_count[i, j] + 1
                else:
                    hide_count[i, j] = hide_count[i, j] + 1

    fig, axs = plt.subplots(4)
    fig.suptitle('A. Actions Per Hour (First ' + str(day_range) + ' Days)')
    plt.figure(2)
    axs[0].bar(['0', '1', '2', '3', '4', '5'], np.mean(roam_count, axis=0), yerr=np.std(roam_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[0].set_ylabel('Roam', fontsize=10)  # Set font size directly in ylabel
    axs[0].set_ylim(0, day_range+1)
    axs[1].bar(['0', '1', '2', '3', '4', '5'], np.mean(shrimp_count, axis=0), yerr=np.std(shrimp_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[1].set_ylabel('Hunt Shrimp', fontsize=10)  # Set font size directly in ylabel
    axs[1].set_ylim(0, day_range+1)
    axs[2].bar(['0', '1', '2', '3', '4', '5'], np.mean(crab_count, axis=0), yerr=np.std(crab_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[2].set_ylabel('Hunt Crab', fontsize=10)  # Set font size directly in ylabel
    axs[2].set_ylim(0, day_range+1)
    axs[3].bar(['0', '1', '2', '3', '4', '5'], np.mean(hide_count, axis=0), yerr=np.std(hide_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[3].set_ylabel('Hide', fontsize=10)  # Set font size directly in ylabel
    axs[3].set_ylim(0, day_range+1)
    axs[3].set_xlabel('Hour', fontsize=10)  # Set font size directly in ylabel
    plt.tight_layout()
    plt.savefig('fig2_what_epimem_predprey_'+formatted_date+'.png', dpi=300)

    roam_count = np.zeros((RUNS, HOURS))
    shrimp_count = np.zeros((RUNS, HOURS))
    crab_count = np.zeros((RUNS, HOURS))
    hide_count = np.zeros((RUNS, HOURS))
    day = DAYS-day_range
    for i in range(RUNS):
        for j in range(HOURS):
            for k in range(day_range):
                if p4[i, k+day, j] == 0:
                    roam_count[i, j] = roam_count[i, j] + 1
                elif p4[i, k+day, j] == 1:
                    shrimp_count[i, j] = shrimp_count[i, j] + 1
                elif p4[i, k+day, j] == 2:
                    crab_count[i, j] = crab_count[i, j] + 1
                else:
                    hide_count[i, j] = hide_count[i, j] + 1

    fig, axs = plt.subplots(4)
    fig.suptitle('B. Actions Per Hour (Last ' + str(day_range) + ' Days)')
    plt.figure(3)
    axs[3].set_xlabel('Hour', fontsize=10)  # Set font size directly in ylabel
    axs[0].bar(['0', '1', '2', '3', '4', '5'], np.mean(roam_count, axis=0), yerr=np.std(roam_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[0].set_ylabel('Roam', fontsize=10)  # Set font size directly in ylabel
    axs[0].set_ylim(0, day_range+1)
    axs[1].bar(['0', '1', '2', '3', '4', '5'], np.mean(shrimp_count, axis=0), yerr=np.std(shrimp_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[1].set_ylabel('Hunt Shrimp', fontsize=10)  # Set font size directly in ylabel
    axs[1].set_ylim(0, day_range+1)
    axs[2].bar(['0', '1', '2', '3', '4', '5'], np.mean(crab_count, axis=0), yerr=np.std(crab_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[2].set_ylabel('Hunt Crab', fontsize=10)  # Set font size directly in ylabel
    axs[2].set_ylim(0, day_range+1)
    axs[3].bar(['0', '1', '2', '3', '4', '5'], np.mean(hide_count, axis=0), yerr=np.std(hide_count, axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
    axs[3].set_ylabel('Hide', fontsize=10)  # Set font size directly in ylabel
    axs[3].set_ylim(0, day_range+1)
    axs[3].set_xlabel('Hour', fontsize=10)  # Set font size directly in ylabel
    plt.tight_layout()
    plt.savefig('fig3_what_epimem_predprey_'+formatted_date+'.png', dpi=300)

    plt.show()


# Plots for where query experiments
def plot_where_results(p1, p2, p3, p4):

    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d%m%Y-%H%M")

    p1_values = np.zeros((DAYS, RUNS))
    p2_values = np.zeros((DAYS, RUNS))
    p3_values = np.zeros((DAYS, RUNS))
    for i in range(DAYS):
        for j in range(RUNS):
            for k in range(HOURS):
                p1_values[i, j] = p1_values[i, j] + p1[j, i, k]
                p2_values[i, j] = p2_values[i, j] + p2[j, i, k]
                p3_values[i, j] = p3_values[i, j] + p3[j, i, k]
    plt.figure(1)
    x = np.arange(1, DAYS+1)
    offset = 0.15
    plt.errorbar(x, np.mean(p1_values, axis=(1)),
                 yerr=stats.sem(p1_values, axis=(1)),
                 fmt='g*-', label='shrimps', capsize=5)
    plt.errorbar(x, np.mean(p2_values, axis=(1)),
                 yerr=stats.sem(p2_values, axis=(1)),
                 fmt='b*-', label='crabs', capsize=5)
    plt.errorbar(x, np.mean(p3_values, axis=(1)),
                 yerr=stats.sem(p3_values, axis=(1)),
                 fmt='r*-', label='predators', capsize=5)
    plt.title("Cuttlebot Outcomes - Where Query")
    plt.xlabel("Day")
    plt.ylabel("Number Eaten")
    plt.ylim(0, 4)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('fig1_where_epimem_predprey-'+formatted_date+'.png', dpi=300)

    day_range = 20

    where_count = np.zeros((RUNS, HOURS, REGIONS))
    for i in range(RUNS):
        for j in range(HOURS):
            for k in range(day_range):
                a = int(p4[i, k, j])
                where_count[i, j, a] = where_count[i, j, a] + 1
    fig, axs = plt.subplots(REGIONS)
    fig.set_size_inches(5, 8.5)
    fig.suptitle('A. Where Actions Per Hour (First ' + str(day_range) + ' Days)')
    plt.figure(2)
    for i in range(REGIONS):
        # axs[i].boxplot(where_count[:, :, i])
        axs[i].bar(['0', '1', '2', '3', '4', '5'], np.mean(where_count[:, :, i], axis=0), yerr=np.std(where_count[:, :, i], axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
        axs[i].set_ylabel('Region ' + str(i), fontsize=10)  # Set font size directly in ylabel
        axs[i].set_ylim(0, day_range+1)
    axs[REGIONS-1].set_xlabel('Hour', fontsize=10)  # Set font size directly in ylabel
    plt.tight_layout()
    plt.savefig('fig2_where_epimem_predprey-'+formatted_date+'.png', dpi=300)

    day = DAYS-day_range
    where_count = np.zeros((RUNS, HOURS, REGIONS))
    for i in range(RUNS):
        for j in range(HOURS):
            for k in range(day_range):
                a = int(p4[i, k+day, j])
                where_count[i, j, a] = where_count[i, j, a] + 1
    fig, axs = plt.subplots(REGIONS)
    fig.set_size_inches(5, 8.5)
    fig.suptitle('B. Where Actions Per Hour (Last ' + str(day_range) + ' Days)')
    plt.figure(3)
    for i in range(REGIONS):
        # axs[i].boxplot(where_count[:, :, i])
        axs[i].bar(['0', '1', '2', '3', '4', '5'], np.mean(where_count[:, :, i], axis=0), yerr=np.std(where_count[:, :, i], axis=0))  # , capsize=5, ecolor='black', alpha=0.7)
        axs[i].set_ylabel('Region ' + str(i), fontsize=10)  # Set font size directly in ylabel
        axs[i].set_ylim(0, day_range+1)
    axs[REGIONS-1].set_xlabel('Hour', fontsize=10)  # Set font size directly in ylabel
    plt.tight_layout()
    plt.savefig('fig3_where_epimem_predprey-'+formatted_date+'.png', dpi=300)

    plt.show()

if __name__ == "__main__":
    mem, shrimp, crab, pred, actions = run_simulations()
    if WHAT_ACTIONS:
        plot_what_results(shrimp, crab, pred, actions)
    else:
        plot_where_results(shrimp, crab, pred, actions)

    print("Simulation completed.")
