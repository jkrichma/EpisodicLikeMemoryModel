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
#    Python code to run the "Episodic Like Memory Scenario" in the
#    paper, "Episodic-Like Memory in a Simulation of Cuttlefish Behavior",
#    by S. Kandimalla, Q. Wong, K. Zheng and J. Krichmar, bioRxiv, 2025.
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

ENV_SIZE = 8
HOURS = 3
TS_PER_HOUR = 100
TIME_STEPS = 12000
DAYS = 100
NUM_CREATURES = 3
SHRIMP = 0
CRAB = 1


# getLocation - converts Cartesian coordinates into a location index
def getLocation(x, y, env_size):
    return y * env_size + x


# distance - returns the Euclidean distance between two Cartesian coordinates
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# roam - returns a new (x,y) location one random step away from the current (x,y)
def roam(x_curr, y_curr, env_size):
    while True:
        x_new = x_curr + np.random.randint(-1, 2)
        y_new = y_curr + np.random.randint(-1, 2)
        # check for environment boundary condition
        if 0 < x_new < env_size and 0 < y_new < env_size:
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


# getWhatWhenValue - given what and when indices, returns the maximum expected value across all where locations.
def getWhatWhenValue(epiMem, what, when):
    return np.max(epiMem[what, when, :])


# getWhatWhenLocationLocation - given what and when indices, returns the (x,y) coordinates of the maximum
#     expected value location.
def getWhatWhenLocation(epiMem, what, when):
    where = np.argmax(epiMem[what, when, :])
    env_size = int(np.sqrt(epiMem.shape[2]))
    y = (where // env_size)
    x = (where % env_size)
    return x, y


# Prey - class for the different prey (shrimp and crab).
class Prey:
    def __init__(self, prey_type, rwd, vis_begin, vis_end, env_size):
        self.type = prey_type
        self.env_size = env_size
        self.vis_begin = vis_begin
        self.rwd = rwd
        self.vis_end = vis_end
        self.env_size = env_size
        self.x, self.y = self.generate_position()

    # generate_position - sets the position of the shrimp and crab.
    def generate_position(self):
        if self.type == SHRIMP:
            x = ENV_SIZE - 1
            y = 0
        else:
            x = ENV_SIZE - 1
            y = ENV_SIZE - 1
        return x, y

    def respawn(self, spawn):
        if spawn:
            self.x, self.y = self.generate_position()
        return self.x, self.y


# ceph_epimem - runs the episodic memory model for one agent over the simulated DAYS
def ceph_epimem(lr, beta,
                shrimp_type, rwd_shrimp, shrimp_begin, shrimp_end,
                crab_type, rwd_crab, crab_begin, crab_end):
    HOME_X, HOME_Y = 0, ENV_SIZE / 2
    ROAM, HUNT_SHRIMP, HUNT_CRAB, = 0, 1, 2  # ESCAPE = 0, 1, 2, 3
    VAL_ROAM = 0.5
    CLOSE = 2
    SHRIMP_INDEX = 0
    CRAB_INDEX = 1
    cuttlebot_x = HOME_X
    cuttlebot_y = HOME_Y

    epiMem = np.zeros((NUM_CREATURES, HOURS, ENV_SIZE * ENV_SIZE))  # 3D episodic memory matrix
    shrimp_count = np.zeros((DAYS, HOURS))
    crab_count = np.zeros((DAYS, HOURS))

    # initialize shrimp and crab
    preys = [Prey(shrimp_type, rwd_shrimp, shrimp_begin, shrimp_end, ENV_SIZE),
             Prey(crab_type, rwd_crab, crab_begin, crab_end, ENV_SIZE)]

    current_time = 0
    current_day = 0
    act = ROAM
    ts = 0

    while current_day < DAYS:

        # if this is the start of a new hour
        #    increment the hour and day if it is the last hour
        #    get the expected values for hunting each prey type
        #    select an action (hunt shrimp, hunt crab, or roam).
        if ts % TS_PER_HOUR == 0:
            current_time = current_time + 1
            if current_time == HOURS:
                current_time = 0
                current_day = current_day + 1

            # in phase 1 (the first half of the simulation) the shrimp rewarded during all 3 hours
            # in phase 2 (the last half of the simulation) the shrimp is only rewarded during the 3rd hour
            if current_day > DAYS / 2 and current_time < 2:
                preys[SHRIMP_INDEX].rwd = 0
            else:
                preys[SHRIMP_INDEX].rwd = rwd_shrimp

            values = np.zeros(3)
            values[ROAM] = VAL_ROAM
            values[HUNT_SHRIMP] = getWhatWhenValue(epiMem, SHRIMP, current_time)
            values[HUNT_CRAB] = getWhatWhenValue(epiMem, CRAB, current_time)
            act = action_select(values, beta)
            if values[act] == 0:
                act = ROAM
            cuttlebot_x, cuttlebot_y = HOME_X, HOME_Y

        ######################
        #                    #
        # CUTTLEBOT BEHAVIOR #
        #                    #
        ######################
        if act == HUNT_SHRIMP:
            x_prey, y_prey = getWhatWhenLocation(epiMem, SHRIMP, current_time)
            if x_prey != 0 or y_prey != 0:
                cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, x_prey, y_prey)
            else:
                cuttlebot_x, cuttlebot_y = roam(cuttlebot_x, cuttlebot_y, ENV_SIZE)

        elif act == HUNT_CRAB:
            x_prey, y_prey = getWhatWhenLocation(epiMem, CRAB, current_time)
            if x_prey != 0 or y_prey != 0:
                cuttlebot_x, cuttlebot_y = move_to(cuttlebot_x, cuttlebot_y, x_prey, y_prey)
            else:
                cuttlebot_x, cuttlebot_y = roam(cuttlebot_x, cuttlebot_y, ENV_SIZE)
        else:
            cuttlebot_x, cuttlebot_y = roam(cuttlebot_x, cuttlebot_y, ENV_SIZE)

        # Randomize the prey list so there are no order effects.
        # The old "for prey in preys" prioritized hunting the first
        # prey in the list
        jump_to_next_hour = False
        for p in np.random.permutation(len(preys)):

            # if a prey type is visible at the current hour and the agent is close to that prey
            #    get the location of the prey and update the episodic memory.
            visible = (preys[p].vis_begin <= current_time <= preys[p].vis_end)
            if visible and distance(cuttlebot_x, cuttlebot_y, preys[p].x, preys[p].y) < CLOSE:
                loc = getLocation(preys[p].x, preys[p].y, ENV_SIZE)
                # update for the specific prey index and reward
                if preys[p].type == SHRIMP:
                    epiMem = assocMemoryLR(epiMem, lr, preys[p].rwd, SHRIMP, current_time, loc)
                    shrimp_count[current_day, current_time] += 1
                else:  # preys[p].type == CRAB:
                    epiMem = assocMemoryLR(epiMem, lr, preys[p].rwd, CRAB, current_time, loc)
                    crab_count[current_day, current_time] += 1
                cuttlebot_x, cuttlebot_y = HOME_X, HOME_Y
                jump_to_next_hour = True

        # if a prey is found, jump to the next hour
        if jump_to_next_hour:
            ts = TS_PER_HOUR
        else:
            ts += 1

    # return episodic memory and the number of shrimp and crab eaten
    return epiMem, shrimp_count, crab_count


# run_simulations - runs the cuttlefish simulation 100 times
def run_simulations(runs=100):

    # initialize the prey counts and episodic memory
    shrimp = np.zeros((runs, DAYS, HOURS))
    crab = np.zeros((runs, DAYS, HOURS))
    mem = np.zeros((runs, NUM_CREATURES, HOURS, ENV_SIZE * ENV_SIZE))

    LR = 0.1
    BETA = 1.0
    SHRIMP_RWD = 4
    CRAB_RWD = 1
    SHRIMP_BEGIN = 0
    SHRIMP_END = 2
    CRAB_BEGIN = 0
    CRAB_END = 2

    for i in range(runs):
        print(f"\nRun {i + 1}/{runs}")
        mem[i, :, :, :], shrimp[i, :, :], crab[i, :, :] = ceph_epimem(
            LR, BETA,
            SHRIMP, SHRIMP_RWD, SHRIMP_BEGIN, SHRIMP_END,
            CRAB, CRAB_RWD, CRAB_BEGIN, CRAB_END
        )

    # time stamp and save results as NPY files
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d%m%Y-%H%M")
    np.save('epiLike_shrimp-'+formatted_date, shrimp)
    np.save('epiLike_crab-'+formatted_date, crab)
    np.save('epiLike_episodic_memory-'+formatted_date, mem)

    return shrimp, crab


# plot_results - creates boxplots for the results.
#    Plots are shown on the screen and saved as png files
def plot_results(p1, p2):

    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d%m%Y-%H%M")

    begin_day = 40
    end_day = 50
    plt.figure(1)
    fig = plt.gcf()
    fig.set_size_inches(6, 5)

    p1_correct = np.zeros(p1.shape[0])
    p2_correct = np.zeros(p2.shape[0])  # Assuming p2 has the same first dimension as p1 for this initialization

    for i in range(p1.shape[0]):
        p1_correct[i] = np.sum(p1[i, begin_day:end_day, 0:3])  # MATLAB's 3rd dim is index 2 in Python
        p2_correct[i] = np.sum(p2[i, begin_day:end_day, 0:3])

    data_to_plot = np.array([p1_correct/(p1_correct+p2_correct)*100, p2_correct/(p1_correct+p2_correct)*100]).T
    bp = plt.boxplot(data_to_plot, widths=0.5, tick_labels=['shrimp', 'crabs'])
    plt.ylabel('Percentage of Choices', fontsize=14)  # Set font size directly in ylabel
    # Set y-axis limits
    plt.ylim(0, 110)
    plt.title('A. Phase 1 (last ' + str(end_day-begin_day) + ' choices)')
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_linewidth(1)
    plt.savefig('epiLike_fig1_phase1-'+formatted_date+'.png', dpi=300)

    begin_day = 90
    end_day = 100
    plt.figure(2)
    fig = plt.gcf()
    fig.set_size_inches(6, 5)

    p1_1hr = np.zeros(p1.shape[0])
    p2_1hr = np.zeros(p2.shape[0])
    p1_3hr = np.zeros(p1.shape[0])
    p2_3hr = np.zeros(p2.shape[0])
    for i in range(p1.shape[0]):
        p1_1hr[i] = np.sum(p1[i, begin_day:end_day, 0:2])
        p2_1hr[i] = np.sum(p2[i, begin_day:end_day, 0:2])
        p1_3hr[i] = np.sum(p1[i, begin_day:end_day, 2])
        p2_3hr[i] = np.sum(p2[i, begin_day:end_day, 2])

    data_to_plot = np.array([p1_1hr/(p1_1hr+p2_1hr)*100, p2_1hr/(p1_1hr+p2_1hr)*100, p1_3hr/(p1_3hr+p2_3hr)*100, p2_3hr/(p1_3hr+p2_3hr)*100]).T
    bp = plt.boxplot(data_to_plot, widths=0.75, tick_labels=['shrimp (1hr)', 'crabs(1hr)', 'shrimp (3hr)', 'crabs(3hr)'])

    plt.ylabel('Percentage of Choices', fontsize=14)  # Set font size directly in ylabel
    plt.ylim(0, 110)
    plt.title('B. Phase 2 (last ' + str(end_day-begin_day) + ' choices)')
    plt.xlim(0, 5)  # Keeping similar to MATLAB's axis setting, but note how boxplots are positioned.
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_linewidth(1)
    # plt.tight_layout()
    plt.savefig('epiLike_fig2_phase2-'+formatted_date+'.png', dpi=300)

    plt.show()

if __name__ == "__main__":
    shrimp, crab = run_simulations()
    plot_results(shrimp, crab)

    print("Simulation completed.")
