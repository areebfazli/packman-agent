# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from contest.capture_agents import CaptureAgent
import distance_calculator
import random, time, util, sys
from contest.game import Directions
from contest.util import nearest_point
import game
import math

##############
# Team Arshe #
##############

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using first_index and second_index as their agent
    index numbers. is_red is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def get_successor(self, game_state, action):

        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):

        """
        Computes a linear combination of features and feature weights
        """

        features = self.evaluate_attack_parameters(game_state, action)
        weights = self.get_cost_of_attack_parameter(game_state, action)
        return features * weights

    def evaluate_attack_parameters(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_cost_of_attack_parameter(self, game_state, action):
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """Primary offensive agent specializing in food collection and enemy avoidance"""

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.present_coordinates = (-5, -5)
        self.counter = 0
        self.attack = False
        self.last_food = []
        self.present_food_list = []
        self.should_return = False
        self.capsule_power = False
        self.target_mode = None
        self.eaten_food = 0
        self.initial_target = []
        self.has_stopped = 0
        self.capsule_left = 0
        self.prev_capsule_left = 0
    
    # Setup of initial states
    def register_initial_state(self, game_state):
        self.current_food_size = 9999999
        CaptureAgent.register_initial_state(self, game_state)
        self.init_position = game_state.get_agent_state(self.index).get_position()
        self.initial_attack_coordinates(game_state)

    # Basic setup of the intial states
    def initial_attack_coordinates(self, game_state):
        """
        Determine initial attack path coordinates near the center line
        """
        layout_info = []
        x = (game_state.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        y = (game_state.data.layout.height - 2) // 2
        layout_info.extend((game_state.data.layout.width, game_state.data.layout.height, x, y))
        self.initial_target = []
        for i in range(1, layout_info[1] - 1):
            if not game_state.has_wall(layout_info[2], i):
                self.initial_target.append((layout_info[2], i))
        no_targets = len(self.initial_target)
        if no_targets % 2 == 0:
            no_targets = no_targets // 2
            self.initial_target = [self.initial_target[no_targets]]
        else:
            no_targets = (no_targets - 1) // 2
            self.initial_target = [self.initial_target[no_targets]]

    def evaluate_attack_parameters(self, game_state, action):
        """
        Calculate strategic features for action evaluation
        
        Features:
        - Offense status (pacman/ghost mode)
        - Distance to nearest food
        - Distance to nearest threatening ghost
        - Successor score potential
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        position = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = self.get_score(successor)

        if successor.get_agent_state(self.index).is_pacman:
            features['offence'] = 1
        else:
            features['offence'] = 0

        if food_list:
            features['food_distance'] = min([self.get_maze_distance(position, food) for food in food_list])

        dis_to_ghost = []
        opponents_list = self.get_opponents(successor)

        for i in range(len(opponents_list)):
            enemy_pos = opponents_list[i]
            enemy = successor.get_agent_state(enemy_pos)
            if not enemy.is_pacman and enemy.get_position() is not None:
                ghost_pos = enemy.get_position()
                dis_to_ghost.append(self.get_maze_distance(position, ghost_pos))

        if len(dis_to_ghost) > 0:
            min_dis_to_ghost = min(dis_to_ghost)
            if min_dis_to_ghost < 5:
                features['distance_to_ghost'] = min_dis_to_ghost + features['successor_score']
            else:
                features['distance_to_ghost'] = 0

        return features

    def get_cost_of_attack_parameter(self, game_state, action):
        '''
        Manual setup of the weights after various tweaks
        '''
        if self.attack:
            if self.should_return is True:
                return {'offence': 3010,
                        'successor_score': 202,
                        'food_distance': -8,
                        'distance_to_ghost': 215}
            else:
                return {'offence': 0,
                        'successor_score': 202,
                        'food_distance': -8,
                        'distance_to_ghost': 215}
        else:
            successor = self.get_successor(game_state, action)
            weight_ghost = 210
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            if len(invaders) > 0:
                if invaders[-1].scared_timer > 0:
                    weight_ghost = 0
            return {'offence': 0,
                    'successor_score': 202,
                    'food_distance': -8,
                    'distance_to_ghost': weight_ghost}

    def get_opponent_positions(self, game_state):
        return [game_state.get_agent_position(enemy) for enemy in self.get_opponents(game_state)]

    def best_possible_action(self, mcsc):
        actions = mcsc.get_legal_actions(self.index)
        actions.remove(Directions.STOP)

        if len(actions) == 1:
            return actions[0]
        else:
            reverse_dir = Directions.REVERSE[mcsc.get_agent_state(self.index).configuration.direction]
            if reverse_dir in actions:
                actions.remove(reverse_dir)
            return random.choice(actions)

    def monte_carlo_simulation(self, game_state, depth):
        """Simulate future state for decision evaluation"""

        sim = game_state.deep_copy()
        while depth > 0:
            sim = sim.generate_successor(self.index, self.best_possible_action(sim))
            depth -= 1
        return self.evaluate(sim, Directions.STOP)

    def get_best_action(self, legal_actions, game_state, possible_actions, distance_to_target):
        shortest_distance = 9999999999
        for i in range(len(legal_actions)):
            action = legal_actions[i]
            next_state = game_state.generate_successor(self.index, action)
            next_position = next_state.get_agent_position(self.index)
            distance = self.get_maze_distance(next_position, self.initial_target[0])
            distance_to_target.append(distance)
            if distance < shortest_distance:
                shortest_distance = distance

        best_actions_list = [a for a, distance in zip(legal_actions, distance_to_target) if distance == shortest_distance]
        best_action = random.choice(best_actions_list)
        return best_action

    def choose_action(self, game_state):
        self.present_coordinates = game_state.get_agent_state(self.index).get_position()

        if self.present_coordinates == self.init_position:
            self.has_stopped = 1
        if self.present_coordinates == self.initial_target[0]:
            self.has_stopped = 0

        # find next possible best move 
        if self.has_stopped == 1:
            legal_actions = game_state.get_legal_actions(self.index)
            legal_actions.remove(Directions.STOP)
            possible_actions = []
            distance_to_target = []
            
            best_action = self.get_best_action(legal_actions, game_state, possible_actions, distance_to_target)
            
            return best_action

        if self.has_stopped == 0:
            self.present_food_list = self.get_food(game_state).as_list()
            self.capsule_left = len(self.get_capsules(game_state))
            real_last_capsule_len = self.prev_capsule_left
            real_last_food_len = len(self.last_food)

            # Set returned = 1 when pacman has secured some food and should return back home           
            if len(self.present_food_list) < len(self.last_food):
                self.should_return = True
            self.last_food = self.present_food_list
            self.prev_capsule_left = self.capsule_left

            if not game_state.get_agent_state(self.index).is_pacman:
                self.should_return = False

            # checks the attack situation           
            remaining_food_list = self.get_food(game_state).as_list()
            remaining_food_size = len(remaining_food_list)
    
            if remaining_food_size == self.current_food_size:
                self.counter = self.counter + 1
            else:
                self.current_food_size = remaining_food_size
                self.counter = 0
            if game_state.get_initial_agent_position(self.index) == game_state.get_agent_state(self.index).get_position():
                self.counter = 0
            if self.counter > 20:
                self.attack = True
            else:
                self.attack = False
            
            actions_base = game_state.get_legal_actions(self.index)
            actions_base.remove(Directions.STOP)

            # distance to closest enemy        
            distance_to_enemy = 999999
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
            if len(invaders) > 0:
                distance_to_enemy = min([self.get_maze_distance(self.present_coordinates, a.get_position()) for a in invaders])
            
            '''
            Capsule eating:
            -> If there is capsule available then capsule_power is True.
            -> If enemy Distance is less than 5 then capsule_power is False.
            -> If pacman scored a food then return to home capsule_power is False.
            '''
            if self.capsule_left < real_last_capsule_len:
                self.capsule_power = True
                self.eaten_food = 0
            if distance_to_enemy <= 5:
                self.capsule_power = False
            if len(self.present_food_list) < len(self.last_food):
                self.capsule_power = False

            if self.capsule_power:
                if not game_state.get_agent_state(self.index).is_pacman:
                    self.eaten_food = 0

                mode_min_distance = 999999

                if len(self.present_food_list) < real_last_food_len:
                    self.eaten_food += 1

                if len(self.present_food_list) == 0 or self.eaten_food >= 5:
                    self.target_mode = self.init_position
                else:
                    for food in self.present_food_list:
                        distance = self.get_maze_distance(self.present_coordinates, food)
                        if distance < mode_min_distance:
                            mode_min_distance = distance
                            self.target_mode = food

                legal_actions = game_state.get_legal_actions(self.index)
                legal_actions.remove(Directions.STOP)
                possible_actions = []
                distance_to_target = []
                
                k = 0
                while k != len(legal_actions):
                    a = legal_actions[k]
                    newpos = (game_state.generate_successor(self.index, a)).get_agent_position(self.index)
                    possible_actions.append(a)
                    distance_to_target.append(self.get_maze_distance(newpos, self.target_mode))
                    k += 1
                
                min_dis = min(distance_to_target)
                best_actions = [a for a, dis in zip(possible_actions, distance_to_target) if dis == min_dis]
                best_action = random.choice(best_actions)
                return best_action
            else:
                self.eaten_food = 0
                distance_to_target = []
                for a in actions_base:
                    next_state = game_state.generate_successor(self.index, a)
                    value = 0
                    for i in range(1, 24):
                        value += self.monte_carlo_simulation(next_state, 20)
                    distance_to_target.append(value)

                best = max(distance_to_target)
                best_actions = [a for a, v in zip(actions_base, distance_to_target) if v == best]
                best_action = random.choice(best_actions)
            return best_action


class DefensiveReflexAgent(ReflexCaptureAgent):
    """Defensive specialist agent for territory protection and invader elimination"""

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.previous_food = []
        self.counter = 0

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.set_patrol_point(game_state)

    def set_patrol_point(self, game_state):
        '''
        Look for center of the maze for patrolling
        '''
        x = (game_state.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        self.patrol_points = []
        for i in range(1, game_state.data.layout.height - 1):
            if not game_state.has_wall(x, i):
                self.patrol_points.append((x, i))

        for i in range(len(self.patrol_points)):
            if len(self.patrol_points) > 2:
                self.patrol_points.remove(self.patrol_points[0])
                self.patrol_points.remove(self.patrol_points[-1])
            else:
                break

    def get_next_defensive_move(self, game_state):
        agent_actions = []
        actions = game_state.get_legal_actions(self.index)
        
        rev_dir = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        actions.remove(Directions.STOP)

        for i in range(0, len(actions) - 1):
            if rev_dir == actions[i]:
                actions.remove(rev_dir)

        for i in range(len(actions)):
            a = actions[i]
            new_state = game_state.generate_successor(self.index, a)
            if not new_state.get_agent_state(self.index).is_pacman:
                agent_actions.append(a)
        
        if len(agent_actions) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
        if self.counter > 4 or self.counter == 0:
            agent_actions.append(rev_dir)

        return agent_actions

    def choose_action(self, game_state):
        # Target priority: invaders > stolen food > patrol points

        position = game_state.get_agent_position(self.index)
        if position == self.target:
            self.target = None
        invaders = []
        nearest_invader = []
        min_distance = float("inf")

        # Look for enemy position in our home        
        opponents_positions = self.get_opponents(game_state)
        i = 0
        while i != len(opponents_positions):
            opponent_pos = opponents_positions[i]
            opponent = game_state.get_agent_state(opponent_pos)
            if opponent.is_pacman and opponent.get_position() is not None:
                opponent_pos = opponent.get_position()
                invaders.append(opponent_pos)
            i = i + 1

        # if enemy is found chase it and kill it
        if len(invaders) > 0:
            for opp_position in invaders:
                dist = self.get_maze_distance(opp_position, position)
                if dist < min_distance:
                    min_distance = dist
                    nearest_invader.append(opp_position)
            self.target = nearest_invader[-1]

        # if enemy has eaten some food, then remove it from targets
        else:
            if len(self.previous_food) > 0:
                if len(self.get_food_you_are_defending(game_state).as_list()) < len(self.previous_food):
                    yummy = set(self.previous_food) - set(self.get_food_you_are_defending(game_state).as_list())
                    self.target = yummy.pop()

        self.previous_food = self.get_food_you_are_defending(game_state).as_list()
        
        if self.target is None:
            if len(self.get_food_you_are_defending(game_state).as_list()) <= 4:
                high_priority_food = self.get_food_you_are_defending(game_state).as_list() + self.get_capsules_you_are_defending(game_state)
                self.target = random.choice(high_priority_food)
            else:
                self.target = random.choice(self.patrol_points)
        cand_act = self.get_next_defensive_move(game_state)
        awesome_moves = []
        fvalues = []

        i = 0
        # find the best move       
        while i < len(cand_act):
            a = cand_act[i]
            next_state = game_state.generate_successor(self.index, a)
            newpos = next_state.get_agent_position(self.index)
            awesome_moves.append(a)
            fvalues.append(self.get_maze_distance(newpos, self.target))
            i = i + 1

        best = min(fvalues)
        best_actions = [a for a, v in zip(awesome_moves, fvalues) if v == best]
        best_action = random.choice(best_actions)
        return best_action
