
import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

import sys
from contest.capture import run_game

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AdaptiveReflexAgent', second='AdaptiveReflexAgent', num_training=0):
    """
    Returns a list of two agents that will form the team.
    By default both agents are AdaptiveReflexAgent, which uses a mix of offensive
    and defensive strategies.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions.
    This is nearly identical to the provided version.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # When almost all food is captured, return to the start position.
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered; generate another successor.
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state.
        (Basic version -- overridden by subclasses)
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Returns a dictionary of feature weights.
        (Basic version -- overridden by subclasses)
        """
        return {'successor_score': 1.0}


class AdaptiveReflexAgent(ReflexCaptureAgent):
    """
    An adaptive agent that switches between offensive and defensive strategies.
    It gathers features for both food collection (offense) and enemy invaders (defense)
    as well as considers power capsules. Weights are adjusted when enemies are detected.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Offensive features: Focus on collecting food
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # Fewer remaining food is better
        if food_list:
            features['distance_to_food'] = min([self.get_maze_distance(my_pos, food)
                                                for food in food_list])
        else:
            features['distance_to_food'] = 0

        # Defensive features: Respond to enemy invaders (enemy Pacmen)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [enemy for enemy in enemies
                    if enemy.is_pacman and enemy.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, enemy.get_position())
                                                for enemy in invaders])
        else:
            features['invader_distance'] = 0

        # Power Capsule: Evaluate the distance to the nearest capsule
        capsules = self.get_capsules(successor)
        if capsules:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, cap)
                                                   for cap in capsules])
        else:
            features['distance_to_capsule'] = 0

        # Penalize stopping and reversing to encourage progress
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Returns weights that balance offensive and defensive behaviors.
        When enemy invaders are present, defense is prioritized by modifying
        the weights for invader features and de-emphasizing food collection.
        """
        weights = {
            'successor_score': 100,
            'distance_to_food': -1,
            'num_invaders': -1000,
            'invader_distance': -10,
            'distance_to_capsule': -5,
            'stop': -100,
            'reverse': -2
        }

        # Adjust weights if enemy invaders are detected:
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [enemy for enemy in enemies
                    if enemy.is_pacman and enemy.get_position() is not None]
        if invaders:
            # When under attack, increase the priority of closing the gap with invaders
            weights['invader_distance'] = -20
            # Decrease the emphasis on food collection to focus on defense
            weights['distance_to_food'] = 0

        return weights


if __name__ == '__main__':
    
    run_game(sys.argv)
