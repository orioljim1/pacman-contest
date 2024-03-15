# myTeam.py
# ---------
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



from captureAgents import CaptureAgent
from game import Actions
from game import Directions
from util import PriorityQueue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='Terranator', second='Terranator', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


class Terranator(CaptureAgent):
    
    def register_initial_state(self, game_state):

        CaptureAgent.register_initial_state(self, game_state)

        self.start = game_state.get_agent_position(self.index)

        self.spawn = self.get_spawn_position(game_state)

        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width

        self.midWidth = game_state.data.layout.width / 2
        self.midHeight = game_state.data.layout.height / 2

        self.defending_position = self.find_most_centric_tuple(self.get_boundary_positions(game_state))
        self.defending_capsules = self.get_capsules_you_are_defending(game_state)

        self.enemy_last_position = None

        self.strategy = "defender"

    def find_most_centric_tuple(self,tuples):
        second_elements = [t[1] for t in tuples]
        second_elements.sort()
        n = len(second_elements)
        mid = n // 2
        if n % 2 == 0:
            centric_value = second_elements[mid - 1]
        else:
            centric_value = second_elements[mid]
        
        for t in tuples:
            if t[1] == centric_value:
                return [t]


    def null_heuristic(self,state, constraint=None):
        return 0

    def get_boundaries(self):
        if self.red:
            x_boundary = self.midWidth - 1
        else:
            x_boundary = self.midWidth + 1
        return [(x_boundary,y_boundary) for y_boundary in  range(self.height)]

    def get_boundary_positions(self,game_state): #Retrieve list of positions of boundary
        boundaries = self.get_boundaries()
        possible_pos = []
        for bound in boundaries:
            x = int(bound[0])
            y = int(bound[1])
            if not game_state.has_wall(x,y):
                possible_pos.append(bound)
        return possible_pos

    def distance_home(self, game_state):  #Compute distance to nearest boundary
        
        this_state = game_state.get_agent_state(self.index)
        this_pos = this_state.get_position()
        
        possible_pos = self.get_boundary_positions(game_state)
        distance = 100000
        for pos in possible_pos:
            dist =  self.get_maze_distance(pos,this_pos)
            if dist < distance:
                distance = dist
        return distance

    def distance_capsule(self,game_state):  #Compute distance to capsule

        capsules = self.get_capsules(game_state)
        if len(capsules) > 1:
            distance = 100000
            for capsule in capsules:
                dist = self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), capsule)
                if dist < distance:
                    distance = dist
                    self.debugDraw(capsule, [125, 125, 211], True)
            return distance

        elif len(capsules) == 1 :
            distance = self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), capsules[0])
            self.debugDraw(capsules[0], [125, 125, 211], True)
            return distance


    def locationOfLastEatenFood(self,game_state): #Get location of the last eaten food
        if len(self.observationHistory) > 1:
            prev_state = self.get_previous_observation()
            prev_food_positions = set(self.get_food_you_are_defending(prev_state).as_list())
            current_food_positions = set(self.get_food_you_are_defending(game_state).as_list())
            eaten_foods = prev_food_positions - current_food_positions
            if len(eaten_foods) == 1:
                self.lastEatenFoodPosition = eaten_foods.pop() 


    def get_nearest_ghost_distance(self, game_state, need_state): #Compute distance to the nearest ghost
        this_position =  game_state.get_agent_state(self.index).get_position()
        oponents =  [game_state.get_agent_state(oponent) for oponent in self.get_opponents(game_state)]
        ghosts = [agent for agent in oponents if not agent.is_pacman and agent.get_position() != None]
        
        if ghosts:
            distances = [self.get_maze_distance(this_position, ghost.get_position()) for ghost in ghosts]
            min_dist =  min(distances)
            if need_state:
                return min_dist, ghosts[distances.index(min_dist)]
            return min_dist
        else:
            return None


    def opponentscaredTime(self,game_state):
        scared_timers = (game_state.get_agent_state(opponent).scared_timer
                     for opponent in self.get_opponents(game_state)
                     if game_state.get_agent_state(opponent).scared_timer > 1)
        return next(scared_timers, 0)


    def a_search(self, problem, game_state, heuristic=null_heuristic):
        start_state = problem.init_state()
        start_node = (start_state, [], 0)
        queue = PriorityQueue()
        queue.push(start_node, 0 + heuristic(start_state, game_state))
        visited = set()

        while not queue.isEmpty():
            current_node = queue.pop()
            state, path, cost_to_current = current_node

            if state in visited:
                continue
            visited.add(state)

            if problem.is_terminal(state):
                return path

            for successor in problem.get_successors(state):
                successor_state, action, cost_to_successor = successor
                if successor_state not in visited:
                    new_path = path + [action]
                    new_cost = cost_to_current + cost_to_successor
                    f_score = new_cost + heuristic(successor_state, game_state)
                    queue.push((successor_state, new_path, new_cost), f_score)
        return []


    def general_heuristic(self, state, game_state): #Heuristic to avoid ghosts
        # Initialize heuristic value
        heuristic_value = 0

        # Check if there is any ghost nearby
        nearest_ghost_distance = self.get_nearest_ghost_distance(game_state, False)
        if nearest_ghost_distance is not None:
            # Gather information about potential ghost threats
            opponent_states = [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]
            threatening_ghosts = [opponent for opponent in opponent_states if not opponent.is_pacman and opponent.scared_timer < 2 and opponent.get_position() is not None]

            # If there are any threatening ghosts, calculate the heuristic based on distance
            if threatening_ghosts:
                ghost_positions = [ghost.get_position() for ghost in threatening_ghosts]
                distances_to_ghosts = [self.get_maze_distance(state, ghost_pos) for ghost_pos in ghost_positions]
                min_distance_to_ghost = min(distances_to_ghosts)

                # Apply a scaling penalty for being too close to a ghost
                if min_distance_to_ghost < 2:
                    heuristic_value = pow((5 - min_distance_to_ghost), 5)

        return heuristic_value

    def get_spawn_position(self, game_state):
        #returns the base point that will be used as base point for the team
        base_team_idx =  self.get_team(game_state)[0]
        i_state = game_state.get_agent_state(base_team_idx)
        return i_state.get_position()


    def get_strategy(self, game_state):
        #this will return what strategy to use if attacker or defending 

        #evaluate self positon

        def key_with_max_value(d, base_index):
            # to get the further pacman form the spawn point
            if len(set(d.values())) == 1:
                return base_index
            else:
                return max(d, key=d.get)

        team_indices =  self.get_team(game_state)
        distances = {}
        for i in team_indices:
            i_state = game_state.get_agent_state(i)
            d = self.distancer.getDistance(i_state.get_position(), self.spawn)
            distances[i]= d

        if self.index == key_with_max_value(distances, team_indices[0]):
            return "attacker"
        else:
            return "defender"
        
    def process_move(self, constraint, game_state, turn):
        found_moves = self.a_search(constraint, game_state, self.general_heuristic)
        if len(found_moves)>0:
            return found_moves[turn]
        else:
            return "Stop"


    def choose_action(self, game_state):
        
        self.strategy = self.get_strategy(game_state)
        turn = 0#self.index % 2        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman]
        knowninvaders = [a for a in enemies if a.is_pacman and a.get_position() !=None ]

        if game_state.get_agent_state(self.index).scared_timer > 10:
            #if enemies got the capsule attack as there's no point in standing still
            self.strategy = "attacker" 

        if self.strategy == "attacker":
            if game_state.get_agent_state(self.index).num_carrying == 0 and len(self.get_food(game_state).as_list()) == 0:
                return 'Stop'
            
            if  len(self.get_capsules(game_state)) != 0 and self.opponentscaredTime(game_state) < 10:
                constraint = UnifiedSearchProblem(game_state, self, self.index, "path_capsule")
                return self.process_move(constraint, game_state, turn)
           
            
            if game_state.get_agent_state(self.index).num_carrying < 1:
                constraint = UnifiedSearchProblem(game_state, self, self.index,"path_food", 'default')
                return self.process_move(constraint, game_state, turn)
            
            if self.get_nearest_ghost_distance(game_state,True) != None and self.get_nearest_ghost_distance(game_state,True)[0]< 6 and self.get_nearest_ghost_distance(game_state,True)[1].scared_timer < 5:
                constraint = UnifiedSearchProblem(game_state, self, self.index, "run")
                if len(self.a_search(constraint, self.general_heuristic)) == 0:
                    return 'Stop'
                else:
                    return self.process_move(constraint, game_state, turn)
                    
            if len(self.get_food(game_state).as_list()) < 3 or game_state.data.timeleft < self.distance_home(game_state) + 60 or game_state.get_agent_state(self.index).num_carrying > 15:
                constraint = UnifiedSearchProblem(game_state, self, self.index, "to_base")
                if len(self.a_search(constraint, self.general_heuristic)) == 0:
                    return 'Stop'
                else:
                    return self.process_move(constraint, game_state, turn)

            constraint = UnifiedSearchProblem(game_state, self, self.index,"path_food" ,'default')
            return self.process_move(constraint, game_state, turn)
        else: #defender
            
            self.locationOfLastEatenFood(game_state)  # detect last eaten food
            if len(invaders) == 0 or game_state.get_agent_position(self.index) == self.enemy_last_position or len(knowninvaders) > 0:
                self.enemy_last_position = None

            if len(knowninvaders) == 0 and  self.enemy_last_position!=None and game_state.get_agent_state(self.index).scared_timer == 0:
                constraint = UnifiedSearchProblem(game_state,self,self.index, "detect_eaten")
                return self.process_move(constraint, game_state, turn)
            
            # chase the invader only the distance is Known and ghost not scared
            if len(knowninvaders) > 0 and game_state.get_agent_state(self.index).scared_timer == 0:
                constraint =  UnifiedSearchProblem(game_state,self,self.index, "home_enemies")
                return self.process_move(constraint, game_state, turn)
            
            #default action
            constraint = UnifiedSearchProblem(game_state, self, self.index, "path_team_middle")
            defending_capsule = self.a_search(constraint, game_state, self.general_heuristic)
            if len(defending_capsule)> 0:
                return defending_capsule[turn]
            else:
                return "Stop"

class UnifiedSearchProblem:
    #this class handles the multiple behaviours that we're computing depending on the satte of the game
    def __init__(self, game_state, agent, a_I=0, constraint_type='food', type=''):
        
        self.agent = agent
        self.a_I= a_I
        self.constraint_type = constraint_type
        self.type = type
        self.game_state = game_state
        self.initialize_problem()
        self.reset_tracking()

    def reset_tracking(self):
       
        self.checked = {}  
        self.checked_list = [] 
        self.expanded_states = 0 

    def initialize_problem(self):
       
        self.starting_place = self.game_state.get_agent_state(self.a_I).get_position()
        self.cost = lambda x: 1
        self.capsule = self.agent.get_capsules(self.game_state)
        self.walls = self.game_state.get_walls()

        self.food = self.agent.get_food(self.game_state)
        if self.constraint_type == 'path_food':
            self.foods = self.food.as_list()
            self.terminal_states = self.foods
        elif self.constraint_type == 'run':
            self.defending_zone = self.agent.get_boundary_positions(self.game_state)
            self.terminal_states = self.defending_zone + self.capsule
        elif self.constraint_type == 'path_team_middle':
            self.terminal_states = self.agent.defending_position
        elif self.constraint_type == 'to_base':
            self.defending_zone = self.agent.get_boundary_positions(self.game_state)
            self.terminal_states = self.defending_zone
        elif self.constraint_type == 'detect_eaten':
            self.enemy_last_position = self.agent.enemy_last_position
            self.terminal_states = [self.enemy_last_position]
        elif self.constraint_type == 'path_capsule':
            self.terminal_states = self.capsule
        elif self.constraint_type == 'home_enemies':
            enemies = [self.game_state.get_agent_state(i) for i in self.agent.get_opponents(self.game_state)]
            self.invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            self.terminal_states = [invader.get_position() for invader in self.invaders] if self.invaders else []

    def init_state(self):
        return self.starting_place

    def is_terminal(self, state):
        if self.constraint_type in ['path_food', 'path_capsule', 'path_team_middle', 'run', 'to_base', 'detect_eaten', 'home_enemies']:
            return state in self.terminal_states
        else:
            raise ValueError("Unknown constraint type")

    def get_next_state(self, current_position, direction):
        dx, dy = Actions.direction_to_vector(direction)
        next_state = (int(current_position[0] + dx), int(current_position[1] + dy))
        if not self.walls[next_state[0]][next_state[1]]:
            return next_state, True
        return (0, 0), False

    def get_successors(self, state):
        possible_dirs = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        
        successors = [
            ((nextx, nexty), action, self.cost((nextx, nexty)))
            for action in possible_dirs
            if (next_state := self.get_next_state(state, action))[1]
            for nextx, nexty in [next_state[0]]
        ]

        self.expanded_states += 1
        if state not in self.checked:
            self.checked[state] = True
            self.checked_list.append(state)

        return successors
