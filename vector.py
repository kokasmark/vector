import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange

import time

class Vector:
    def __init__(self, dimensions, weights):
        """Dimensions: Stats counted in the Vector system | Weights: How much a stat is worth"""
        self.dimensions = dimensions
        self.points = np.empty((0, dimensions))
        self.weights = np.array(weights)
        self.players = {}

    """Helper functions"""

    def add_point(self, *coordinates):
        if len(coordinates) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(coordinates)}")
        point = np.array(coordinates).reshape(1, -1)
        self.points = np.vstack((self.points, point))

    def remove_point(self, *coordinates):
        if len(coordinates) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(coordinates)}")
        point = np.array(coordinates).reshape(1, -1)
        mask = ~np.all(self.points == point, axis=1)
        self.points = self.points[mask]

    def closest_point(self,domain,exclude,*target, k=1):
        if len(target) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(target)}")
        
        # Apply exclusion mask
        mask = ~np.all(domain == exclude, axis=1)
        filtered_points = domain[mask]

        target = np.array(target).reshape(1, -1)
        
        # Correct distance calculation
        distances = np.linalg.norm(filtered_points - target, axis=1)
        closest_indices = np.argsort(distances)[:k]
        
        return filtered_points[closest_indices], distances[closest_indices]
    
    def get_player_by_point(self, *coordinates):
        if len(coordinates) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(coordinates)}")
        
        target_point = np.array(coordinates).reshape(1, -1) 

        player_points = np.vstack([data["point"] for data in self.players.values()])
        
        mask = np.all(player_points == target_point, axis=1)
        
        if np.any(mask):
            matching_player_index = np.where(mask)[0][0]
            matching_player = list(self.players.keys())[matching_player_index]
            return matching_player
        return None
    
    
    """Main functions"""
    def register(self, player_name):
        self.players[player_name] = {"point": np.array([0] * self.dimensions), "matches": 0}  # Store the actual point for the player
        self.add_point(*self.players[player_name]["point"])

    def update(self, player_name, stats):
        point = self.players[player_name]["point"]
        self.remove_point(*point)
        point = np.add(point, np.power(stats,self.weights))

        self.add_point(*point)

        self.players[player_name]["point"] = point
        self.players[player_name]["matches"] += 1

        self.elo(player_name)

    def elo(self,player_name):
        point = self.players[player_name]["point"]
        a = 0

        for stat in point:
            a += math.pow(stat,2)

        elo = math.sqrt(a)

        self.players[player_name]["elo"] = elo/self.players[player_name]["matches"]

        return elo
        
    def lobby(self,player_name,number_of_players=1, skill_gap = 10, max_search = 5,lobby_weights = [1.0,1.0,1.0]):
        """Creates a lobby"""

        lobby = {"domain": [], "players": []}
        done = False
        print("Searching for players")

        point = self.players[player_name]["point"]

        for search in range(max_search):
            mask = np.all((self.points >= point - skill_gap) & (self.points <= point + skill_gap), axis=1)
            domain = self.points[mask]

            lobby["domain"] = domain

            closest, distance = vectorDB.closest_point(domain,point,*np.power(point,lobby_weights), k=number_of_players)

            for closest_p in closest:
                found_player = self.get_player_by_point(*closest_p)

                if found_player is player_name:
                    continue
                
                if len(lobby["players"]) < number_of_players:
                    lobby["players"].append(self.players[found_player])
                    print(f'{found_player} with elo: {self.players[found_player]["elo"]} stats: {self.players[found_player]["point"]}')
                else:
                    done = True
                    break
            
            if done:
                print(f"Found lobby on {search} extent.")
                return lobby
            
            skill_gap+=1

        

def random_players(vectorDB,num_of_players, num_of_games):
    for p in range(num_of_players):
        vectorDB.register(f"player_{p}")
        for g in range(num_of_games):
            game_stat = np.random.randint(0, 10,vectorDB.dimensions) 
            vectorDB.update(f"player_{p}",game_stat)

import matplotlib.pyplot as plt
import numpy as np

def plot_lobby(lobby_elos, player_searching_elo):
    players_count = len(lobby_elos)
    x_positions = np.arange(players_count)  # X-axis indices for lobby players

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lobby players
    ax.scatter(
        x_positions, 
        lobby_elos, 
        color='r', 
        s=100, 
        label="Lobby Player"
    )
    for idx, elo in zip(x_positions, lobby_elos):
        ax.plot(
            [idx, idx], 
            [0, elo], 
            color='r', 
            linewidth=2, 
            alpha=0.7
        )

    # Highlight searching player
    ax.scatter(
        -1, 
        player_searching_elo, 
        color='g', 
        marker='v', 
        s=150, 
        label="Searching Player"
    )
    ax.plot(
        [-1, -1], 
        [0, player_searching_elo], 
        color='g', 
        linewidth=2, 
        alpha=0.7
    )

    ax.set_title(f"Lobby with {players_count} selected players")
    ax.set_xlabel("Players (Index)")
    ax.set_ylabel("ELO Value")
    ax.set_xticks(np.append(x_positions, -1))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    plt.show()




if __name__ == "__main__":
    start = time.time()
    vectorDB = Vector(3, np.array([1.2, 1.0, 1.1]))
    print(f'VectorDB Initialized in {time.time() - start}')
    start = time.time()
    random_players(vectorDB,1000,5)
    print(f'VectorDB Filled with dummy data in {time.time() - start}')
    start = time.time()

    lobby = vectorDB.lobby("player_0",number_of_players=20,skill_gap=3,max_search=10,lobby_weights = [1.0,1.0,1.0])
    print(f'Lobby filled in {time.time() - start}')

    points = np.array([player["elo"] for player in lobby["players"]])
    plot_lobby(points,vectorDB.players["player_0"]["elo"])