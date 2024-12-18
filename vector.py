import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Vector:
    def __init__(self, dimensions, weights):
        """Dimensions: Stats counted in the Vector system | Weights: How much a stat is worth"""
        self.dimensions = dimensions
        self.points = np.empty((0, dimensions))
        self.weights = np.array(weights)
        self.players = {}

    """Helper functions"""

    def mask_point(self,point):
        mask = ~np.all(self.points == point, axis=1)

        return mask

    def add_point(self, *coordinates):
        if len(coordinates) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(coordinates)}")
        point = np.array(coordinates).reshape(1, -1)
        self.points = np.vstack((self.points, point))

    def remove_point(self, *coordinates):
        if len(coordinates) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(coordinates)}")
        point = np.array(coordinates).reshape(1, -1)
        mask = self.mask_point(point)
        self.points = self.points[mask]

    def closest_point(self, exclude, *target_coordinates, k=1):
        if len(target_coordinates) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(target_coordinates)}")
        
        # Apply exclusion mask
        mask = self.mask_point(exclude)
        filtered_points = self.points[mask]

        target = np.array(target_coordinates).reshape(1, -1)
        
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
        
    def lobby(self,player_name,number_of_players=1, skill_gap = 10, lobby_weights = [1.0,1.0,1.0]):
        """Creates a lobby"""
        lobby = []
        print("Searching for players")

        point = self.players[player_name]["point"]
        closest, distance = vectorDB.closest_point(point,*np.power(point,lobby_weights), k=number_of_players)

        for closest_p in closest:
            found_player = self.get_player_by_point(*closest_p)
            if found_player is player_name:
                continue

            lobby.append(self.players[found_player])
            print(f'{found_player} with elo: {self.players[found_player]["elo"]} stats: {self.players[found_player]["point"]}')
        
        return lobby

def random_players(vectorDB,num_of_players, num_of_games):
    for p in range(num_of_players):
        vectorDB.register(f"player_{p}")
        for g in range(num_of_players):
            game_stat = np.random.randint(0, 10, size=vectorDB.dimensions) 
            vectorDB.update(f"player_{p}",game_stat)

def plot_lobby(all,lobby, player_searching):
    points = lobby

    all = np.array([p for p in all if not any(np.allclose(p, pt) for pt in points)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    ax.scatter(player_searching[0], player_searching[1], player_searching[2], c='g', marker='v')
    ax.scatter(all[:, 0], all[:, 1], all[:, 2], c='b', marker='.')

    ax.set_title("3D Point Plot")
    ax.set_xlabel("K Point")
    ax.set_ylabel("D Point")
    ax.set_zlabel("A Point")
    plt.show()


if __name__ == "__main__":
    vectorDB = Vector(3, np.array([1.2, 1.0, 1.1]))

    random_players(vectorDB,200,5)

    lobby = vectorDB.lobby("player_0",10,lobby_weights = [1.0,1.0,1.0])
    points = np.array([player["point"] for player in lobby])
    plot_lobby(vectorDB.points,points,vectorDB.players["player_0"]["point"])