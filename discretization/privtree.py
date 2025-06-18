import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation.trajectory_set import TrajectorySet
from config.parameter_carrier import ParameterCarrier
from tools.general_tools import GeneralTools
from tools.noise import Noise

from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional
import numpy as np
import copy as cop

@dataclass
class Rect:
    #center point
    x: float
    y: float
    #rect size
    width: float
    height: float

class Node:
    ...

@dataclass
class Node:
    boundry: Rect
    index: int = -1     # -1 means it's not leaf node
    is_leaf_node = True
    parent: Optional[Node] = None
    topleft: Optional[Node] = None
    topright: Optional[Node] = None
    bottomleft: Optional[Node] = None
    bottomright: Optional[Node] = None
    point_list: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

class PrivTree:
    def __init__(self, cc: ParameterCarrier, delta: float=1.8, split_threshold = 5000) -> None:
        self.max_x, self.min_x, self.max_y, self.min_y = float("inf"), float("-inf"), float("inf"), float("-inf")
        self.cc = cc
        self.root: Optional[Node] = None
        self.leaf_list: list[Node] = []
        self.delta = delta
        self.split_threshold = split_threshold
        self.eplison = cc.total_epsilon * cc.epsilon_partition[0]
        self.usable_state_number = 0
        self.subcell_neighbors_usable_index = [[]]
        self.real_subcell_index_to_usable_index_dict = []

    def add_neighbors_to_distribution(self, original_distribution, indicator=1):
        result_distribution = cop.deepcopy(original_distribution)
        neighbors_list = []
        for i in range(original_distribution.shape[0]):
            for j in range(original_distribution.shape[1]):
                if original_distribution[i, j] > 0:
                    neighbors_of_row_state = self.subcell_neighbors_usable_index[i]
                    neighbors_of_col_state = self.subcell_neighbors_usable_index[j]
                    for nei_state in neighbors_of_row_state:
                        neighbors_list.append((nei_state, j))
                    for nei_state in neighbors_of_col_state:
                        neighbors_list.append((i, nei_state))
        for distribution_neighbor_pairs in neighbors_list:
            row_index = distribution_neighbor_pairs[0]
            col_index = distribution_neighbor_pairs[1]
            result_distribution[row_index, col_index] = indicator
        return result_distribution
    
    def usable_state_neighbors(self, usable_state):
        neighbors = self.subcell_neighbors_usable_index[usable_state]
        return neighbors
    
    def usable_state_central_points(self):
        return np.array([(node.boundry.x, node.boundry.y) for node in self.leaf_list])

    def build_tree(self, trajectory_set: TrajectorySet):
        print("building tree")
        self.point_number = trajectory_set.get_whole_point_number()
        self.trajectory_number = trajectory_set.trajectory_number
        temp = []
        for trajectory_index in range(self.trajectory_number):
            trajectory = trajectory_set.give_trajectory_by_index(trajectory_index)
            for point in trajectory.get_trajectory_list():
                temp.append(point)
        temp = np.array(temp)
        self.min_x = np.min(temp[:, 0])
        self.max_x = np.max(temp[:, 0])
        self.min_y = np.min(temp[:, 1])
        self.max_y = np.max(temp[:, 1])
        self.root = Node(Rect((self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2, self.max_x - self.min_x, self.max_y - self.min_y))
        self.root.parent = self.root
        self.root.point_list = np.array(temp)
        self.usable_state_number = 1
        self.__build_tree(self.root, 0)
        self.__build_leaf_list()
        self.__set_trajectory_set_cell_num(trajectory_set)
        self.__set_neightbor()
        self.__priv_trace_trick()

    def __priv_trace_trick(self):
        """Some tricks to make this stupid code to work"""
        if __name__ != "__main__":
            usable_indicator = np.ones(len(self.leaf_list), dtype=bool)
            usable_number = int(np.sum(usable_indicator))
            self.real_subcell_index_to_usable_index_dict = np.zeros(usable_number, dtype=int) - 1
            self.real_subcell_index_to_usable_index_dict[usable_indicator] = np.arange(usable_number)
            gt1 = GeneralTools()
            usable_to_real = gt1.inverse_index_dict(usable_number, self.real_subcell_index_to_usable_index_dict)
            self.usable_subcell_index_to_real_index_dict = usable_to_real
            self.level2_subcell_to_large_cell_dict = np.arange(len(self.leaf_list))
            self.level1_cell_number = len(self.leaf_list)
            self.level2_subdividing_parameter = np.zeros(len(self.leaf_list))
            self.level2_borders = [(point.boundry.y + point.boundry.height / 2, point.boundry.y - point.boundry.height / 2, point.boundry.x + point.boundry.width / 2, point.boundry.x - point.boundry.width / 2) for point in self.leaf_list]

    def get_node_index(self, point: tuple[float, float]) -> int:
        assert self.max_x >= point[0] >= self.min_x and self.max_y >= point[1] >= self.min_y
        assert self.root is not None, "Build tree first"
        cur_node: Node = self.root
        while not cur_node.is_leaf_node:
            if point[0] > cur_node.boundry.x:
                if point[1] > cur_node.boundry.y:
                    cur_node = cur_node.topright
                else:
                    cur_node = cur_node.bottomright
            else:
                if point[1] > cur_node.boundry.y:
                    cur_node = cur_node.topleft
                else:
                    cur_node = cur_node.bottomleft
        return cur_node.index
    
    def print_tree(self):
        assert self.root is not None, "Build tree first"
        queue = deque([self.root])
        while len(queue) != 0:
            cur = queue.pop()
            print(cur.is_leaf_node)
            if not cur.is_leaf_node:
                queue.appendleft(cur.topleft)
                queue.appendleft(cur.topright)
                queue.appendleft(cur.bottomleft)
                queue.appendleft(cur.bottomright)
    
    def draw_graph(self, trajectory_set = None, showing_GUI = False):
        assert self.root is not None, "Build tree first"
        print("drawing graph")
        def draw_node(ax, node):
            # Recursively draw child nodes
            if not node.is_leaf_node:
                draw_node(ax, node.topleft)
                draw_node(ax, node.topright)
                draw_node(ax, node.bottomleft)
                draw_node(ax, node.bottomright)
            else:
                # Draw the boundary of the current node
                rect = node.boundry
                ax.add_patch(Rectangle(
                    (rect.x - rect.width / 2, rect.y - rect.height / 2),
                    rect.width,
                    rect.height,
                    fill=False,
                    edgecolor='black'
                ))
        fig, ax = plt.subplots()
        if trajectory_set is not None:
            num_trajectories = trajectory_set.trajectory_number
            cmap = plt.cm.jet
            norm = Normalize(vmin=0, vmax=num_trajectories)
            for idx in range(num_trajectories):
                trajectory = trajectory_set.give_trajectory_by_index(idx)
                points = trajectory.get_trajectory_list()
                color = cmap(norm(idx))
                if len(points) >= 2:
                    x_vals, y_vals = zip(*points)
                    ax.plot(x_vals, y_vals, linewidth=0.8, alpha=0.6, color=color)  # adjust style if needed
        draw_node(ax, self.root)
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_ylim(self.min_y, self.max_y)
        ax.set_aspect('equal')
        plt.title("Quadtree Visualization")
        if showing_GUI:
            plt.show()
        plt.savefig("./grid_layout.png")
        plt.close('all')

    def __build_tree(self, node: Node, depth: int):
        """Recusive function for building tree"""
        bias = len(node.point_list) - depth * self.delta
        noise_tool = Noise()
        noisy_bias = noise_tool.add_laplace(np.array([[bias]]), self.eplison, 1)
        print("depth:", depth, "bias:", noisy_bias)
        if noisy_bias > self.split_threshold and depth <= 100:
            self.__split_node(node)
            self.usable_state_number += 3   # minus one parent node add four child node
            self.__build_tree(node.topleft, depth+1)
            self.__build_tree(node.topright, depth+1)
            self.__build_tree(node.bottomleft, depth+1)
            self.__build_tree(node.bottomright, depth+1)

    def __build_leaf_list(self):
        assert self.root is not None, "Build tree first"
        queue = deque([self.root])
        self.leaf_list = []
        while len(queue) != 0:
            node = queue.pop()
            if node.is_leaf_node:
                node.index = len(self.leaf_list)
                self.leaf_list.append(node)
            else:
                queue.appendleft(node.topleft)
                queue.appendleft(node.topright)
                queue.appendleft(node.bottomleft)
                queue.appendleft(node.bottomright)

    def __split_node(self, node: Node):
        assert node.is_leaf_node, "only leaf node can be splited."
        node.is_leaf_node = False
        center_x = node.boundry.x
        center_y = node.boundry.y
        width = node.boundry.width / 2
        height = node.boundry.height / 2
        left = center_x - width
        right = center_x + width
        top = center_y + height
        bottom = center_y - height
        node.topleft = Node(Rect((center_x + left) / 2, (center_y + top) / 2, width, height), parent=node)
        node.topright = Node(Rect((center_x + right) / 2, (center_y + top) / 2, width, height), parent=node)
        node.bottomleft = Node(Rect((center_x + left) / 2, (center_y + bottom) / 2, width, height), parent=node)
        node.bottomright = Node(Rect((center_x + right) / 2, (center_y + bottom) / 2, width, height), parent=node)
        self.__split_list(node)
    
    def __split_list(self, node: Node):
        """Should be call after split node"""
        assert not node.is_leaf_node
        x, y = node.point_list[:, 0], node.point_list[:, 1]
        top = y > node.boundry.y
        right = x > node.boundry.x
        node.topright.point_list = node.point_list[top & right]
        node.topleft.point_list = node.point_list[top & ~right]
        node.bottomright.point_list = node.point_list[~top & right]
        node.bottomleft.point_list = node.point_list[~top & ~right]
        node.point_list = np.empty((0, 2))

    def __set_trajectory_set_cell_num(self, trajectory_set):
        counter = 0
        trajectory_number = trajectory_set.trajectory_number
        for trajectory_index in range(trajectory_number):
            counter += 1
            print(f"\rsetting trajectory: {round(counter / trajectory_number * 100, 1)}% ", end="")
            trajectory = trajectory_set.give_trajectory_by_index(trajectory_index)
            trajectory_cell_num = [self.get_node_index(point) for point in trajectory.get_trajectory_list()]
            trajectory.give_level1_index_array(np.array(trajectory_cell_num))
            # I don't want to use level 2
            trajectory.level2_cell_index_sequence = np.array(trajectory_cell_num)
        print()

    def __set_neightbor(self):
        self.subcell_neighbors_usable_index = [[] for _ in range(len(self.leaf_list))]
        leaf_count = len(self.leaf_list)
        for i, node in enumerate(self.leaf_list):
            print(f"\rsetting neightbor: {round(i / leaf_count * 100, 1)}% ", end="")
            # Climb to grandparent
            #start_node = node.parent.parent
            #candidate_nodes = self.__collect_leaf_descendants(start_node)
            for neighbor in self.leaf_list:
                if neighbor == node:
                    continue
                if self.__are_neighbors(node.boundry, neighbor.boundry):
                    self.subcell_neighbors_usable_index[i].append(neighbor.index)
        print()
        self.subcell_neighbors_usable_index = [np.array(l) for l in self.subcell_neighbors_usable_index]

    def __collect_leaf_descendants(self, node: Node) -> list[Node]:
            """Collect all leaf descendants under given node"""
            if node is None:
                return []
            queue = deque([node])
            result = []
            while queue:
                current = queue.pop()
                if current.is_leaf_node:
                    result.append(current)
                else:
                    queue.extendleft([current.topleft, current.topright, current.bottomleft, current.bottomright])
            return result
    
    def __are_neighbors(self, a: Rect, b: Rect) -> bool:
        """
        Check if two rectangles a and b are adjacent or overlapping.
        Returns True if they touch by edge or corner (8-connected).
        """
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        max_dx = (a.width + b.width) / 2
        max_dy = (a.height + b.height) / 2
        return dx <= max_dx + 0.0001 and dy <= max_dy + 0.0001

if __name__ == "__main__":
    from data_preparation.data_preparer import DataPreparer
    from config.parameter_setter import ParSetter
    par = ParSetter().set_up_args()
    pc = ParameterCarrier(par)
    data_preparer = DataPreparer(par)
    trajectory_set = data_preparer.get_trajectory_set()
    tree = PrivTree(pc)
    tree.build_tree(trajectory_set)
    tree.draw_graph(None, showing_GUI=True)
