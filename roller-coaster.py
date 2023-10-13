import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import random as random
import amulet
from amulet_nbt import StringTag
from amulet.api.block import Block

"""Path to existing minecraft world"""
minecraft_level = amulet.load_level("C:/Users/Home/AppData/Roaming/.minecraft/saves/flat/")
game_version = ("java", (1, 20, 0))

"""The seed to use for the roller coaster, comment out to make random each time"""
random.seed(0)

"""Location that the roller coaster should be generated at"""
x = 0
y = -60
z = 0

"""Width and depth of the roller coaster"""
width = 10
depth = 10
if not depth % 2 == 0:
    raise Exception("depth must be even")

"""
The maximum height of the roller coaster, each layer is 3 blocks high
Note that small roller coasters or small values of the extend_amount can stop a roller coasters reaching their full height
"""
layers = 3

"""
The amount to extend each layer by
This should be a number between 0 and 1
0: no extension
1: fully extended
"""
extend_amount = 1

"""
When set to true, the area taken up by the roller coaster is cleaned up first
"""
clean_up_area = True


"""A bunch of constants used by the code, don't change unless you want it to crash"""
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

EMPTY = 0
TRACK = 1
BLOCKED = 2

FLAT = 0
ASCENDING_NORTH = 1
DESCENDING_NORTH = 2
ASCENDING_EAST = 3
DESCENDING_EAST = 4


class Square:
    north = False
    east = False
    south = False
    west = False
    height = 0
    type = EMPTY
    gradient = FLAT

    def is_straight(self):
        return self.east == self.west

    def is_flat_track(self):
        return self.type == TRACK and self.gradient == FLAT and self.height == 0

    def copy(self):
        copy = Square()
        copy.north = self.north
        copy.east = self.east
        copy.west = self.west
        copy.south = self.south
        copy.height = self.height
        copy.type = self.type
        copy.gradient = self.gradient
        return copy

    def get_track_name(self):
        if self.east and self.west:
            if self.gradient == FLAT:
                return 'east_west'
            elif self.gradient == ASCENDING_EAST:
                return 'ascending_east'
            elif self.gradient == DESCENDING_EAST:
                return 'ascending_west'
        elif self.south and self.east:
            return 'north_east'
        elif self.north and self.east:
            return 'south_east'
        elif self.west and self.south:
            return 'north_west'
        elif self.west and self.north:
            return 'south_west'
        elif self.north and self.south:
            if self.gradient == FLAT:
                return 'north_south'
            elif self.gradient == ASCENDING_NORTH:
                return 'ascending_north'
            elif self.gradient == DESCENDING_NORTH:
                return 'ascending_south'

        return None


class Level:
    def __init__(self):
        self.grid = np.zeros((width, depth), Square)
        for i in range(width):
            for j in range(depth):
                self.grid[i, j] = Square()

    def create_default_circuit(self):
        p = [0, 0]
        self.grid[p[0], p[1]].type = TRACK

        def left():
            while p[0] > 1:
                self.grid[p[0], p[1]].west = True
                self.grid[p[0] - 1, p[1]].east = True
                self.grid[p[0] - 1, p[1]].type = TRACK
                p[0] -= 1

        def right():
            while p[0] < width - 1:
                self.grid[p[0], p[1]].east = True
                self.grid[p[0] + 1, p[1]].west = True
                self.grid[p[0] + 1, p[1]].type = TRACK
                p[0] += 1

        def down():
            while p[1] > 0:
                self.grid[p[0], p[1]].south = True
                self.grid[p[0], p[1] - 1].north = True
                self.grid[p[0], p[1] - 1].type = TRACK
                p[1] -= 1

        def up_one():
            self.grid[p[0], p[1]].north = True
            self.grid[p[0], p[1] + 1].south = True
            self.grid[p[0], p[1] + 1].type = TRACK
            p[1] += 1

        def left_one():
            self.grid[p[0], p[1]].west = True
            self.grid[p[0] - 1, p[1]].east = True
            self.grid[p[0] - 1, p[1]].type = TRACK
            p[0] -= 1

        while True:
            right()
            if p[1] == depth - 1:
                break

            up_one()
            left()

            if p[1] == depth - 1:
                break

            up_one()

        left_one()
        down()

    def add_random_permutation(self):
        indices = [x for x in np.ndindex((width - 1, depth - 1))]
        random.shuffle(indices)

        rotated_index = None
        for index in indices:
            if self.try_rotate(index[0], index[1]):
                rotated_index = index
                break

        if rotated_index is None:
            return False

        orphan_path = Path(self, rotated_index[0], rotated_index[1])

        indices = [x for x in np.ndindex((width - 1, depth - 1))]
        random.shuffle(indices)

        for index in indices:
            if orphan_path.is_half_in(index[0], index[1]):
                if self.try_rotate(index[0], index[1]):
                    return True

        return False

    def add_random_permutations(self, num):
        for i in range(num):
            if not self.add_random_permutation():
                return i

        return num

    def try_rotate(self, i, j):
        if self.grid[i, j].east and not self.grid[i, j].north and not self.grid[i + 1, j].north:
            if self.grid[i, j + 1].east and not self.grid[i, j + 1].south and not self.grid[i + 1, j + 1].south:
                self.grid[i, j].east = False
                self.grid[i, j].north = True
                self.grid[i, j + 1].east = False
                self.grid[i, j + 1].south = True

                self.grid[i + 1, j].west = False
                self.grid[i + 1, j].north = True
                self.grid[i + 1, j + 1].west = False
                self.grid[i + 1, j + 1].south = True
                return True

        if self.grid[i, j].north and not self.grid[i, j].east and not self.grid[i, j + 1].east:
            if self.grid[i + 1, j].north and not self.grid[i + 1, j].west and not self.grid[i + 1, j + 1].west:
                self.grid[i, j].east = True
                self.grid[i, j].north = False
                self.grid[i, j + 1].east = True
                self.grid[i, j + 1].south = False

                self.grid[i + 1, j].west = True
                self.grid[i + 1, j].north = False
                self.grid[i + 1, j + 1].west = True
                self.grid[i + 1, j + 1].south = False
                return True

        return False

    def add_heights_to_path(self, path):
        num_flat = path.count_flat()

        num_height_changes = int(num_flat / 12)

        num_ascending = num_height_changes * 6
        num_non_ascending = num_flat - num_ascending

        flat_lengths = np.zeros((num_height_changes * 2 + 1), int)

        for i in range(num_non_ascending):
            flat_lengths[random.randrange(len(flat_lengths))] += 1

        class Context:
            current_height = 0
            going_up = True
            current = None
            next = None

        context = Context()

        def set_height():
            square = self.grid[context.current[0], context.current[1]]

            if not square.is_straight():
                square.height = context.current_height
            elif flat_lengths[0] > 0:
                flat_lengths[0] -= 1
                square.height = context.current_height
            else:
                if context.going_up:
                    square.height = context.current_height
                    context.current_height += 1

                    if context.current[0] < context.next[0]:
                        square.gradient = ASCENDING_EAST
                    elif context.current[0] > context.next[0]:
                        square.gradient = DESCENDING_EAST
                    elif context.current[1] < context.next[1]:
                        square.gradient = DESCENDING_NORTH
                    else:
                        square.gradient = ASCENDING_NORTH

                    if context.current_height == 3:
                        context.going_up = False
                        # left shift
                        flat_lengths[:-1] = flat_lengths[1:]
                else:
                    context.current_height -= 1
                    square.height = context.current_height

                    if context.current[0] < context.next[0]:
                        square.gradient = DESCENDING_EAST
                    elif context.current[0] > context.next[0]:
                        square.gradient = ASCENDING_EAST
                    elif context.current[1] < context.next[1]:
                        square.gradient = ASCENDING_NORTH
                    else:
                        square.gradient = DESCENDING_NORTH

                    if context.current_height == 0:
                        context.going_up = True
                        # left shift
                        flat_lengths[:-1] = flat_lengths[1:]

        previous = path.path[0]
        for part in path.path[1:]:
            context.current = previous
            context.next = part
            set_height()

            previous = part

    def add_heights_to_level(self):
        for path in self.get_all_paths():
            self.add_heights_to_path(path)

    def try_extend_path_at(self, i, j):
        if self.grid[i, j].type == EMPTY and self.grid[i + 1, j].type == EMPTY:
            if self.grid[i, j + 1].is_flat_track() and self.grid[i + 1, j + 1].is_flat_track():
                if self.grid[i, j + 1].east:
                    self.grid[i, j].north = True
                    self.grid[i, j].east = True
                    self.grid[i, j].depth = 0
                    self.grid[i, j].type = TRACK

                    self.grid[i + 1, j].north = True
                    self.grid[i + 1, j].west = True
                    self.grid[i + 1, j].depth = 0
                    self.grid[i + 1, j].type = TRACK

                    self.grid[i, j + 1].east = False
                    self.grid[i, j + 1].south = True
                    self.grid[i + 1, j + 1].west = False
                    self.grid[i + 1, j + 1].south = True
                    return True
        if self.grid[i, j].type == EMPTY and self.grid[i, j + 1].type == EMPTY:
            if self.grid[i + 1, j].is_flat_track() and self.grid[i + 1, j + 1].is_flat_track():
                if self.grid[i + 1, j].north:
                    self.grid[i, j].north = True
                    self.grid[i, j].east = True
                    self.grid[i, j + 1].south = True
                    self.grid[i, j + 1].east = True
                    self.grid[i, j].depth = 0
                    self.grid[i, j + 1].depth = 0

                    self.grid[i + 1, j].north = False
                    self.grid[i + 1, j].west = True
                    self.grid[i + 1, j + 1].south = False
                    self.grid[i + 1, j + 1].west = True
                    self.grid[i, j].type = TRACK
                    self.grid[i, j + 1].type = TRACK
                    return True
        if self.grid[i, j].is_flat_track() and self.grid[i + 1, j].is_flat_track():
            if self.grid[i, j + 1].type == EMPTY and self.grid[i + 1, j + 1].type == EMPTY:
                if self.grid[i, j].east:
                    self.grid[i, j].north = True
                    self.grid[i, j].east = False
                    self.grid[i + 1, j].north = True
                    self.grid[i + 1, j].west = False
                    self.grid[i, j + 1].east = True
                    self.grid[i, j + 1].south = True
                    self.grid[i + 1, j + 1].west = True
                    self.grid[i + 1, j + 1].south = True
                    self.grid[i, j + 1].depth = 0
                    self.grid[i + 1, j + 1].depth = 0
                    self.grid[i, j + 1].type = TRACK
                    self.grid[i + 1, j + 1].type = TRACK
                    return True
        if self.grid[i, j].is_flat_track() and self.grid[i, j + 1].is_flat_track():
            if self.grid[i + 1, j].type == EMPTY and self.grid[i + 1, j + 1].type == EMPTY:
                if self.grid[i, j].north:
                    self.grid[i, j].north = False
                    self.grid[i, j].east = True
                    self.grid[i, j + 1].south = False
                    self.grid[i, j + 1].east = True
                    self.grid[i + 1, j].depth = 0
                    self.grid[i + 1, j + 1].depth = 0

                    self.grid[i + 1, j].north = True
                    self.grid[i + 1, j].west = True
                    self.grid[i + 1, j + 1].south = True
                    self.grid[i + 1, j + 1].west = True
                    self.grid[i + 1, j].type = TRACK
                    self.grid[i + 1, j + 1].type = TRACK
                    return True

        return False

    def extend_path_until_full(self):
        def extend_path_iterated():
            for i in range(width - 1):
                for j in range(depth - 1):
                    if self.try_extend_path_at(i, j):
                        return True

            return False

        while True:
            if random.uniform(0, 1) > extend_amount:
                break
            if not extend_path_iterated():
                break

    def get_upper_layer(self):
        upper_level = Level()
        for i in range(width):
            for j in range(depth):
                if self.grid[i, j].height == 3:
                    upper_level.grid[i, j] = self.grid[i, j].copy()
                    upper_level.grid[i, j].height = 0
                elif not self.grid[i, j].height == 0:
                    upper_level.grid[i, j].type = BLOCKED

        return upper_level

    def remove_upper_layer(self):
        for i in range(width):
            for j in range(depth):
                if self.grid[i, j].height == 3:
                    self.grid[i, j].east = False
                    self.grid[i, j].north = False
                    self.grid[i, j].south = False
                    self.grid[i, j].west = False
                    self.grid[i, j].height = 0
                    self.grid[i, j].type = EMPTY

    def get_all_paths(self):
        paths = []
        for i in range(width):
            for j in range(depth):
                if self.grid[i, j].type != TRACK:
                    continue

                is_part_of_existing_path = False
                for path in paths:
                    if path.mask[i, j]:
                        is_part_of_existing_path = True
                        break

                if is_part_of_existing_path:
                    continue

                paths.append(Path(self, i, j))

        return paths

    def render_minecraft(self, delta_y):
        for i in range(width):
            for j in range(depth):
                p = self.grid[i, j]

                if not p.type == TRACK:
                    continue

                shape = p.get_track_name()

                if shape.startswith('ascending'):
                    block = Block("minecraft", "powered_rail", {"shape": StringTag(shape), "powered": StringTag("true")})
                else:
                    block = Block("minecraft", "rail", {"shape": StringTag(shape)})

                minecraft_level.set_version_block(i + x, y + delta_y + p.height, j + z, "minecraft:overworld", game_version, block)

    def render_matplotlib(self, include_heights=False, include_blocked=False):
        ax = plt.gca()
        plt.xlim(0, width)
        plt.ylim(0, depth)
        ax.set_xticks(np.arange(0, width, 1))
        ax.set_yticks(np.arange(0, depth, 1))
        ax.set_aspect("equal")
        plt.grid()

        for i in range(0, width):
            for j in range(0, depth):
                p = self.grid[i, j]

                if include_heights:
                    if p.height == 0:
                        color = '#666666'
                    elif p.height == 1:
                        color = '#999999'
                    elif p.height == 2:
                        color = '#cccccc'
                    else:
                        color = '#ffffff'

                    rect = Rectangle((i, j), 1, 1, facecolor=color)
                    ax.add_patch(rect)

                if include_blocked:
                    if p.type == BLOCKED or not p.gradient == FLAT:
                        rect = Rectangle((i, j), 1, 1, facecolor='#777799')
                        ax.add_patch(rect)

                    p = self.grid[i, j]
                    if p.east and not self.grid[i + 1, j].type == BLOCKED:
                        ax.plot([i + 0.5, i + 1.5], [j + 0.5, j + 0.5], color='r')
                    if p.north and not self.grid[i, j + 1].type == BLOCKED:
                        ax.plot([i + 0.5, i + 0.5], [j + 0.5, j + 1.5], color='r')

                    continue

                if p.east:
                    ax.plot([i + 0.5, i + 1.5], [j + 0.5, j + 0.5], color='r')
                if p.north:
                    ax.plot([i + 0.5, i + 0.5], [j + 0.5, j + 1.5], color='r')

        plt.show()


class Path:
    def __init__(self, level, start_i, start_j):
        self.mask = np.zeros((width, depth), bool)
        self.path = []
        self.level = level
        self.start_i = start_i
        self.start_j = start_j
        self.is_loop = False
        self.find_path()

    def find_path(self):
        """	Finds a path that includes the given start_i and start_j
            This point does not have to be at the start of the path
        """
        starting_moves = self.get_starting_moves()

        last_move, i, j = starting_moves[0], self.start_i, self.start_j
        # flip the starting move, so we don't move in the same direction
        last_move = (last_move + 2) % 4

        self.path.append([i, j])
        self.mask[i, j] = True

        while True:
            last_move, i, j = self.move_once(last_move, i, j)

            if last_move is None:
                # hit dead end
                self.is_loop = False
                break

            if self.mask[i, j]:
                # path is a loop
                self.is_loop = True
                return

            self.path.append([i, j])
            self.mask[i, j] = True

        # now to explore in the other direction along the path
        if len(starting_moves) < 2:
            return

        last_move, i, j = starting_moves[1], self.start_i, self.start_j
        # flip the starting move, so we don't move in the same direction
        last_move = (last_move + 2) % 4

        while True:
            last_move, i, j = self.move_once(last_move, i, j)
            if last_move is None:
                # hit dead end
                break

            self.path.insert(0, [i, j])
            self.mask[i, j] = True

    def count_flat(self):
        count = 0

        for p in self.path:
            if self.level.grid[p[0], p[1]].north == self.level.grid[p[0], p[1]].south:
                count += 1

        return count

    def get_starting_moves(self):
        moves = []
        if self.level.grid[self.start_i, self.start_j].east:
            moves.append(EAST)

        if self.level.grid[self.start_i, self.start_j].west:
            moves.append(WEST)

        if self.level.grid[self.start_i, self.start_j].north:
            moves.append(NORTH)

        if self.level.grid[self.start_i, self.start_j].south:
            moves.append(SOUTH)

        return moves

    def move_once(self, last_move, i, j):
        if (not last_move == WEST) and self.level.grid[i, j].east:
            return EAST, i + 1, j
        elif (not last_move == EAST) and self.level.grid[i, j].west:
            return WEST, i - 1, j
        elif (not last_move == SOUTH) and self.level.grid[i, j].north:
            return NORTH, i, j + 1
        elif (not last_move == NORTH) and self.level.grid[i, j].south:
            return SOUTH, i, j - 1
        else:
            return None, i, j

    # returns true if the 2x2 square at i, j occupies two squares in the path
    def is_half_in(self, i, j):
        if self.mask[i, j] and self.mask[i + 1, j]:
            if (not self.mask[i, j + 1]) and (not self.mask[i + 1, j + 1]):
                return True

        if (not self.mask[i, j]) and (not self.mask[i + 1, j]):
            if self.mask[i, j + 1] and self.mask[i + 1, j + 1]:
                return True

        if self.mask[i, j] and self.mask[i, j + 1]:
            if (not self.mask[i + 1, j]) and (not self.mask[i + 1, j + 1]):
                return True

        if (not self.mask[i, j]) and (not self.mask[i, j + 1]):
            if self.mask[i + 1, j] and self.mask[i + 1, j + 1]:
                return True

        return False

    def render_matplotlib(self):
        ax = plt.gca()
        plt.xlim(0, width)
        plt.ylim(0, depth)
        ax.set_xticks(np.arange(0, width, 1))
        ax.set_yticks(np.arange(0, depth, 1))
        ax.set_aspect("equal")
        plt.grid()

        for i in range(0, width):
            for j in range(0, depth):
                if self.mask[i, j]:
                    rect = Rectangle((i, j), 1, 1, facecolor='#9999ff')
                    ax.add_patch(rect)

        plt.show()


class RollerCoaster:
    levels = []

    def __init__(self):
        if clean_up_area:
            self.clean_up_area()

        # the bottom layer
        level = Level()
        self.levels.append(level)

        level.create_default_circuit()
        level.add_random_permutations(100)

        # intermediate layers
        for _ in range(layers - 2):
            level.add_heights_to_level()

            upper_layer = level.get_upper_layer()

            self.levels.append(upper_layer)
            level.remove_upper_layer()

            level.extend_path_until_full()
            upper_layer.extend_path_until_full()

            level = upper_layer

        # the top layer will only have heights added
        # no extend_path_until_full() needed
        level.add_heights_to_level()

    def clean_up_area(self):
        for i in range(width):
            for j in range(depth):
                for h in range(layers * 3):
                    block = Block("minecraft", "air")
                    minecraft_level.set_version_block(x + i, y + h, z + j, "minecraft:overworld", game_version, block)

    def render_to_minecraft(self):
        delta_y = 0
        for level in self.levels:
            level.render_minecraft(delta_y)
            delta_y += 3


roller_coaster = RollerCoaster()
roller_coaster.render_to_minecraft()

minecraft_level.save()
minecraft_level.close()
