import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

# Game constants
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors (R, G, B)
WHITE = (255, 255, 255)
RED = (200, 0, 0)       # Fire Core
ORANGE = (255, 165, 0)  # Fire Edge
BLUE1 = (0, 0, 255)     # Drone
BLUE2 = (0, 100, 255)   # Drone Propellers
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)   # Forest Floor
DARK_GREEN = (0, 100, 0)# Trees

BLOCK_SIZE = 20
SPEED = 40  # Training speed (higher = faster training visual)

class FireEnv:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('NeuroFire: AI Autonomous Firefighter Drone')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # Init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        # NO TAIL: Just the head
        
        self.ammo = 5 # Max Water Shots
        self.score = 0
        self.food = None # 'Food' is the Fire
        self.lake = Point(self.w - BLOCK_SIZE*3, self.h - BLOCK_SIZE*3) # Lake in bottom right
        
        self._place_fire()
        self.frame_iteration = 0
        return self.get_state_for_human_view()


    def get_state_for_human_view(self):
        # Helper, not used by agent directly usually but good for debug
        return self.head, self.food, self.score, self.ammo


    def _place_fire(self):
        # Randomly ignite a fire in the forest
        # Avoid the Lake Area
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            
            # Don't spawn on lake
            if abs(x - self.lake.x) < 50 and abs(y - self.lake.y) < 50:
                continue
            
            self.food = Point(x, y)
            if self.food != self.head:
                break


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input (for quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action) # update the head
        
        # 3. Check if game over
        reward = 0
        game_over = False
        
        # Collision with Wall
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # Loop limit
        if self.frame_iteration > 100 * (self.score + 1):
             game_over = True
             reward = -10
             return reward, game_over, self.score

        # 4. Mechanics Interaction
        
        # A) Reached FIRE
        if self.head == self.food:
            if self.ammo > 0:
                self.score += 1
                self.ammo -= 1 # Use Water
                reward = 10
                self._place_fire()
            else:
                # No Ammo! Penalize slightly to encourage finding lake, but don't kill
                reward = -5 # Tried to fight fire without water
        
        # B) Reached LAKE
        elif abs(self.head.x - self.lake.x) < BLOCK_SIZE*2 and abs(self.head.y - self.lake.y) < BLOCK_SIZE*2:
            if self.ammo < 5:
                self.ammo = 5 # Refill
                reward = 5 # Good job refilling
            else:
                reward = -1 # Don't camp at the lake if full

        else:
            # Just moving
            reward = -0.1 # Slight penalty for time wasting
            # Distance shaping (optional): Reward getting closer to target? 
            # Let's keep it simple sparse rewards first.
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        return False


    def _update_ui(self):
        self.display.fill(GREEN) # Forest Floor

        # Draw Trees
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                if (x + y) % (BLOCK_SIZE*2) == 0: # Checkered pattern
                     pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(x+5, y+5, 4, 4))

        # Draw Lake (Blue Zone)
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.lake.x - 20, self.lake.y - 20, 60, 60))
        pygame.draw.rect(self.display, (0, 150, 255), pygame.Rect(self.lake.x - 10, self.lake.y - 10, 40, 40))

        # Draw Fire
        pygame.draw.rect(self.display, ORANGE, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x+4, self.food.y+4, 12, 12))

        # Draw Drone (Triangle)
        x = self.head.x + BLOCK_SIZE//2
        y = self.head.y + BLOCK_SIZE//2
        
        # Determine nose of drone based on direction
        if self.direction == Direction.RIGHT:
            nose = (x + 10, y)
            left_wing = (x - 10, y - 10)
            right_wing = (x - 10, y + 10)
        elif self.direction == Direction.LEFT:
            nose = (x - 10, y)
            left_wing = (x + 10, y - 10)
            right_wing = (x + 10, y + 10)
        elif self.direction == Direction.UP:
            nose = (x, y - 10)
            left_wing = (x - 10, y + 10)
            right_wing = (x + 10, y + 10)
        else: # DOWN
            nose = (x, y + 10)
            left_wing = (x - 10, y - 10)
            right_wing = (x + 10, y - 10)

        pygame.draw.polygon(self.display, (200, 200, 200), [nose, left_wing, right_wing]) # Silver Body
        pygame.draw.circle(self.display, BLUE2, (int(x), int(y)), 5) # Central Rotor

        # Draw HUD
        text = font.render(f"Fires: {self.score} | Water: {self.ammo}/5", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # Action is [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
