import random
import pygame
from sys import exit
import torch
import torch.nn as nn
import torch.nn.functional as Function
from matplotlib import pyplot as plt
import os

class NeuralNetwork(nn.Module):
    def __init__(self, inputs):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(inputs, 32)
        self.layer1 = nn.Linear(32, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64,32)
        self.output = nn.Linear(32,1)
        nn.init.uniform_(self.input.weight, -1, 1)
        nn.init.uniform_(self.layer1.weight, -1, 1)
        nn.init.uniform_(self.layer2.weight, -1, 1)
        nn.init.uniform_(self.layer3.weight, -1, 1)
        self.dropout = nn.Dropout(p=0.2)

    def FeedForward(self, x):
        x = Function.relu(self.input(x))
        x = self.dropout(x)
        x = Function.relu(self.layer1(x))
        x = self.dropout(x)
        x = Function.relu(self.layer2(x))
        x = self.dropout(x)
        x = Function.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x

class Player:
    def __init__(self):
        self.x = 50
        self.y = 200
        self.rect = pygame.Rect(self.x, self.y, 20 ,20)
        self.color = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
        self.Velocity = 0
        self.score = 0
        self.flap = False
        self.alive = True
        self.fitness = 0
        self.decision = None
        self.vision = [0.5, 1, 0.5]
        self.input = 3
        self.brain = self.inheritBrain()
    
    def inheritBrain(self):
        Neural_Network = NeuralNetwork(self.input) 
        models_folder = "Genetic_Algorithm\Best_Models"
        # models_folder = "ParticleSwarmOptimization_Algorithm\Best_Models"
        model_files = os.listdir(models_folder)
        selected_models = random.sample(model_files, min(5, len(model_files)))
        for _, model_file in enumerate(selected_models):
            model_path = os.path.join(models_folder, model_file)
            Neural_Network.load_state_dict(torch.load(model_path))
        return Neural_Network

    def draw(self, window):
        pygame.draw.rect(window, 
                         self.color,
                         self.rect)
        
    def Ground_Collission(self, ground):
        return pygame.Rect.colliderect(self.rect, ground)
    
    def Sky_Collission(self):
        return bool(self.rect.y < 30)
    
    def Pipe_Collision(self, Pipes):
        for Pipe in Pipes:
            if pygame.Rect.colliderect(self.rect, Pipe.topRectangle):
                return True
            elif pygame.Rect.colliderect(self.rect, Pipe.bottomRectangle):
                return True
        return False
        
    def update(self, ground, Pipes):
        if not(self.Ground_Collission(ground) or self.Pipe_Collision(Pipes)):
            self.Velocity += 0.25
            self.rect.y += self.Velocity
            if self.Velocity > 5:
                self.Velocity = 5
            self.fitness += 1
        else:
            self.alive = False
            self.flap = False
            self.val = 0

    def Flap(self):
        if not self.flap and not self.Sky_Collission():
            self.flap = True
            self.Velocity = -5
        if self.Velocity >= 3:
            self.flap = False
    
    def IncreaseScore(self, Pipes_List):
        for Pipe in Pipes_List:
            if Pipe.passed:
                self.score += 1
                return
    
    def ClosestPipe(self, Pipes_List):
        for Pipe in Pipes_List:
            if not Pipe.passed:
                return Pipe

    def Vision(self, window, Pipes_List):
        closest_pipe = None
        min_distance = float('inf')
       
        for Pipe in Pipes_List:
            if not Pipe.passed:
                distance_to_pipe = abs(Pipe.x - self.rect.center[0])
                if distance_to_pipe < min_distance:
                    min_distance = distance_to_pipe
                    closest_pipe = Pipe

        if closest_pipe:
            self.vision[0] = max(0, self.rect.center[1] - closest_pipe.topRectangle.bottom) / 500
            pygame.draw.line(window, self.color, self.rect.center, (self.rect.center[0], closest_pipe.topRectangle.bottom))

            self.vision[1] = max(0, closest_pipe.x - self.rect.center[0]) / 500
            pygame.draw.line(window, self.color, self.rect.center, (closest_pipe.x, self.rect.center[1]))

            self.vision[2] = max(0, closest_pipe.bottomRectangle.top - self.rect.center[1]) / 500
            pygame.draw.line(window, self.color, self.rect.center, (self.rect.center[0], closest_pipe.bottomRectangle.top))

    def think(self):
        self.decision = self.brain.FeedForward(torch.tensor(self.vision))
        if float(self.decision) > 0.73:
            self.Flap()

class Population(Player):
    def __init__(self, size):
        self.players = []
        self.generation = 1
        self.size = size
        self.Highest_Score = 0
        for _ in range(0,self.size):
            self.players.append(Player())
    
    def Update_Player(self, window, ground, pipes):
        for Player in self.players:
            if Player.alive:
                Player.Vision(window, pipes)
                Player.think()
                Player.draw(window)
                Player.update(ground, pipes)
                Player.IncreaseScore(pipes)
    
    def score(self):
        alive_players = [player for player in self.players if player.alive]
        if alive_players:
            generation_score = max(player.score for player in alive_players)
            if self.Highest_Score < generation_score:
                self.Highest_Score = generation_score
            max_score = self.Highest_Score
            return max_score, generation_score
        else:
            return 0, 0
    
    def highest_fitness(self):
        alive_players = [player for player in self.players if player.alive]
        if alive_players:
            max_fitness = max(player.fitness for player in alive_players)
            return max_fitness
        else:
            return 0

    def current_generation(self):
        return self.generation
    
    def extinct(self):
        extinct = True
        for p in self.players:
            if p.alive:
                extinct = False
        return extinct
    
class Ground:
    ground_level = 700

    def __init__(self, width):
        self.x, self.y = 0, Ground.ground_level
        self.rect = pygame.Rect(self.x, self.y, width, 5)

    def draw(self, window):
        pygame.draw.rect(window, (255, 255, 255), self.rect)

class Pipes:
    width = 15
    opening = 200

    def __init__(self, width):
        self.x = width
        self.bottom = random.randint(10,300)
        self.top    = Ground.ground_level - self.bottom - self.opening
        self.bottomRectangle = pygame.Rect(0, 0, 0, 0)
        self.topRectangle = pygame.Rect(0, 0, 0, 0)
        self.passed = False
        self.OffScreen = False

    def drawPipes(self, window):
        self.bottomRectangle = pygame.Rect(self.x, 
                                           Ground.ground_level - self.bottom,
                                           self.width,
                                           self.bottom)
        pygame.draw.rect(window, (255, 255, 255), self.bottomRectangle)
        
        self.topRectangle = pygame.Rect(self.x,0,self.width,self.top)
        pygame.draw.rect(window, (255, 255, 255), self.topRectangle)
    
    def MoveForward(self):
        self.x -= 1
        if self.x + Pipes.width <= 50:
            self.passed = True
        if self.x <= -self.width:
            self.OffScreen = True

def Quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

def GameConfig():
    Height = 750
    Width = 550
    Window_Component = pygame.display.set_mode((Width, Height))
    Ground_Component = Ground(Width)
    Pipes = []
    return Height, Width, Window_Component, Ground_Component, Pipes

def display_Highestscore(window, x):
    font = pygame.font.SysFont(None, 30)
    text = font.render(f"Highest Score          : {int(x/50)}", True, (255, 255, 255))
    window.blit(text, (10, 10))

def display_GenerationScore(window, x):
    font = pygame.font.SysFont(None, 30)
    text = font.render(f"Generation Score    : {int(x/50)}", True, (255, 255, 255))
    window.blit(text, (10, 30))

def display_HighestFitness(window, x):
    font = pygame.font.SysFont(None, 30)
    text = font.render(f"Highest Fitness        : {int(x)}", True, (255, 255, 255))
    window.blit(text, (10, 50))

def display_Generation(window, x):
    font = pygame.font.SysFont(None, 30)
    text = font.render(f"Current Generation : {int(x)}", True, (255, 255, 255))
    window.blit(text, (10, 70))

def main():
    pygame.init()
    clock = pygame.time.Clock()
    population = Population(100)

    Spawn_Time = 10
    h, w, Window_Component, Ground_Component, Pipes_Component= GameConfig()

    while True:
        Quit()
        Window_Component.fill((0, 0, 0))
        Ground_Component.draw(Window_Component)
        
        if Spawn_Time <= 0:
            Pipes_Component.append(Pipes(w))
            Spawn_Time = 200
        Spawn_Time -= 1

        for Pipe in Pipes_Component:
            Pipe.drawPipes(Window_Component)
            Pipe.MoveForward()
            if Pipe.OffScreen:
                Pipes_Component.remove(Pipe)
        
        if not population.extinct():
            population.Update_Player(Window_Component, Ground_Component, Pipes_Component)
            highest_score, generation_score = population.score()
            highest_fitness = population.highest_fitness()
            display_Highestscore(Window_Component, highest_score)
            display_GenerationScore(Window_Component, generation_score)
            display_HighestFitness(Window_Component, highest_fitness)
            display_Generation(Window_Component, population.current_generation())
        else:
            Pipes_Component.clear()
            population = Population(100)

        clock.tick(60)
        pygame.display.flip()

if __name__ == '__main__':
    main()