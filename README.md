# RL Agent Sim
 A reinforcement learning simulator in Python

![image](src/eddie.png)

## What is it?
This is a simulation of a boxy creature whose goal is to pick up the blue 'fruit' and deliver it to the green-colored goal marker. The agent is rewarded for actions such as picking up the object and walking over to the goal with it.

## How does it work?
The script saves the agent's progress every 10 episodes to *agent.sav*. Each episode has a total of 200 steps.
When loading the script a second time, it will attempt to load a brain from *agent.sav* if it exists.

## Prerequisites:
- Numpy
- PyGame
- Pickle
