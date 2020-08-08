{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Paper\
\
## Ablation study\
-  interaction objective function + different variations of this (position, velocity, acceleration) over mean trajectory over each mode\
-  Accuracy of HJ value function: Different interpolation approaches\
- Warm-starting: goal-based, safety-concerned (goal based + safety consideration), pre-compute solution and use nearest neighbour as warm-start, use a lower-fidelity model to get initial guess.\
- Safety attention: forward reachability, proximity (euclidean),\
\
## Baselines planners: RRT*, ORCA, MCTS, IPOPT (ours).\
- MCTS uses same objective (search vs continuous)\
- ORCA: simplified pedestrian model\
- RRT*: Sampling-based, replanning each step, assume static obstacles.\
\
## Core contribution:\
Gradients in a neural-network and leveraging this within a trajectory optimization}