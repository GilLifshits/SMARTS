# Robust-Simulation-for-DRL-Autonomous-Vehicle

## Table of Contents  
  * [Authors](#authors)
  * [Introduction](#introduction)
  * [Goals](#goals)
  * [Data Description](#data-description)
  * [Tools and Methods](#tools-and-methods)

## Authors
[Tal Amar](https://github.com/Tal-Amar), [Itay Saig](https://github.com/Itay-Saig), [Ido Pascaro](https://github.com/idopasc)

## Introduction
This research proposes a novel, distributed learning-based approach designed to facilitate collaborative navigation among autonomous vehicles within local environments. The objective is to create a
distributed and robust platform for autonomous driving that can support a varying number of vehicles and complex scenarios. Specifically, we aim to optimize route planning to ensure efficient traffic flow while mitigating instances of drastic speed reductions (”starvation”).

Our contribution involves the development of Deep Reinforcement Learning (DRL) models and adapting them to an advanced and realistic simulator known as [’SMARTS’](https://arxiv.org/abs/2010.09776) (Scalable Multi-Agent Reinforcement Learning Training School).

## Goals
Our research is focused on developing a distributed, versatile, and resilient platform for autonomous driving, capable of accommodating a changing number of vehicles. This platform is designed to adapt to new scenarios beyond its training data and manage a variable number of vehicles within its operational scope. The objective is to optimize routes for most vehicles without significantly decreasing their speed to prevent delays. We are building upon an existing model that employs generative deep reinforcement learning, aiming to enhance its efficacy and robustness.

## Data Description
Unlike supervised and unsupervised learning, our research harnesses the power of a generative model to produce data. The data that is used as the input of the model is actually obtained from a simulation. I.e., at each step (which will be defined for each ∆t), it is possible to extract new data, which is vital information about vehicle positions, speeds, distances, desired destinations, and more. The model then outputs decision-making actions for each vehicle, such as speed adjustments or lane changes, tailored to the specific problem at hand.

This process is central to the domain of deep reinforcement learning, where information is received, processed, and translated into decisions that impact reality. Feedback in the form of rewards is received based on decision quality at each step, guiding the network’s weight updates through backpropagation.

## Tools and Methods
The primary tool we use is the SMARTS simulator, utilized for executing our Deep Reinforcement Learning (DRL) models. SMARTS operates as a Linux-based simulation platform mainly developed
in Python.

Our research methodology involves several stages:
-	Exploratory testing of simulator capabilities.
-	Evaluation of different RL algorithms for multi-agent interaction.
-	Integration of selected algorithms into SMARTS.
-	Development and validation of evaluation metrics.

Throughout these stages, we leverage SMARTS to simulate diverse driving scenarios, allowing us to train and assess the effectiveness of our DRL models in handling multi-agent interactions.
