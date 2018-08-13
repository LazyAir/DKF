# Discrete Kalman Filter

## 1. SUMMARY
This is a small project for simple simulation of indoor robot navigation with Discrete Kalman Filter correcting the noisy position information given by positioning sensors.

In this project, we assume that indoor robot is required to move automatically from the source position to the destination position in a complex environment, where there are several obstacles, but the ground of other work space is flat. The position information and a global map of work space are given by a digital camera in real-time. The robot can run along a straight and rotate itself when it is not running. 

For illustration, imagine a scenario as in the Figure 1: the blue and red blocks are the source and destination position of the robot respectively, and the black block is an obstacle. A path planning algorithm, such as A*, suggests the shortest path from the source to the target by avoiding collision against the obstacles for the robot. The white line is the suggested path.
<div align=center>
  <img width="400" height="300" src="https://github.com/LazyAir/DKF/blob/master/imgs/Picture1.png"/>
</div>
<p align=center>  Figure 1: The may of robots and obstacles </p>
<br/>

Then the robot tries to move and follow the path from source to destination and its real-time position is captured by digital cameras. However, the digital cameras have measurement error such that the returned position information is not accurate enough. Therefore, we use Discrete Kalman Filter (DKF) to correct the noisy position information. 

<div align=center>
  <img width="695" height="339" src="https://github.com/LazyAir/DKF/blob/master/imgs/Picture2.png"/>
</div>
<p align=center>  Figure 2: Procedure of Kalman Filter </p>
<br/>

The procedure of DKF is shown as in Figure 2, which is a redrawn picture based on the source figure in the following paper:

> Bishop, Gary, and Greg Welch. "An introduction to the Kalman filter." Proc of SIGGRAPH, Course 8, no. 27599-3175 (2001): 59.

The simulation of indoor robot navigation with Discrete Kalman Filter is implemented by the code in discrete_kalman_filter.py and the simulation result is shown in Figure 3. 

<div align=center>
  <img width="650" height="453" src="https://github.com/LazyAir/DKF/blob/master/imgs/Picture3.png"/>
</div>
<p align=center>Figure 3: Simulation of robot navigation with Kalman Filter algorithm to correct the position data </p>
<br/>

## 2. Building Tools
Python

run <code>python discrete_kalman_filter.py</code> to get the simulation results.

## 3. License
MIT License
