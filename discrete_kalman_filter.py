#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Discrete Kalman Filter simulation for robot navigation using the position data given by digital camera sensor
"""

import numpy as np
import pylab
import math

class KalmanFilter(object):
	def __init__(self, F, p_est_err, proc_sigma, meas_sigma):
		"""
		Initialize KF
		@param F: process transfer matrix
		@param p_est_err: posteriori estimate error standard deviation
		@param proc_sigma: process noise standard deviation
		@param meas_sigma: measurement standard deviation
		"""
		self.F = F
		self.P_hat = np.diag(p_est_err**2, k = 0) #posteriori estimate error covariance matrix
		self.Q = np.diag(proc_sigma**2, k = 0) #process noise matrix
		self.R = np.diag(meas_sigma**2, k = 0) #measurement noise matrix
		self.H = np.matrix(np.identity(F[0].size))	#gain matrix

	def setF(self, F):
		"""
		Update the process transfer matrix
		@param F: process transfer matrix
		"""
		self.F = F

	def one_iteration_KF(self, X_h, Z):
		"""
		Perform one iteration of kalman filter
		@param X_h: the posteriori state estimate from last iteration
		@param Z: actual sensor measurement
		@return X_hat: the posteriori state estimate of this iteration
		"""

		#Time Update Phase
		X_ = self.F * X_h  #(1) Project the state ahead
		X_hat =  X_h 
		P_ = self.F * self.P_hat * self.F.getT() + self.Q #(2) Project the error convariance ahead

		#Measurement Update Phase	
		K = self.P_hat * self.H.getT() * (self.H * self.P_hat * self.H.getT() + self.R).getI() #(3) Compute the Kalman gain
		X_hat = X_ + K * (Z - self.H * X_) #(4) Update estimate with measurement
		self.P_hat = (np.identity(X_h.size) - K * self.H) * P_  #(5) Update the error covariance

		return X_hat


def generateData(num_iter, X_true, Z, theta, delta_t, meas_noise_sigma):
	"""
	Generate series of true position data and the sensor position data obtained from digital camera
	@param num_iter: number of iterations
	@param X_true: true state
	@param Z: actual sensor measurement
	@param theta: angle between heading and positive x-axis
	@param delta_t: time interval of receiving data from sensor
	@param meas_noise_sigma: measurement standard deviation
	@return x_series_true, y_series_true, x_series_sensor, y_series_sensor: 
		    a series of true x-coordinates, true y-coordinates, sensor x-coordinates, sensor y-coordinates
	"""
	x_series_true = np.zeros(num_iter)
	y_series_true = np.zeros(num_iter)

	x_series_sensor = np.zeros(num_iter) 
	y_series_sensor = np.zeros(num_iter)

	x_series_true[0] = X_true[0]
	y_series_true[0] = X_true[1]

	x_series_sensor[0] = Z[0][0]
	y_series_sensor[0] = Z[1][0]

	for i in xrange(1, num_iter):
		x_series_true[i] = x_series_true[i-1] + X_true[2]*np.cos(theta)*delta_t
		y_series_true[i] = y_series_true[i-1] + X_true[2]*np.sin(theta)*delta_t

		x_series_sensor[i] = np.random.normal(x_series_true[i], meas_noise_sigma[0], 1)
		y_series_sensor[i] = np.random.normal(y_series_true[i], meas_noise_sigma[1], 1)

	return x_series_true, y_series_true, x_series_sensor, y_series_sensor

def filteredData(kf_obj, num_iter, X_hat, x_series_sensor, y_series_sensor, velocity):
	"""
	Obtain a series of predicted position data by kalman filter
	@param kf_obj: an instance of KF
	@param num_iter: number of iterations
	@param X_hat: posteriori state estimate of last iteration
	@param x_series_sensor: a series of sensor x-coordinates
	@param y_series_sensor: a series of sensor y-coordinates
	@param velocity: velocity of robot
	@return x_series_predict, y_series_predict: a series of predicted x-coordinates, y-coordinates
	"""
	x_series_predict = np.zeros(num_iter)  
	y_series_predict = np.zeros(num_iter)

	x_series_predict[0] = X_hat[0][0] 
	y_series_predict[0] = X_hat[1][0]
	#iterations of kalman filter
	X_return = X_hat
	for i in xrange(1, num_iter):
		Z = np.matrix([x_series_sensor[i], y_series_sensor[i], velocity]).getT()
		X_return = kf_obj.one_iteration_KF(X_return, Z)
		x_series_predict[i] = X_return[0][0]
		y_series_predict[i] = X_return[1][0]

	return x_series_predict, y_series_predict

def constructF(theta, delta_t): 
	"""
	Helper function to construct the process transfer matrix
	@param theta: angle between heading and positive x-axis
	@param delta_t: time interval of receiving data from sensor
	@return F: process transfer matrix
	"""
	d_x = np.cos(theta) * delta_t
	d_y = np.sin(theta) * delta_t
	F = np.matrix([[1,0,d_x],
				   [0,1,d_y],
				   [0,0,1]])
	
	return F


def simulation():
	"""
	Simulate robot navigation, generate true robote trajectories, the position data returned by sensor and the posteriori position estimate by KF.
	"""
	#init the parameters of KF
	init_post_estimate_err = np.array([100, 100, 100]) #arbitrary initial values
	process_noise_sigma = np.array([np.exp(-5), np.exp(-5), 0.00001]) #suppose the process noise for x,y,v follws normal distribution with the specified standard deviation
	meas_noise_sigma = np.array([0.08, 0.08, 0.0001]) #mesasuing error for x,y,v follws normal distribution with the specified standard deviation
	F = np.identity(3) #randomly initilize process transfer matrix F, will update later
	kf = KalmanFilter(F, init_post_estimate_err, process_noise_sigma, meas_noise_sigma)


	######### suppose the robot departs from (0,0), with speed 0.1m/s, and take two turns to avoid the obstacles
	#1st segament of the robot trajectory
	#initialize posintion, velocity and heading, and the initial values given by the camera
	X_true = np.array([0,0,0.1]) # true State X: x, y, velocity
	Z = np.matrix( [np.random.normal(X_true[0], meas_noise_sigma[0], 1),
						np.random.normal(X_true[1], meas_noise_sigma[1], 1),
						X_true[2] ]).getT() #initial simulated sensor measurement matrix
	theta = -np.pi/2 #initial angle against positive x-axis
	delta_t = 0.2 #time interval of receiving data from sensor
	L1 = 0.6  #displacement of segament 1
	iter_n_1 = int(math.ceil(L1/(X_true[2]*delta_t))) + 1
	X_hat = Z #arbitrarily initialize posteriori state estimate, here we use Z 

	x_series_true_1, y_series_true_1, x_series_sensor_1, y_series_sensor_1 \
		= generateData(iter_n_1, X_true, Z, theta, delta_t, meas_noise_sigma) 
	kf.setF(constructF(theta, delta_t)) #update the process transfer matrix F
	x_series_predict_1, y_series_predict_1 \
		= filteredData(kf, iter_n_1, X_hat, x_series_sensor_1, y_series_sensor_1, Z[2])


	#2nd segament of the robot trajectory
	X_true = np.array([x_series_true_1[iter_n_1-1], y_series_true_1[iter_n_1-1], X_true[2]])
	Z = np.matrix([x_series_sensor_1[iter_n_1-1], y_series_sensor_1[iter_n_1-1], Z[2]]).getT()
	theta = 0 
	L2 = 1.3
	iter_n_2 = int(math.ceil(L2/(X_true[2]*delta_t))) + 1
	X_hat = np.matrix([x_series_predict_1[iter_n_1-1], y_series_predict_1[iter_n_1-1], X_true[2]]).getT()

	x_series_true_2, y_series_true_2, x_series_sensor_2, y_series_sensor_2 \
		= generateData(iter_n_2, X_true, Z, theta, delta_t, meas_noise_sigma) 
	kf.setF(constructF(theta, delta_t))
	x_series_predict_2, y_series_predict_2 \
		= filteredData(kf, iter_n_2, X_hat, x_series_sensor_2, y_series_sensor_2, Z[2])


	#3nd segament of the robot trajectory
	X_true = np.array([x_series_true_2[iter_n_2-1], y_series_true_2[iter_n_2-1], X_true[2]])
	Z = np.matrix([x_series_sensor_2[iter_n_2-1], y_series_sensor_2[iter_n_2-1], Z[2]]).getT()
	theta = -np.pi/2 
	L3 = 0.3
	iter_n_3 = int(math.ceil(L3/(X_true[2]*delta_t))) + 1
	X_hat = np.matrix([x_series_predict_2[iter_n_2-1], y_series_predict_2[iter_n_2-1], X_true[2]]).getT()

	x_series_true_3, y_series_true_3, x_series_sensor_3, y_series_sensor_3 \
		= generateData(iter_n_3, X_true, Z, theta, delta_t, meas_noise_sigma) 
	kf.setF(constructF(theta, delta_t))
	x_series_predict_3, y_series_predict_3 \
		= filteredData(kf, iter_n_3, X_hat, x_series_sensor_3, y_series_sensor_3, Z[2])


	####### plot simulation results
	pylab.figure(figsize=(8,8))
	pylab.plot(x_series_true_1, y_series_true_1,'g-',label='true trajectory')
	pylab.plot(x_series_sensor_1, y_series_sensor_1,'k+',label='noisy sensor measurement data')
	pylab.plot(x_series_predict_1,y_series_predict_1,'b*-',label='KF filtered positoin data')

	pylab.plot(x_series_true_2, y_series_true_2,'g-')
	pylab.plot(x_series_sensor_2, y_series_sensor_2,'k+')
	pylab.plot(x_series_predict_2,y_series_predict_2,'b*-')

	pylab.plot(x_series_true_3, y_series_true_3,'g-')
	pylab.plot(x_series_sensor_3, y_series_sensor_3,'k+')
	pylab.plot(x_series_predict_3,y_series_predict_3,'b*-')

	pylab.xlim(-0.2,1.6)
	pylab.ylim(-1.6,0.2)
	pylab.legend()
	pylab.xlabel('X-coordinate')
	pylab.ylabel('Y-coordinate')
	pylab.grid()
	pylab.show()


if __name__ == '__main__':
	simulation()
