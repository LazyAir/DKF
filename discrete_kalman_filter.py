# -*- coding: UTF-8 -*-
"""
Discrete Kalman Filter simulation for robot navigation using the position data given by digital camera sensor
"""

import numpy as np
import pylab
import math
import matplotlib.pyplot

class KalmanFilter(object):
	def __init__(self, F, p_est_err, proc_sigma, meas_sigma):
		self.F = F # process transfer matrix
		self.P_hat = np.diag(p_est_err**2, k = 0) #posteriori estimate error covariance matrix
		self.Q = np.diag(proc_sigma**2, k = 0) #process noise matrix
		self.R = np.diag(meas_sigma**2, k = 0) #measurement noise matrix
		self.H = np.matrix(np.identity(F[0].size))	#gain matrix


	def setF(self, F): #update the process transfer matrix
		self.F = F

	#one iteration of kalman filter
	def one_iteration_KF(self, X_h, Z):
		#Time Updates Phase
		X_ = self.F * X_h  #(1) Project the state ahead
		X_hat =  X_h 
		P_ = self.F * self.P_hat * self.F.getT() + self.Q #(2) Project the error convariance ahead

		#Measurement Updates Phase	
		K = self.P_hat * self.H.getT() * (self.H * self.P_hat * self.H.getT() + self.R).getI() #(3) Compute the Kalman gain
		X_hat = X_ + K * (Z - self.H * X_) #(4) Update estimate with measurement
		self.P_hat = (np.identity(X_h.size) - K * self.H) * P_  #(5) Update the error covariance

		return X_hat

def generateSimulatedData(num_iter, X_true, X_sensor, theta, delta_t, meas_noise_sigma):
	#generate simulated data as the real position data obtained from digital camera
	x_array_true = np.zeros(num_iter)
	y_array_true = np.zeros(num_iter)

	x_array_sensor = np.zeros(num_iter) 
	y_array_sensor = np.zeros(num_iter)

	x_array_true[0] = X_true[0]
	y_array_true[0] = X_true[1]

	x_array_sensor[0] = X_sensor[0]
	y_array_sensor[0] = X_sensor[1]

	for i in xrange(1, num_iter):
		x_array_true[i] = x_array_true[i-1] + X_true[2]*np.cos(theta)*delta_t
		y_array_true[i] = y_array_true[i-1] + X_true[2]*np.sin(theta)*delta_t

		x_array_sensor[i] = np.random.normal(x_array_true[i], meas_noise_sigma[0],1)
		y_array_sensor[i] = np.random.normal(y_array_true[i], meas_noise_sigma[1],1)

	return x_array_true, y_array_true, x_array_sensor, y_array_sensor

def obtainFilteredData(kf_obj, num_iter, X_hat, x_array_sensor, y_array_sensor, velocity):
	x_array_predict = np.zeros(num_iter+1)  
	y_array_predict = np.zeros(num_iter+1)

	x_array_predict[0] = X_hat[0] 
	y_array_predict[0] = X_hat[1]
	#Iterations of kalman filter
	X_return = X_hat
	for i in xrange(0, num_iter):
		#####Z is the observed data vector from the sensor
		Z = np.matrix([[x_array_sensor[i]],[y_array_sensor[i]],[velocity]])   
		X_return = kf_obj.one_iteration_KF(X_return, Z)
		x_array_predict[i+1] = X_return[0]
		y_array_predict[i+1] = X_return[1]

	return x_array_predict, y_array_predict



def constructF(theta, delta_t): 
	d_x = np.cos(theta) * delta_t
	d_y = np.sin(theta) * delta_t
	F = np.matrix([[1,0,d_x],
				   [0,1,d_y],
				   [0,0,1]])
	return F


def simulation():
	#init the parameters of KF
	init_post_estimate_err = np.array([100, 100, 100]) #arbitrary initial values
	process_noise_sigma = np.array([np.exp(-5), np.exp(-5), 0.00001]) #suppose the process noise for x,y,v follws normal distribution with the specified standard deviation
	meas_noise_sigma = np.array([0.08, 0.08, 0.0001]) #mesasuing error for x,y,v follws normal distribution with the specified standard deviation
	F = np.identity(3) #random initilization of F, with update later
	kf = KalmanFilter(F, init_post_estimate_err, process_noise_sigma, meas_noise_sigma)


	######### suppose the robot departs from (0,0), with speed 0.1m/s, and take two turns to avoid the obstacles
	#1st segament of the robot trajectory
	##Initialization
	#initial posintion, velocity and heading, the initial value given by the camera
	X_true = np.array([0,0,0.1]) # State X: x, y, velocity
	X_sensor = np.array( [np.random.normal(X_true[0], meas_noise_sigma[0],1),
						  np.random.normal(X_true[1], meas_noise_sigma[1],1),
						  X_true[2] ] )

	theta = -np.pi/2 #initial angle against positive x-axis
	delta_t = 0.2
	L1 = 0.6  #displacement of segament 1
	iter_n_1 = int(math.ceil(L1/(X_true[2]*delta_t))) + 1
	X_hat = np.matrix(X_sensor).getT()

	x_array_true_1, y_array_true_1, x_array_sensor_1, y_array_sensor_1 \
		= generateSimulatedData(iter_n_1, X_true, X_sensor, theta, delta_t, meas_noise_sigma) 
	
	kf.setF(constructF(theta, delta_t)) #update the process transfer matrix F
	x_array_predict_1, y_array_predict_1 \
		= obtainFilteredData(kf, iter_n_1, X_hat, x_array_sensor_1, y_array_sensor_1, X_sensor[2])


	#########2nd segament of the robot trajectory
	X_true = np.array([x_array_true_1[iter_n_1-1], y_array_true_1[iter_n_1-1], X_true[2]])
	X_sensor = np.array([x_array_predict_1[iter_n_1], y_array_predict_1[iter_n_1], X_sensor[2]])

	theta = 0 
	L2 = 1.3
	iter_n_2 = int(math.ceil(L2/(X_true[2]*delta_t))) + 1
	X_hat = np.matrix(X_sensor).getT()

	x_array_true_2, y_array_true_2, x_array_sensor_2, y_array_sensor_2 \
		= generateSimulatedData(iter_n_2, X_true, X_sensor, theta, delta_t, meas_noise_sigma) 
	
	kf.setF(constructF(theta, delta_t))
	x_array_predict_2, y_array_predict_2 \
		= obtainFilteredData(kf, iter_n_2, X_hat, x_array_sensor_2, y_array_sensor_2, X_sensor[2])


	#########3nd segament of the robot trajectory
	X_true = np.array([x_array_true_2[iter_n_2-1], y_array_true_2[iter_n_2-1], X_true[2]])
	X_sensor = np.array([x_array_predict_2[iter_n_2], y_array_predict_2[iter_n_2], X_sensor[2]])

	theta = -np.pi/2 
	L3 = 0.6
	iter_n_3 = int(math.ceil(L3/(X_true[2]*delta_t))) + 1
	X_hat = np.matrix(X_sensor).getT()

	x_array_true_3, y_array_true_3, x_array_sensor_3, y_array_sensor_3 \
		= generateSimulatedData(iter_n_3, X_true, X_sensor, theta, delta_t, meas_noise_sigma) 
	
	kf.setF(constructF(theta, delta_t))
	x_array_predict_3, y_array_predict_3 \
		= obtainFilteredData(kf, iter_n_3, X_hat, x_array_sensor_3, y_array_sensor_3, X_sensor[2])



	####### simulation results and plots
	pylab.figure(figsize=(8,8))
	pylab.plot(x_array_true_1, y_array_true_1,'g-',label='truth trajectory')
	pylab.plot(x_array_sensor_1, y_array_sensor_1,'k+',label='noisy sensor data')
	pylab.plot(x_array_predict_1,y_array_predict_1,'b*-',label='KF filtered data')

	pylab.plot(x_array_true_2, y_array_true_2,'g-')
	pylab.plot(x_array_sensor_2, y_array_sensor_2,'k+')
	pylab.plot(x_array_predict_2,y_array_predict_2,'b*-')

	pylab.plot(x_array_true_3, y_array_true_3,'g-')
	pylab.plot(x_array_sensor_3, y_array_sensor_3,'k+')
	pylab.plot(x_array_predict_3,y_array_predict_3,'b*-')


	pylab.xlim(-0.2,1.6)
	pylab.ylim(-1.6,0.2)

	pylab.legend()
	pylab.xlabel('X-coordinate')
	pylab.ylabel('Y-coordinate')
	pylab.grid()
	pylab.show()


if __name__ == '__main__':
	simulation()
