########################################################################################################################
#	#																											  #    #
#	# This is a Python script for implenting optimization of a problem using Steepest Descent and Newton's Methods#    #
#	# The code is also able to choose between constant step size and variable step size using Armijo's rule		  #    #
#	# The below code optimizes the function z = 3*x^2 + y^4														  #    #
#	#																											  #    #
#	# The code optimizes the problem ans spits out number of iterations, function evaluations, gradient and		  #    #
#	# hessian evaluations. It also outputs the final point and the gradient at that point						  #    #
#	#																											  #    #
#	# >>>>>>>>||| Please use the initialize function to vary parameters as required for testing |||<<<<<<<<		  #    #
#	#																											  #    #
########################################################################################################################


import numpy as np





#Init

def initialize():


	x = 2 #initial x | suggested value: 1
	y = 2 #initial y | suggested value: -2
	step_param = 0 #step update type 0 = constant step, 1 = Armijo | suggested value: 1
	s0 = 1 #constant step size | suggested value: 0.05 | 0.001 is the largest for grad method
	s = 1 #initial changing step size | suggested value: 0.5
	sig = 0.25 #improvement threshold sigma | suggested value: 0.25
	bet = 0.3 #reduction factor beta | suggested value: 0.3
	g = [0 , 0] #gradient initialization value not important
	h = [0, 0, 0, 0] #hessian initialization value not important
	htype = 0 #Hessian calculation method | 0:actual hessian, 1:random, 2:first hess, 3:diag, 4:periodic, 5:memory
	eps = 0.00001 #epsilon | suggested value: 0.00001
	ittermax = 1000000 #maximum iterations | suggested value: 1,000,000
	method = 1 #0 for steepest descent, 1 for newton's | suggested value: 1
	func_eval_count = 0 #number of tiems the function was evaluated | must be 0
	func_grad_count = 0 #number of times the gradient was evaluated | must be 0
	func_hess_inv_count = 0 #number of times the hessian was evaluated | must be 0

	return x, y, step_param, s0, s, sig, bet, g, h, eps, ittermax, method, func_eval_count, func_grad_count, func_hess_inv_count, htype









#Functional evaluations

def func_eval(x, y): #Evaluate function at given point
	a=1
	b=100
	z = (a-x)**2 + b*(y-x**2)**2
	#z = (3*(x**2))+(y**4)
	return z

def func_grad(x, y, g, func_grad_count): #Evaluate gradient at given point

	a=1
	b=100
	g[0] = 2*(x-a)+4*b*x*(x**2-y)
	g[1] = 2*b*(y-x**2)
	#g[0] = 6*x
	#g[1] = 4*(y**3)
	func_grad_count = func_grad_count+1
	return g, func_grad_count

def func_hess_inv(x, y, h, func_hess_inv_count): #Evaluate Hessian at a given point

	a=1
	b=100
	if abs(4*b*(x**2-y)+2) <=0.00001:
		print("Danger!! Hessian inverse approaching inf!!")
		if abs(4*b*(x**2-y)+2) <=0.0000001:
			print("Unable to continue, Hessian inverse too large")
			exit()
	h[0] = 1/(2+4*b*(x**2-y))
	h[1] = (2*x)/(2+4*b*(x**2-y))
	h[2] = (2*x)/(2+4*b*(x**2-y))
	h[3] = (1/(2*b))+(4*x**2)/(2+4*b*(x**2-y))
	#act_ross_h[0] = 2+(4*b*(x**2-y))+(8*b*x**2)
	#act_ross_h[1] = -4*b*x
	#act_ross_h[2] = -4*b*x
	#act_ross_h[3] = 2*b
	#h[0] = 1/6
	#h[1] = 0
	#h[2] = 0
	#if y == 0: #prevent division by 0
	#	print ('Unable to calculate hessian due to 0 value of y! Please try a different initial guess')
	#	exit()
	#else:
	#	h[3] = 1/(12*(y**2))
	func_hess_inv_count = func_hess_inv_count+1
	return h, func_hess_inv_count

def func_random_hess_inv(x, y, h, func_hess_inv_count): #Give a random value for hessian at every point

	h = np.random.uniform(low=-1, high=1, size=(4,))
	func_hess_inv_count = func_hess_inv_count+1
	return h, func_hess_inv_count

def func_first_hess_inv(x, y, h, func_hess_inv_count): #Uses the first calculated hessian value for every point

	a=1
	b=100
	if func_hess_inv_count == 0:
		if abs(4*b*(x**2-y)+2) <=0.00001:
			print("Danger!! Hessian inverse approaching inf!!")
			if abs(4*b*(x**2-y)+2) <=0.0000001:
				print("Unable to continue, Hessian inverse too large")
				exit()
		h[0] = 1/(2+4*b*(x**2-y))
		h[1] = (2*x)/(2+4*b*(x**2-y))
		h[2] = (2*x)/(2+4*b*(x**2-y))
		h[3] = (1/(2*b))+(4*x**2)/(2+4*b*(x**2-y))
		#h[0] = 1/6
		#h[1] = 0
		#h[2] = 0
		#if y == 0: #prevent division by 0
		#	print ('Unable to calculate hessian due to 0 value of y! Please try a different initial guess')
		#	exit()
		#else:
		#	h[3] = 1/(12*(y**2))
		func_hess_inv_count = func_hess_inv_count+1		
	return h, func_hess_inv_count

def func_diag_hess_inv(x, y, h, func_hess_inv_count): #Uses just diagonal elements

	a=1
	b=100
	if abs(4*b*(x**2-y)+2) <=0.00001:
		print("Danger!! Hessian inverse approaching inf!!")
		if abs(4*b*(x**2-y)+2) <=0.0000001:
			print("Unable to continue, Hessian inverse too large")
			exit()
	h[0] = 1/(2+4*b*(x**2-y))
	h[1] = 0
	h[2] = 0
	h[3] = (1/(2*b))+(4*x**2)/(2+4*b*(x**2-y))
	#h[0] = 1/6
	#h[1] = 0
	#h[2] = 0
	#if y == 0: #prevent division by 0
	#	print ('Unable to calculate hessian due to 0 value of y! Please try a different initial guess')
	#	exit()
	#else:
	#	h[3] = 1/(12*(y**2))
	func_hess_inv_count = func_hess_inv_count+1
	return h, func_hess_inv_count


def func_periodic_hess_inv(x, y, h, func_hess_inv_count, period): #Calculates hessian inverse periodically

	a=1
	b=100
	if func_hess_inv_count%period == 0:
		if abs(4*b*(x**2-y)+2) <=0.00001:
			print("Danger!! Hessian inverse approaching inf!!")
			if abs(4*b*(x**2-y)+2) <=0.0000001:
				print("Unable to continue, Hessian inverse too large")
				exit()
		h[0] = 1/(2+4*b*(x**2-y))
		h[1] = (2*x)/(2+4*b*(x**2-y))
		h[2] = (2*x)/(2+4*b*(x**2-y))
		h[3] = (1/(2*b))+(4*x**2)/(2+4*b*(x**2-y))
		#h[0] = 1/6
		#h[1] = 0
		#h[2] = 0
		#if y == 0: #prevent division by 0
		#	print ('Unable to calculate hessian due to 0 value of y! Please try a different initial guess')
		#	exit()
		#else:
		#	h[3] = 1/(12*(y**2))
		func_hess_inv_count = func_hess_inv_count+1
	else:
		func_hess_inv_count = func_hess_inv_count+1	
	return h, func_hess_inv_count

def func_hess_inv_memory(x, y, h, func_hess_inv_count, old): #Evaluate Hessian at a given point but also remembers old Hessian

	a=1
	b=100
	if abs(4*b*(x**2-y)+2) <=0.00001:
		print("Danger!! Hessian inverse approaching inf!!")
		if abs(4*b*(x**2-y)+2) <=0.0000001:
			print("Unable to continue, Hessian inverse too large")
			exit()
	h[0] = (1/(2+4*b*(x**2-y)))*(1-old) + h[0]*old
	h[1] = ((2*x)/(2+4*b*(x**2-y)))*(1-old) + h[0]*old
	h[2] = ((2*x)/(2+4*b*(x**2-y)))*(1-old) + h[0]*old
	h[3] = ((1/(2*b))+(4*x**2)/(2+4*b*(x**2-y)))*(1-old) + h[0]*old
	#h[0] = (1/6)*(1-old) + h[0]*old
	#h[1] = 0*(1-old) + h[1]*old
	#h[2] = 0*(1-old) + h[2]*old
	#if y == 0: #prevent division by 0
	#	print ('Unable to calculate hessian due to 0 value of y! Please try a different initial guess')
	#	exit()
	#else:
	#	h[3] = (1/(12*(y**2)))*(1-old) + h[3]*old
	func_hess_inv_count = func_hess_inv_count+1
	return h, func_hess_inv_count


def func_hess_inv_BFGS(x, y, h, func_hess_inv_count, g, g_old, x_old, y_old):
	a=1
	b=100
	if func_hess_inv_count == 0:
		h[0] = 1
		h[1] = 0
		h[2] = 0
		h[3] = 1
	else:
		sn[0] = x - x_old
		sn[1] = y - y_old
		qn = g - g_old
		#convert to numpy for matrix operations



	if abs(4*b*(x**2-y)+2) <=0.00001:
		print("Danger!! Hessian inverse approaching inf!!")
		if abs(4*b*(x**2-y)+2) <=0.0000001:
			print("Unable to continue, Hessian inverse too large")
			exit()
	h[0] = 1/(2+4*b*(x**2-y))
	h[1] = (2*x)/(2+4*b*(x**2-y))
	h[2] = (2*x)/(2+4*b*(x**2-y))
	h[3] = (1/(2*b))+(4*x**2)/(2+4*b*(x**2-y))
	#act_ross_h[0] = 2+(4*b*(x**2-y))+(8*b*x**2)
	#act_ross_h[1] = -4*b*x
	#act_ross_h[2] = -4*b*x
	#act_ross_h[3] = 2*b
	#h[0] = 1/6
	#h[1] = 0
	#h[2] = 0
	#if y == 0: #prevent division by 0
	#	print ('Unable to calculate hessian due to 0 value of y! Please try a different initial guess')
	#	exit()
	#else:
	#	h[3] = 1/(12*(y**2))
	func_hess_inv_count = func_hess_inv_count+1
	return h, func_hess_inv_count











#Step size calculation

def step(x, y, s0, s, sig, bet, itter, step_param, g, h, method, func_eval_count, func_grad_count, func_hess_inv_count): #Determine step size
	
	if step_param == 0:#constant step size
		s = s0

	elif step_param == 1:#Armijo's rule

		if method == 0:#steepest descent method
			m = 1
			while func_eval(x, y) - func_eval((x-(bet**m)*s*g[0]), (y-(bet**m)*s*g[1])) < sig*(bet**m)*s*(g[0]*g[0]+g[1]*g[1]) :
				func_eval_count = func_eval_count + 2 #since function was evaluated twice to check while condition
				m = m+1
			s = bet**m
			func_eval_count = func_eval_count + 2 #since function was evaluated twice to check last while condition

		else:#newton's method
			m = 1
			while func_eval(x, y) - func_eval((x-(bet**m)*s*(h[0]*g[0]+h[1]*g[1])), (y-(bet**m)*s*(h[2]*g[0]+h[3]*g[1]))) < sig*(bet**m)*s*(g[0]*(h[0]*g[0]+h[1]*g[1])+g[1]*(h[2]*g[0]+h[3]*g[1])) :
				func_eval_count = func_eval_count + 2 #since function was evaluated twice to check while condition
				m = m+1
			s = bet**m
			func_eval_count = func_eval_count + 2 #since function was evaluated twice to check last while condition
	
	else:
		print('Invalid step_param variable in Initialization. Please use 0 for constant step size and 1 for Armijo\'s method')
		exit()

	return s, func_eval_count, func_grad_count, func_hess_inv_count









#Optimization methods

def steepest_desc(x, y, s0, s, sig, bet, g, eps, ittermax, step_param, func_eval_count, func_grad_count, func_hess_inv_count): #Steepest descent

	itter = 0
	while itter < ittermax:

		#Calculate gradient and hessian
		g, func_grad_count = func_grad(x, y, g, func_grad_count)
		h = 0 #hessian is not needed for steepest descent

		if (abs(g[0])+abs(g[1]))/2 > eps: #Arithmatic mean is larger than eps

			#Determine step size with method = 0
			s, func_eval_count, func_grad_count, func_hess_inv_count = step(x, y, s0, s, sig, bet, itter, step_param, g, h, 0, func_eval_count, func_grad_count, func_hess_inv_count)

			#Update x and y and print at each itter
			x = x - g[0]*s
			y = y - g[1]*s
			print(str(itter)+': '+str(x)+', '+str(y))
			itter = itter + 1

		else:#Terminate
			ittermax = 0

	#Print all the results
	print(str(itter)+': '+str(x)+', '+str(y))
	print('Done')
	print('Final point: '+str(x)+', '+str(y))
	print('Gradient='+str(g))
	print('\nEvaluation counters: Function Evaluation = '+str(func_eval_count)+', Gradient Evaluation = '+str(func_grad_count)+ ', Hessian Evaluation = '+str(func_hess_inv_count) )


def newt(x, y, s0, s, sig, bet, g, h, eps, ittermax, step_param, func_eval_count, func_grad_count, func_hess_inv_count, htype): #Newtons method

	itter = 0
	period = 100
	while itter < ittermax:

		#Calculate gradient and hessian
		g_old = g
		g, func_grad_count = func_grad(x, y, g, func_grad_count)
		if htype == 0:#Actual hessian at each step
			h, func_hess_inv_count = func_hess_inv(x, y, h, func_hess_inv_count)
		elif htype == 1:#Random hessian at each step
			h, func_hess_inv_count = func_random_hess_inv(x, y, h, func_hess_inv_count)
		elif htype == 2:#Just the first hessian
			h, func_hess_inv_count = func_first_hess_inv(x, y, h, func_hess_inv_count)
		elif htype == 3:#Diagonal hessian
			h, func_hess_inv_count = func_diag_hess_inv(x, y, h, func_hess_inv_count)
		elif htype == 4:#Periodic hessian
			h, func_hess_inv_count = func_periodic_hess_inv(x, y, h, func_hess_inv_count, period)
		elif htype == 5:#Memory hessian
			h, func_hess_inv_count = func_hess_inv_memory(x, y, h, func_hess_inv_count, 0.25)
		elif htype == 6:#BFGS
			h, func_hess_inv_count = func_hess_inv_BFGS(x, y, h, func_hess_inv_count, g, g_old, x_old, y_old)

		if (abs(g[0])+abs(g[1]))/2 > eps: #Arithmatic mean is larger than eps

			#Determine step size with method = 1
			s, func_eval_count, func_grad_count, func_hess_inv_count = step(x, y, s0, s, sig, bet, itter, step_param, g, h, 1, func_eval_count, func_grad_count, func_hess_inv_count)

			#Update x and y and print at each itter
			x_old = x
			y_old = y
			x = x - s*((h[0]*g[0])+(h[1]*g[1]))
			y = y - s*(h[2]*g[0]+h[3]*g[1])
			print(str(itter)+': '+str(x)+', '+str(y))
			itter = itter + 1

		else:#Terminate
			ittermax = 0

	#Print all the results
	print(str(itter)+': '+str(x)+', '+str(y))
	print('Done')
	print('Final point: '+str(x)+', '+str(y))
	print('Gradient='+str(g))
	z=func_eval(x,y)
	print('Value='+str(z))
	if htype == 4:
		func_hess_inv_count_corrected = func_hess_inv_count//period
	else:
		func_hess_inv_count_corrected = func_hess_inv_count
	print('\nEvaluation counters: Function Evaluation = '+str(func_eval_count)+', Gradient Evaluation = '+str(func_grad_count)+ ', Hessian Evaluation = '+str(func_hess_inv_count_corrected) )









#Main loop

def main():

	x, y, step_param, s0 , s, sig, bet, g, h, eps, ittermax, method, func_eval_count, func_grad_count, func_hess_inv_count, htype = initialize()
	if method == 0:
		steepest_desc(x, y, s0, s, sig, bet, g, eps, ittermax, step_param, func_eval_count, func_grad_count, func_hess_inv_count)
	elif method == 1:
		newt(x, y, s0, s, sig, bet, g, h, eps, ittermax, step_param, func_eval_count, func_grad_count, func_hess_inv_count, htype)
	else:
		print ('Invalid method varible in Initialization. Please use 0 for Steepest Descent and 1 for Newton\'s method')

main()