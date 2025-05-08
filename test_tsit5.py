import numpy as np
import matplotlib.pyplot as plt

class Tsit5:
		c1 = 0.161;
		c2 = 0.327;
		c3 = 0.9;
		c4 = 0.9800255409045097;
		c5 = 1;
		c6 = 1;
		a21 = 0.161;
		a31 = -0.008480655492356989;
		a32 = 0.335480655492357;
		a41 = 2.8971530571054935;
		a42 = -6.359448489975075;
		a43 = 4.3622954328695815;
		a51 = 5.325864828439257;
		a52 = -11.748883564062828;
		a53 = 7.4955393428898365;
		a54 = -0.09249506636175525;
		a61 = 5.86145544294642;
		a62 = -12.92096931784711;
		a63 = 8.159367898576159;
		a64 = -0.071584973281401;
		a65 = -0.028269050394068383;
		a71 = 0.09646076681806523; # b1 in paper
		a72 = 0.01; # b2 in paper
		a73 = 0.4798896504144996; # b3 in paper 
		a74 = 1.379008574103742;
		a75 = -3.290069515436081;
		a76 = 2.324710524099774;
		# b1 =    T,0.09468075576583945;
		# b2 =    T,0.009183565540343254;
		# b3 =    T,0.4877705284247616;
		# b4 =    T,1.234297566930479;
		# b5 =    T,-2.7077123499835256;
		# b6 =    T,1.866628418170587;
		# b7 =    T,0.015151515151515152;
		btilde1 = -0.00178001105222577714;
		btilde2 = -0.0008164344596567469;
		btilde3 = 0.007880878010261995;
		btilde4 = -0.1447110071732629;
		btilde5 = 0.5823571654525552;
		btilde6 = -0.45808210592918697;
		btilde7 = 0.015151515151515152;

def euler(x,t,dt,dynamics_func):
	return x + dt * dynamics_func(x,t)

def rk4_(x,t,dt,dynamics_func):
    k1 = dynamics_func(x, t)
    k2 = dynamics_func(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = dynamics_func(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = dynamics_func(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)



def tsit5(x,t,dt,dynamics_func):
	k1 = dynamics_func( x, t);

	tmp = x + dt * (Tsit5.a21 * k1);
	k2 = dynamics_func( tmp, t + Tsit5.c1 * dt );

	tmp = x + dt * ( Tsit5.a31 * k1 + Tsit5.a32 * k2 );
	k3 = dynamics_func( tmp, t + Tsit5.c2 * dt );

	tmp = x + dt * ( Tsit5.a41 * k1 + Tsit5.a42 * k2 + Tsit5.a43 * k3 );
	k4 = dynamics_func( tmp, t + Tsit5.c3 * dt );

	tmp = x + dt * ( Tsit5.a51 * k1 + Tsit5.a52 * k2 + Tsit5.a53 * k3 + Tsit5.a54 * k4 );
	k5 = dynamics_func( tmp, t + Tsit5.c4 * dt );

	tmp = x + dt * ( Tsit5.a61 * k1 + Tsit5.a62 * k2 + Tsit5.a63 * k3 + Tsit5.a64 * k4 + Tsit5.a65 * k5 );
	k6 = dynamics_func( tmp, t + dt );

	x += dt * ( Tsit5.a71 * k1 + Tsit5.a72 * k2 + Tsit5.a73 * k3 + Tsit5.a74 * k4 + Tsit5.a75 * k5 + Tsit5.a76 * k6 );
	return x

def dx_dt(x, t):
	return np.array([
		0.1 * np.cos(x[2]),
		0.1 * np.sin(x[2]),
		0.4
	])

def main():
    
	tf = 10
	# dt = 0.01
	fig, ax = plt.subplots()

	for dt in [0.01, 0.05, 0.1]:

		t = 0

		x_euler = np.array([0.0,0.0,0.0])
		x_rk4 = np.array([0.0,0.0,0.0])
		x_tsit5 = np.array([0.0,0.0,0.0])

		eulers = np.copy(x_euler).reshape(-1,1)
		rk4s = np.copy(x_rk4).reshape(-1,1)
		tsit5s = np.copy(x_tsit5).reshape(-1,1)

		while(t<tf):

			x_euler = euler( x_euler, t, dt, dx_dt )
			x_rk4 = tsit5( x_rk4, t, dt, dx_dt )
			x_tsit5 = tsit5( x_tsit5, t, dt, dx_dt )
			eulers = np.append( eulers, x_euler.reshape(-1,1), axis=1 )
			rk4s = np.append( rk4s, x_rk4.reshape(-1,1), axis=1 )
			tsit5s = np.append( tsit5s, x_tsit5.reshape(-1,1), axis=1 )

			t = t + dt		
		ax.plot(eulers[0,:], eulers[1,:], 'r', alpha = 10*dt)
		ax.plot(rk4s[0,:], rk4s[1,:], 'k--', alpha = 10*dt, linewidth=5.0)
		ax.plot(tsit5s[0,:], tsit5s[1,:], 'g', alpha= 10*dt)
	plt.show()

if __name__ == "__main__":
    main()

