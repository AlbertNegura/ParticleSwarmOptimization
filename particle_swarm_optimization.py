import math

import numpy as np
import configparser
from functools import partial
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d



args = []
kwargs = {}

particle_output = False


config = configparser.ConfigParser()
config.read('config.ini')
pso = config['pso']

swarmsize = pso.getint("swarm_size")
iterations = pso.getint("maximum_iterations")
dimensions = pso.getint("dimensions")
rand_init = pso.getboolean("random_initialization")
omega = pso.getfloat("velocity_scaling_factor")
c1 = pso.getfloat("particle_position_weight")
c2 = pso.getfloat("swarm_position_weight")
T1 = pso.getfloat("step_convergence_threshold")
T2 = pso.getfloat("value_convergence_threshold")
CONVERGENCE = pso.getboolean("converge_early")
DEBUG = pso.getboolean("DEBUG")
PROCESSES = pso.getint("PROCESSES")

function = config['functions'].getint("function_selection")

x_hist = np.zeros((iterations,swarmsize,dimensions))
v_hist = np.zeros((iterations,swarmsize,dimensions))
avg_cost_function = np.zeros((iterations))
min_cost_function = np.zeros((iterations))

lower_bounds = [0, 0]
upper_bounds = [10,10]

if function == 0:
    lower_bounds = [0, 0]
    upper_bounds = [10, 10]
elif function == 1:
    lower_bounds = [-4.5, -4.5]
    upper_bounds = [4.5, 4.5]
elif function == 2:
    lower_bounds = [-2*math.pi, -2*math.pi]
    upper_bounds = [2*math.pi, 2*math.pi]
elif function == 3:
    lower_bounds = [-5.2, -5.2]
    upper_bounds = [5.2, 5.2]


def _obj_wrapper(func, args, kwargs, x):
    return func(x,*args,**kwargs)

def error(x):
    x1 = x[0]
    x2 = x[1]
    if function == 0:
        return - (np.sqrt(x1)*np.sin(x1)*np.sqrt(x2)*np.sin(x2))
    elif function == 1:
        return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
    elif function == 2:
        return np.sin(x1)*np.exp((1-np.cos(x2))**2) + np.cos(x2)*np.exp((1-np.sin(x1))**2)
    elif function == 3:
        return -(1+np.cos(12*np.sqrt(x1**2+x2**2)))/(0.5*(x1**2+x2**2)+2)
    else:
        return - (np.sqrt(x1)*np.sin(x1)*np.sqrt(x2)*np.sin(x2))

def function_of(x,y):
    return error([x,y])

def error_plot(values):
    z = np.zeros(values.shape[0])
    for i in range(values.shape[0]):
        val = values[i]
        z[i] = error(val)

    return z

def plot_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(np.min(lower_bounds), np.max(upper_bounds), 0.05)
    y = np.arange(np.min(lower_bounds), np.max(upper_bounds), 0.05)
    X, Y = np.meshgrid(x,y)
    zs = np.array(function_of(np.ravel(X),np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')
    plt.title("3D Plot of Objective Function")
    plt.show()

def plot_contour():
    fig, ax = plt.subplots()

    if np.max(upper_bounds) > 0 and np.min(lower_bounds) < 0:
        x = np.arange(np.min(lower_bounds)*2, np.max(upper_bounds)*2, 0.05)
        y = np.arange(np.min(lower_bounds)*2, np.max(upper_bounds)*2, 0.05)
    elif np.min(lower_bounds) < 0 and np.max(upper_bounds) < 0:
        x = np.arange(np.min(lower_bounds), 0-np.max(upper_bounds), 0.05)
        y = np.arange(np.min(lower_bounds), 0-np.max(upper_bounds), 0.05)
    elif np.min(lower_bounds) > 0 and np.max(upper_bounds) > 0:
        x = np.arange(abs(np.min(lower_bounds))+np.min(lower_bounds), 2*np.max(upper_bounds), 0.05)
        y = np.arange(abs(np.min(lower_bounds))+np.min(lower_bounds), 2*np.max(upper_bounds), 0.05)
    else:
        x = np.arange(2*np.min(lower_bounds), abs(np.max(upper_bounds))+np.max(upper_bounds), 0.05)
        y = np.arange(2*np.min(lower_bounds), abs(np.max(upper_bounds))+np.max(upper_bounds), 0.05)

    X, Y = np.meshgrid(x,y)
    zs = np.array(function_of(np.ravel(X),np.ravel(Y)))
    Z = zs.reshape(X.shape)

    CS = ax.contour(X,Y,Z, cmap='viridis', edgecolor='none')
    ax.clabel(CS, inline=1, fontsize=10)
    plt.title("2D Contour Plot of Objective Function")
    plt.show()

def pso():
    global mp_pool
    global swarmsize
    global x_hist,v_hist,min_cost_function,avg_cost_function
    assert dimensions > 1, 'Must provide at least 2 dimensions'
    assert dimensions == 2, 'Currently only supports 2 dimensions max'
    lb = np.array(lower_bounds[:dimensions])
    ub = np.array(upper_bounds[:dimensions])
    assert np.all(ub>lb), 'All upper bound values must be greater than the corresponding lower bound values'

    upperV = np.abs(ub-lb)
    lowerV = -upperV

    #plot_figure()
    #plot_contour()

    objective = partial(_obj_wrapper, error, args, kwargs)

    if PROCESSES > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(PROCESSES)

    #initialize a few arrays
    S = swarmsize
    D = dimensions
    x = np.random.rand(S,D)
    v = np.zeros_like(x)
    p = np.zeros_like(x)
    fx = np.zeros(S)
    fp = np.ones(S)*np.inf
    g = []
    fg = np.inf

    # initialize particles
    x = lb + x*(ub-lb)

    #calculate objectives for each particles
    if PROCESSES>1:
        fx = np.array(mp_pool.map(objective,x))
    else:
        for i in range(S):
            fx[i] = objective(x[i,:])

    i_update = fx<fp
    p[i_update,:] = x[i_update,:].copy()
    fp[i_update] = fx[i_update]

    i_min = np.argmin(fp)
    if fp[i_min]<fg:
        fg = fp[i_min]
        g = p[i_min,:].copy()
    else:
        g = x[0,:].copy()

    v=lowerV + np.random.rand(S,D)*(upperV-lowerV)

    it = 1
    while it <= iterations:
        x_hist[it-1] = np.array(x)
        v_hist[it-1] = np.array(v)
        rp = np.random.uniform(size=(S,D))
        rg = np.random.uniform(size=(S,D))

        v = omega*v + c1*rp*(p-x) + c2*rg*(g-x)

        x = x + v

        lower_mask = x<lb
        upper_mask = x>ub

        x = x*(~np.logical_or(lower_mask, upper_mask)) + lb*lower_mask + ub*upper_mask

        if PROCESSES>1:
            fx = np.array(mp_pool.map(objective,x))
        else:
            for i in range(S):
                fx[i] = objective(x[i,:])


        i_update = fx<fp
        p[i_update,:] = x[i_update,:].copy()
        fp[i_update] = fx[i_update]

        i_min = np.argmin(fp)
        min_cost_function[it-1]=fp[i_min]
        avg_cost_function[it-1]=np.average(fp)
        if fp[i_min] < fg:
            if DEBUG:
                print('New best for swarm at iteration {:}: {:} {:}' \
                      .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min) ** 2))

            if CONVERGENCE and np.abs(fg - fp[i_min]) <= T2:
                print('Stopping search: Swarm best objective change less than {:}' \
                      .format(T2))
                x_hist=x_hist[:it]
                v_hist=v_hist[:it]
                min_cost_function=min_cost_function[:it]
                avg_cost_function=avg_cost_function[:it]
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            elif CONVERGENCE and stepsize <= T1:
                print('Stopping search: Swarm best position change less than {:}' \
                      .format(T1))
                x_hist=x_hist[:it]
                v_hist=v_hist[:it]
                min_cost_function=min_cost_function[:it]
                avg_cost_function=avg_cost_function[:it]
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if DEBUG:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(iterations))

    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg

def animate2D(data_used, label):
    global ax1, data, line, stop, ani
    data = data_used.copy()
    stop = np.size(data)
    indices = np.linspace(0,stop,stop-1)
    fig = plt.figure()
    ax1 = plt.axes(xlim=[0, stop],ylim=[np.min(data), np.max(data)])
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(label + ' Cost Function')
    line, = ax1.plot([],[],lw=3)
    ani = animation.FuncAnimation(fig, animate, np.arange(0,stop), fargs=[indices, data, line], interval=20, blit=True)
    plt.show()

def animate(i, x, y, line):
    if(i >= stop-1):
        ani.event_source.stop()
    line.set_data(x[:i],y[:i])
    line.axes.axis([0, np.size(data),np.min(data), np.max(data)])
    return line,

scale_factor = np.abs((np.max(upper_bounds)-np.min(lower_bounds)))

def animate_contour(positions, velocities):
    global ax2, xs, vs, stop, ani, fig, contour_vectors
    xs = positions.copy()
    vs = velocities.copy()

    fig = plt.figure()
    stop = xs.shape[0]
    ax2 = plt.axes(xlim=[np.min(lower_bounds), np.max(upper_bounds)],ylim=[np.min(lower_bounds), np.max(upper_bounds)])

    if np.max(upper_bounds) > 0 and np.min(lower_bounds) < 0:
        x = np.arange(np.min(lower_bounds)*2, np.max(upper_bounds)*2, 0.05)
        y = np.arange(np.min(lower_bounds)*2, np.max(upper_bounds)*2, 0.05)
    elif np.min(lower_bounds) < 0 and np.max(upper_bounds) < 0:
        x = np.arange(np.min(lower_bounds), 0-np.max(upper_bounds), 0.05)
        y = np.arange(np.min(lower_bounds), 0-np.max(upper_bounds), 0.05)
    elif np.min(lower_bounds) > 0 and np.max(upper_bounds) > 0:
        x = np.arange(abs(np.min(lower_bounds))+np.min(lower_bounds), 2*np.max(upper_bounds), 0.05)
        y = np.arange(abs(np.min(lower_bounds))+np.min(lower_bounds), 2*np.max(upper_bounds), 0.05)
    else:
        x = np.arange(2*np.min(lower_bounds), abs(np.max(upper_bounds))+np.max(upper_bounds), 0.05)
        y = np.arange(2*np.min(lower_bounds), abs(np.max(upper_bounds))+np.max(upper_bounds), 0.05)

    X, Y = np.meshgrid(x,y)
    zs = np.array(function_of(np.ravel(X),np.ravel(Y)))
    Z = zs.reshape(X.shape)

    CS = ax2.contour(X,Y,Z, cmap='viridis')
    plt.title("2D Contour Plot of Objective Function")

    Xs = xs[0]
    x_Xs = Xs[:,0]
    y_Xs = Xs[:,1]
    Vs = vs[0]
    x_Vs = Vs[:,0]
    y_Vs = Vs[:,1]
    scatters = ax2.scatter(x_Xs,y_Xs,c="black", marker="o")
    contour_vectors = ax2.quiver(x_Xs,y_Xs,x_Vs,y_Vs, scale=50)
    ani = animation.FuncAnimation(fig, animate2, np.arange(0,stop-2), fargs=[scatters], interval=50, blit=False, repeat=True)
    plt.show()

def animate2(i, scatters):
    global contour_vectors
    plot_data = xs[i]
    v_plot_data = vs[i]

    contour_vectors.remove()
    scatters.set_offsets(plot_data)
    contour_vectors = ax2.quiver(plot_data[:,0],plot_data[:,1],v_plot_data[:,0],v_plot_data[:,1],scale=50)
    return (scatters, contour_vectors),

def animate3D(positions, velocities):
    global ax3, xs, vs, stop, ani3, fig3, vectors, scale_factor
    xs = positions.copy()
    vs = velocities.copy()

    fig3 = plt.figure()
    ax3 = axes3d.Axes3D(fig3)
    x = np.arange(np.min(lower_bounds), np.max(upper_bounds), 0.05)
    y = np.arange(np.min(lower_bounds), np.max(upper_bounds), 0.05)
    X, Y = np.meshgrid(x,y)
    zs = np.array(function_of(np.ravel(X),np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax3.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none', alpha=0.2)
    plt.title("3D Plot of Objective Function")

    stop = xs.shape[0]
    scale_factor /= stop
    Xs = xs[0]
    x_Xs = Xs[:,0]
    y_Xs = Xs[:,1]
    z_Xs = error_plot(Xs[:,:])
    Vs = vs[0]
    x_Vs = Vs[:,0]*scale_factor
    y_Vs = Vs[:,1]*scale_factor
    z_Vs = error_plot(Vs[:,:])*scale_factor
    scatters = ax3.scatter(x_Xs,y_Xs,z_Xs,c="black", marker="o")
    vectors = ax3.quiver(x_Xs,y_Xs,z_Xs,x_Vs,y_Vs,z_Vs)
    ani3 = animation.FuncAnimation(fig3, animate3, np.arange(0,stop-2), fargs=[scatters], interval=20, blit=False, repeat=True)
    plt.show()

def animate3(i, scatters):
    global vectors, scale_factor
    plot_data = xs[i]
    v_plot_data = vs[i]

    vectors.remove()
    scatters._offsets3d = (plot_data[:,0],plot_data[:,1],error_plot(plot_data[:]))
    vectors = ax3.quiver(plot_data[:,0],plot_data[:,1],error_plot(plot_data[:]),v_plot_data[:,0]*scale_factor,v_plot_data[:,1]*scale_factor,error_plot(plot_data[:])*scale_factor)
    return (scatters, vectors),


if __name__ == '__main__':
    pso()
    #print(x_hist)
    #print(v_hist)
    #plt.plot(min_cost_function)
    #plt.ylabel("Cost")
    #plt.xlabel("Iterations")
    #plt.title("Min cost function")
    #plt.show()
    #plt.plot(avg_cost_function)
    #plt.ylabel("Cost")
    #plt.xlabel("Iterations")
    #plt.title("Average cost function")
    #plt.show()


    animate2D(min_cost_function, "Min")
    animate2D(avg_cost_function, "Average")
    animate_contour(x_hist,v_hist)
    animate3D(x_hist, v_hist)
