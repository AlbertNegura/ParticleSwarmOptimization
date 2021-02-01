import math

import numpy as np
import configparser
from functools import partial
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d


class PSO():

    args = []
    kwargs = {}

    particle_output = False

    swarmsize = None
    iterations = None
    rand_init = None
    omega = None
    c1 = None
    c2 = None
    T1 = None
    T2 = None
    CONVERGENCE = None
    DEBUG = None
    PROCESSES = None

    function = None

    x_hist = None
    v_hist = None
    avg_cost_function = None
    min_cost_function = None

    lower_bounds = None
    upper_bounds = None

    scale_factor = None

    def __init__(self, mode='regular', swarmsize = 100, iterations = 100, rand_init = False, omega = 0.5, c1 = 0.5, c2 = 0.5, T1 = 1e-10, T2 = 1e-10, CONVERGENCE = False, DEBUG = False, PROCESSES = 1, function = 0):
        if mode != 'regular':
            self.swarmsize = swarmsize
            self.iterations = iterations
            self.rand_init = rand_init
            self.omega = omega
            self.c1 = c1
            self.c2 = c2
            self.T1 = T1
            self.T2 = T2
            self.CONVERGENCE = CONVERGENCE
            self.DEBUG = DEBUG
            self.PROCESSES = PROCESSES

            self.function = function
        else:

            config = configparser.ConfigParser()
            config.read('config.ini')
            pso = config['pso']

            self.swarmsize = pso.getint("swarm_size")
            self.iterations = pso.getint("maximum_iterations")
            self.rand_init = pso.getboolean("random_initialization")
            self.omega = pso.getfloat("velocity_scaling_factor")
            self.c1 = pso.getfloat("particle_position_weight")
            self.c2 = pso.getfloat("swarm_position_weight")
            self.T1 = pso.getfloat("step_convergence_threshold")
            self.T2 = pso.getfloat("value_convergence_threshold")
            self.CONVERGENCE = pso.getboolean("converge_early")
            self.DEBUG = pso.getboolean("DEBUG")
            self.PROCESSES = pso.getint("PROCESSES")

            self.function = config['functions'].getint("function_selection")

        self.x_hist = np.zeros((self.iterations,self.swarmsize,2))
        self.v_hist = np.zeros((self.iterations,self.swarmsize,2))
        self.avg_cost_function = np.zeros((self.iterations))
        self.min_cost_function = np.zeros((self.iterations))

        self.lower_bounds = [0, 0]
        self.upper_bounds = [10,10]

        if self.function == 0:
            self.lower_bounds = [0, 0]
            self.upper_bounds = [10, 10]
        elif self.function == 1:
            self.lower_bounds = [-4.5, -4.5]
            self.upper_bounds = [4.5, 4.5]
        elif self.function == 2:
            self.lower_bounds = [-2*math.pi, -2*math.pi]
            self.upper_bounds = [2*math.pi, 2*math.pi]
        elif self.function == 3:
            self.lower_bounds = [-5.2, -5.2]
            self.upper_bounds = [5.2, 5.2]

        self.scale_factor = np.abs((np.max(self.upper_bounds)-np.min(self.lower_bounds)))*2


    def _obj_wrapper(self, func, args, kwargs, x):
        return func(x,*args,**kwargs)

    def error(self, x):
        x1 = x[0]
        x2 = x[1]
        if self.function == 0:
            return -(np.sqrt(np.abs(x1))*np.sin(x1)*np.sqrt(np.abs(x2))*np.sin(x2))
        elif self.function == 1:
            return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
        elif self.function == 2:
            return np.sin(x1)*np.exp((1-np.cos(x2))**2) + np.cos(x2)*np.exp((1-np.sin(x1))**2)
        elif self.function == 3:
            return -(1+np.cos(12*np.sqrt(x1**2+x2**2)))/(0.5*(x1**2+x2**2)+2)
        else:
            return - (np.sqrt(x1)*np.sin(x1)*np.sqrt(x2)*np.sin(x2))

    def function_of(self, x,y):
        return self.error([x,y])

    def error_plot(self, values):
        z = np.zeros(values.shape[0])
        for i in range(values.shape[0]):
            val = values[i]
            z[i] = self.error(val)

        return z

    def plot_figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(np.min(self.lower_bounds), np.max(self.upper_bounds), 0.05)
        y = np.arange(np.min(self.lower_bounds), np.max(self.upper_bounds), 0.05)
        X, Y = np.meshgrid(x,y)
        zs = np.array(self.function_of(np.ravel(X),np.ravel(Y)))
        Z = zs.reshape(X.shape)

        ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')
        plt.title("3D Plot of Objective Function")
        plt.show()

    def plot_contour(self):
        fig, ax = plt.subplots()

        if np.max(self.upper_bounds) > 0 and np.min(self.lower_bounds) < 0:
            x = np.arange(np.min(self.lower_bounds)*2, np.max(self.upper_bounds)*2, 0.05)
            y = np.arange(np.min(self.lower_bounds)*2, np.max(self.upper_bounds)*2, 0.05)
        elif np.min(self.lower_bounds) < 0 and np.max(self.upper_bounds) < 0:
            x = np.arange(np.min(self.lower_bounds), 0-np.max(self.upper_bounds), 0.05)
            y = np.arange(np.min(self.lower_bounds), 0-np.max(self.upper_bounds), 0.05)
        elif np.min(self.lower_bounds) > 0 and np.max(self.upper_bounds) > 0:
            x = np.arange(abs(np.min(self.lower_bounds))+np.min(self.lower_bounds), 2*np.max(self.upper_bounds), 0.05)
            y = np.arange(abs(np.min(self.lower_bounds))+np.min(self.lower_bounds), 2*np.max(self.upper_bounds), 0.05)
        else:
            x = np.arange(2*np.min(self.lower_bounds), abs(np.max(self.upper_bounds))+np.max(self.upper_bounds), 0.05)
            y = np.arange(2*np.min(self.lower_bounds), abs(np.max(self.upper_bounds))+np.max(self.upper_bounds), 0.05)

        X, Y = np.meshgrid(x,y)
        zs = np.array(self.function_of(np.ravel(X),np.ravel(Y)))
        Z = zs.reshape(X.shape)

        CS = ax.contour(X,Y,Z, cmap='viridis', edgecolor='none')
        ax.clabel(CS, inline=1, fontsize=10)
        plt.title("2D Contour Plot of Objective Function")
        plt.show()

    def pso(self):
        #global mp_pool
        #global swarmsize
        #global x_hist,v_hist,min_cost_function,avg_cost_function
        lb = np.array(self.lower_bounds.copy())
        ub = np.array(self.upper_bounds.copy())
        assert np.all(ub>lb), 'All upper bound values must be greater than the corresponding lower bound values'

        upperV = np.abs(ub-lb)
        lowerV = -upperV

        #plot_figure()
        #plot_contour()

        objective = partial(self._obj_wrapper, self.error, self.args, self.kwargs)

        if self.PROCESSES > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(self.PROCESSES)

        #initialize a few arrays
        S = self.swarmsize
        x = np.random.rand(S,2)
        v = np.zeros_like(x)
        p = np.zeros_like(x)
        fx = np.zeros(S)
        fp = np.ones(S)*np.inf
        g = []
        fg = np.inf

        # initialize particles
        x = lb + x*(ub-lb)

        #calculate objectives for each particles
        if self.PROCESSES>1:
            fx = np.array(self.mp_pool.map(objective,x))
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

        v=lowerV + np.random.rand(S,2)*(upperV-lowerV)

        it = 1
        while it <= self.iterations:
            self.x_hist[it-1] = np.array(x)
            self.v_hist[it-1] = np.array(v)
            rp = np.random.uniform(size=(S,2))
            rg = np.random.uniform(size=(S,2))

            v = self.omega*v + self.c1*rp*(p-x) + self.c2*rg*(g-x)

            x = x + v

            lower_mask = x<lb
            upper_mask = x>ub

            x = x*(~np.logical_or(lower_mask, upper_mask)) + lb*lower_mask + ub*upper_mask

            if self.PROCESSES>1:
                fx = np.array(mp_pool.map(objective,x))
            else:
                for i in range(S):
                    fx[i] = objective(x[i,:])


            i_update = fx<fp
            p[i_update,:] = x[i_update,:].copy()
            fp[i_update] = fx[i_update]

            i_min = np.argmin(fp)
            self.min_cost_function[it-1]=fp[i_min]
            self.avg_cost_function[it-1]=np.average(fp)
            if fp[i_min] < fg:
                if self.DEBUG:
                    print('New best for swarm at iteration {:}: {:} {:}' \
                          .format(it, p[i_min, :], fp[i_min]))

                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min) ** 2))

                if self.CONVERGENCE and np.abs(fg - fp[i_min]) <= self.T2:
                    print('Stopping search: Swarm best objective change less than {:}' \
                          .format(self.T2))
                    x_hist=self.x_hist[:it]
                    v_hist=self.v_hist[:it]
                    min_cost_function=self.min_cost_function[:it]
                    avg_cost_function=self.avg_cost_function[:it]
                    if self.particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif self.CONVERGENCE and stepsize <= self.T1:
                    print('Stopping search: Swarm best position change less than {:}' \
                          .format(self.T1))
                    x_hist=self.x_hist[:it]
                    v_hist=self.v_hist[:it]
                    min_cost_function=self.min_cost_function[:it]
                    avg_cost_function=self.avg_cost_function[:it]
                    if self.particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]

            if self.DEBUG:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1

        print('Stopping search: maximum iterations reached --> {:}'.format(self.iterations))

        if self.particle_output:
            return g, fg, p, fp
        else:
            return g, fg

    def animate2D(self, data_used, label):
        #global ax1, data, line, stop, ani
        self.data = data_used.copy()
        self.stop = np.size(self.data)
        indices = np.linspace(0,self.stop,self.stop-1)
        fig = plt.figure()
        ax1 = plt.axes(xlim=[0, self.stop],ylim=[np.min(self.data), np.max(self.data)])
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title(label + ' Cost Function')
        self.line, = ax1.plot([],[],lw=3)
        self.ani = animation.FuncAnimation(fig, self.animate, np.arange(0,self.stop), fargs=[indices, self.data, self.line], interval=20, blit=True)
        plt.show()

    def animate(self, i, x, y, line):
        if(i >= self.stop-1):
            self.ani.event_source.stop()
        line.set_data(x[:i],y[:i])
        line.axes.axis([0, np.size(self.data),np.min(self.data), np.max(self.data)])
        return line,

    def rand_cmap(self, nlabels):
        from matplotlib.colors import LinearSegmentedColormap
        import colorsys
        import numpy as np

        # Generate color map for bright colors, based on hsv
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.2, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        return random_colormap

    def animate_contour(self, positions, velocities):
        #global ax2, xs, vs, stop, ani, fig, contour_vectors
        self.xs = positions.copy()
        self.vs = velocities.copy()

        fig = plt.figure()
        self.stop = self.xs.shape[0]
        self.ax2 = plt.axes(xlim=[np.min(self.lower_bounds), np.max(self.upper_bounds)],ylim=[np.min(self.lower_bounds), np.max(self.upper_bounds)])

        if np.max(self.upper_bounds) > 0 and np.min(self.lower_bounds) < 0:
            x = np.arange(np.min(self.lower_bounds)*2, np.max(self.upper_bounds)*2, 0.05)
            y = np.arange(np.min(self.lower_bounds)*2, np.max(self.upper_bounds)*2, 0.05)
        elif np.min(self.lower_bounds) < 0 and np.max(self.upper_bounds) < 0:
            x = np.arange(np.min(self.lower_bounds), 0-np.max(self.upper_bounds), 0.05)
            y = np.arange(np.min(self.lower_bounds), 0-np.max(self.upper_bounds), 0.05)
        elif np.min(self.lower_bounds) > 0 and np.max(self.upper_bounds) > 0:
            x = np.arange(abs(np.min(self.lower_bounds))+np.min(self.lower_bounds), 2*np.max(self.upper_bounds), 0.05)
            y = np.arange(abs(np.min(self.lower_bounds))+np.min(self.lower_bounds), 2*np.max(self.upper_bounds), 0.05)
        else:
            x = np.arange(2*np.min(self.lower_bounds), abs(np.max(self.upper_bounds))+np.max(self.upper_bounds), 0.05)
            y = np.arange(2*np.min(self.lower_bounds), abs(np.max(self.upper_bounds))+np.max(self.upper_bounds), 0.05)

        X, Y = np.meshgrid(x,y)
        zs = np.array(self.function_of(np.ravel(X),np.ravel(Y)))
        Z = zs.reshape(X.shape)

        self.CS = self.ax2.contour(X,Y,Z, cmap='viridis')
        plt.title("2D Contour Plot of Objective Function")

        Xs = self.xs[0]
        x_Xs = Xs[:,0]
        y_Xs = Xs[:,1]
        Vs = self.vs[0]
        x_Vs = Vs[:,0]
        y_Vs = Vs[:,1]


        cmap = self.rand_cmap(self.swarmsize)
        scatters = self.ax2.scatter(x_Xs,y_Xs,c=[i for i in range(self.swarmsize)], cmap=cmap, marker="o",vmin = 0, vmax = self.swarmsize)
        self.contour_vectors = self.ax2.quiver(x_Xs,y_Xs,x_Vs,y_Vs, scale=50)
        lines = []
        for i in range(self.swarmsize):
            line = self.ax2.plot(self.xs[0, i, 0], self.xs[0, i, 1], c=cmap(i), alpha=0.2)
            lines.append(line)
        self.ani2 = animation.FuncAnimation(fig, self.animate2, np.arange(0,self.stop-2), fargs=[scatters, lines], interval=50, blit=False, repeat=True)
        plt.show()

    def animate2(self, i, scatters, lines):
        #global contour_vectors
        plot_data = self.xs[i]
        v_plot_data = self.vs[i]

        self.contour_vectors.remove()
        scatters.set_offsets(plot_data)
        if i > 1:
            for lnum,line in enumerate(lines):
                data = self.xs[:i,lnum,:]
                line[0].set_data(data[:,0],data[:,1])
        self.contour_vectors = self.ax2.quiver(plot_data[:,0],plot_data[:,1],v_plot_data[:,0],v_plot_data[:,1],scale=50)
        return (scatters, self.contour_vectors),

    def animate3D(self, positions, velocities):
        #global ax3, xs, vs, stop, ani3, fig3, vectors, scale_factor
        self.xs = positions.copy()
        self.vs = velocities.copy()

        fig3 = plt.figure()
        self.ax3 = axes3d.Axes3D(fig3)
        x = np.arange(np.min(self.lower_bounds), np.max(self.upper_bounds), 0.05)
        y = np.arange(np.min(self.lower_bounds), np.max(self.upper_bounds), 0.05)
        X, Y = np.meshgrid(x,y)
        zs = np.array(self.function_of(np.ravel(X),np.ravel(Y)))
        Z = zs.reshape(X.shape)

        self.ax3.plot_surface(X,Y,Z, cmap='gray', edgecolor='none', alpha=0.2)
        plt.title("3D Plot of Objective Function")

        self.stop = self.xs.shape[0]
        self.scale_factor /= self.stop
        Xs = self.xs[0]
        x_Xs = Xs[:,0]
        y_Xs = Xs[:,1]
        z_Xs = self.error_plot(Xs[:,:])
        Vs = self.vs[0]
        x_Vs = Vs[:,0]*self.scale_factor
        y_Vs = Vs[:,1]*self.scale_factor
        z_Vs = self.error_plot(Vs[:,:])*self.scale_factor

        cmap = self.rand_cmap(self.swarmsize)
        scatters = self.ax3.scatter(x_Xs,y_Xs,z_Xs,c=[i for i in range(self.swarmsize)], cmap=cmap, marker="o",vmin = 0, vmax = self.swarmsize)
        self.vectors = self.ax3.quiver(x_Xs,y_Xs,z_Xs,x_Vs,y_Vs,z_Vs)
        lines = []
        for i in range(self.swarmsize):
            line = self.ax3.plot(self.xs[0, i, 0], self.xs[0, i, 1], z_Xs[i], c=cmap(i), alpha=0.5)
            lines.append(line)

        self.ani3 = animation.FuncAnimation(fig3, self.animate3, frames=500, fargs=[scatters, lines], interval=100, blit=False, repeat=True)
        plt.show()

    def animate3(self, i, scatters,lines):
        #global vectors, scale_factor
        if i < self.iterations:
            plot_data = self.xs[i]
            v_plot_data = self.vs[i]
            z_Xs = self.error_plot(plot_data[:])

            self.vectors.remove()
            if i > 1:
                for lnum,line in enumerate(lines):
                    data = self.xs[:i,lnum,:]
                    function_data = self.error_plot(data)
                    line[0].set_data(data[:,0],data[:,1])
                    line[0].set_3d_properties(function_data)
            scatters._offsets3d = (plot_data[:,0],plot_data[:,1],z_Xs)
            self.vectors = self.ax3.quiver(plot_data[:,0],plot_data[:,1],z_Xs,v_plot_data[:,0]*self.scale_factor,v_plot_data[:,1]*self.scale_factor,z_Xs*self.scale_factor)
        return (scatters, self.vectors),


if __name__ == '__main__':
    pso = PSO()
    pso.pso()
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


    pso.animate2D(pso.min_cost_function, "Min")
    pso.animate2D(pso.avg_cost_function, "Average")
    pso.animate_contour(pso.x_hist,pso.v_hist)
    pso.animate3D(pso.x_hist, pso.v_hist)
