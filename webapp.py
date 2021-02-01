# -*- coding: utf-8 -*-

# Run this app with `python webapp.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import numpy as np
import pandas as pd
import random
from holoviews import opts, dim
import holoviews as hv
import panel as pn
from holoviews.streams import Stream
import particle_swarm_optimization as pso
hv.extension('bokeh', logo=False)


def generate_data(swarmsize = 100, iterations = 100, dimensions = 2, rand_init = False, omega = 0.5, c1 = 0.5, c2 = 0.5, T1 = 1e-10, T2 = 1e-10, CONVERGENCE = False, DEBUG = False, PROCESSES = 1, function = 0):
    particle_swarm = pso.PSO(swarmsize = 100, iterations = 100, dimensions = 2, rand_init = False, omega = 0.5, c1 = 0.5, c2 = 0.5, T1 = 1e-10, T2 = 1e-10, CONVERGENCE = False, DEBUG = False, PROCESSES = 1, function = 0)
    particle_swarm.pso()

    return particle_swarm

def to_angle(vector):
    x = vector[0]
    y = vector[1]
    mag = np.sqrt(x**2 + y**2)
    angle = (np.pi/2.) - np.arctan2(x/mag, y/mag)
    return mag, angle

def get_vectorfield_data(pso):
    xs, ys, angles, mags, ids = [], [], [], [], []
    for particle in range(pso.swarmsize):
        xs.append(pso.x_hist[0][particle,0])
        ys.append(pso.x_hist[0][particle,1])
        mag, angle = to_angle(pso.v_hist[0][particle])
        mags.append(mag)
        angles.append(angle)
        ids.append(particle)
    return xs, ys, angles, mags, ids



# Running the server
if __name__ == "__main__":
    pso = pso.PSO()
    pso.pso()

    vect_data = get_vectorfield_data(pso)
    vectorfield = hv.VectorField(vect_data, vdims=['Angle', 'Magnitude', 'Index'])

    # [x, y, id] for all particles
    particles = [np.array([vect_data[0], vect_data[1], vect_data[4]]) for i in range(pso.swarmsize)]
    points = hv.Points(particles, vdims=['Index'])
    layout = vectorfield * points
    layout.opts(
        opts.VectorField(color='Index', cmap='tab20c', magnitude=dim('Magnitude').norm() * 10, pivot='tail'),
        opts.Points(color='Index', cmap='tab20c', size=5)
    )
    pn.Column(layout.opts(width=500, height=500))