import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk

# Function to create a skew-symmetric matrix
def skew_symmetric(k):
    return np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

# Function to create a rotation matrix using Rodrigues' formula
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    K = skew_symmetric(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

# Function for nonlinear translation influenced by rotation velocity
def adaptive_translation(R, d, lambda_factor):
    return (np.eye(3) - np.exp(-lambda_factor * R)) @ d

# Initialize vectors (joints) and their rotation weights
num_joints = 5
vectors = np.array([[i, 0, 0] for i in range(num_joints)])  # Initial positions
weights = np.linspace(-1, 1, num_joints)  # Rotation weight factors
distance_weights = np.full(num_joints, 1.0)  # Distance constraint weights

# Store initial distances and unit direction vectors between joints
initial_directions = np.diff(vectors, axis=0)
initial_distances = np.linalg.norm(initial_directions, axis=1)
unit_initial_directions = initial_directions / initial_distances[:, np.newaxis]  # Unit vectors

# Default transformation parameters
default_theta_speed = np.pi / 50
default_lambda_factor = 0.5
default_d = np.array([0.05, 0, 0])
rotation_axis = np.array([1, 1, 1]) / np.sqrt(3)

# Live-adjustable values
theta_speed = default_theta_speed
lambda_factor = default_lambda_factor
d = default_d.copy()

# Create Matplotlib figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim([-2, 5])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Frame capture times (seconds)
frame_capture_times = [0, 2.5, 5]
captured_frames = [False, False, False]
frame_filenames = ["start_frame.png", "mid_frame.png", "end_frame.png"]

# Animation update function
def update(frame):
    global theta_speed, lambda_factor, d, weights, distance_weights, captured_frames

    theta = theta_speed * frame
    R_global = rotation_matrix(rotation_axis, theta)

    # Apply transformations
    transformed_vectors = [vectors[0]]
    for i in range(1, num_joints):
        weighted_rotation = rotation_matrix(rotation_axis, weights[i] * theta)
        adaptive_t = adaptive_translation(R_global, d, lambda_factor)
        
        # Compute new position
        transformed_vec = R_global @ (weighted_rotation @ vectors[i]) + adaptive_t

        # Apply distance constraint
        direction = transformed_vec - transformed_vectors[i-1]
        norm_dir = np.linalg.norm(direction)
        unit_direction = direction / norm_dir if norm_dir > 1e-8 else unit_initial_directions[i-1]

        constrained_position = transformed_vectors[i-1] + (1 - distance_weights[i]) * vectors[i] + distance_weights[i] * (unit_direction * initial_distances[i-1])
        transformed_vectors.append(constrained_position)

    transformed_vectors = np.array(transformed_vectors)

    # Clear plot and redraw
    ax.cla()
    ax.set_xlim([-2, 5])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Plot joints
    ax.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], transformed_vectors[:, 2], c='b', marker='o')

    # Plot lines between joints
    for i in range(num_joints - 1):
        ax.plot(transformed_vectors[i:i+2, 0], transformed_vectors[i:i+2, 1], transformed_vectors[i:i+2, 2], 'r')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=20)

# Tkinter window for sliders
root = tk.Tk()
root.title("Parameter Controls")

# Function to update values
def update_values(event=None):
    global theta_speed, lambda_factor, d, weights, distance_weights
    theta_speed = theta_slider.get()
    lambda_factor = lambda_slider.get()
    d[0] = dx_slider.get()
    d[1] = dy_slider.get()
    d[2] = dz_slider.get()
    
    for i in range(num_joints):
        weights[i] = weight_sliders[i].get()
        distance_weights[i] = distance_sliders[i].get()

# Create sliders in Tkinter window
theta_slider = tk.Scale(root, from_=0.01, to=0.2, resolution=0.001, orient="horizontal", label="Theta Speed", command=update_values)
theta_slider.set(default_theta_speed)
theta_slider.pack()

lambda_slider = tk.Scale(root, from_=0.1, to=2.0, resolution=0.01, orient="horizontal", label="Lambda Factor", command=update_values)
lambda_slider.set(default_lambda_factor)
lambda_slider.pack()

dx_slider = tk.Scale(root, from_=-0.1, to=0.1, resolution=0.01, orient="horizontal", label="Translation X", command=update_values)
dx_slider.set(default_d[0])
dx_slider.pack()

dy_slider = tk.Scale(root, from_=-0.1, to=0.1, resolution=0.01, orient="horizontal", label="Translation Y", command=update_values)
dy_slider.set(default_d[1])
dy_slider.pack()

dz_slider = tk.Scale(root, from_=-0.1, to=0.1, resolution=0.01, orient="horizontal", label="Translation Z", command=update_values)
dz_slider.set(default_d[2])
dz_slider.pack()

# Sliders for joint weights and distance constraints
weight_sliders = []
distance_sliders = []
for i in range(num_joints):
    weight_slider = tk.Scale(root, from_=-1, to=1, resolution=0.01, orient="horizontal", label=f"Rot. Weight {i+1}", command=update_values)
    weight_slider.set(weights[i])
    weight_slider.pack()
    weight_sliders.append(weight_slider)

    distance_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal", label=f"Dist. Weight {i+1}", command=update_values)
    distance_slider.set(distance_weights[i])
    distance_slider.pack()
    distance_sliders.append(distance_slider)

# Run Tkinter in a separate thread
import threading
def run_tk():
    root.mainloop()

tk_thread = threading.Thread(target=run_tk, daemon=True)
tk_thread.start()

plt.show()
