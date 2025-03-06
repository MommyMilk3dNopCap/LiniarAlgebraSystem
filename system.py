import numpy as np
import matplotlib.pyplot as plt
import threading
import time

# Function to create a rotation matrix given an axis and an angle
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])

# Function for nonlinear translation influenced by rotation velocity
def adaptive_translation(R, d, lambda_factor):
    return (np.eye(3) - np.exp(-lambda_factor * R)) @ d

# Initialize vectors (joints) and their rotation weights
num_joints = 5
vectors = np.array([[i, 0, 0] for i in range(num_joints)])  # Initial positions
weights = np.linspace(-1, 1, num_joints)  # Weight factors for each joint

# Default transformation parameters
default_theta_speed = np.pi / 50  # Default rotation speed
default_lambda_factor = 0.5  # Default lambda factor
default_d = np.array([0.05, 0, 0])  # Default translation step
loop_time = 10  # Reset animation every 10 seconds

# Current values (modifiable)
theta_speed = default_theta_speed
lambda_factor = default_lambda_factor
d = default_d.copy()

# Function to handle real-time user input
def user_input_loop():
    global theta_speed, lambda_factor, d
    while True:
        try:
            user_input = input("\nEnter 'theta speed, lambda factor, dx dy dz' (or press Enter to keep current values, or type 'x' to reset): ")
            
            if user_input.strip() == "x":
                theta_speed = default_theta_speed
                lambda_factor = default_lambda_factor
                d = default_d.copy()
                print("\n🔄 Reset to default values!")
                continue

            if user_input.strip() == "":
                continue  # Keep existing values

            parts = user_input.split(",")
            if len(parts) >= 2:
                theta_speed = float(parts[0].strip())  # Adjust rotation speed
                lambda_factor = float(parts[1].strip())  # Adjust lambda factor
            
            if len(parts) == 5:
                d = np.array([float(parts[2].strip()), float(parts[3].strip()), float(parts[4].strip())])  # Adjust translation vector

            print(f"\n🔹 Current Values → Theta Speed: {theta_speed:.4f}, Lambda Factor: {lambda_factor:.4f}, Translation Step: {d}")

        except ValueError:
            print("❌ Invalid input. Please enter numbers in the correct format.")

# Start user input loop in a separate thread
input_thread = threading.Thread(target=user_input_loop, daemon=True)
input_thread.start()

# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 5])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Main simulation loop
while True:
    input("\n🔄 Press Enter to start simulation...")  # Wait for user input before restarting

    start_time = time.time()  # Track start time
    time_steps = 100
    trajectory = []

    for t in range(time_steps):
        if time.time() - start_time > loop_time:
            break  # Restart animation after 10 seconds

        theta = theta_speed * t  # Rotation angle changing over time
        R = rotation_matrix(np.array([0, 0, 1]), theta)  # Rotation matrix

        # Apply transformations to each joint
        transformed_vectors = []
        for i, vec in enumerate(vectors):
            weighted_rotation = rotation_matrix(np.array([0, 0, 1]), weights[i] * theta)
            adaptive_t = adaptive_translation(R, d, lambda_factor)
            transformed_vec = weighted_rotation @ vec + adaptive_t  # Apply transformation
            transformed_vectors.append(transformed_vec)
        
        trajectory.append(np.array(transformed_vectors))

        # Clear plot and redraw
        ax.cla()
        ax.set_xlim([-2, 5])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_title(f"Step {t}")

        # Plot joints
        ax.scatter(trajectory[t][:, 0], trajectory[t][:, 1], trajectory[t][:, 2], c='b', marker='o')

        # Plot lines between joints
        for i in range(num_joints - 1):
            ax.plot(
                [trajectory[t][i, 0], trajectory[t][i+1, 0]],
                [trajectory[t][i, 1], trajectory[t][i+1, 1]],
                [trajectory[t][i, 2], trajectory[t][i+1, 2]],
                'r'
            )

        plt.pause(0.01)  # Pause to create animation effect

    print("\n🔄 Simulation complete. Waiting for user input before restarting...")
