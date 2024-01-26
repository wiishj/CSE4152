import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
x_ref = 10  # Target position
x_0 = 0  # Initial position
dt = 0.1  # Time step for simulation
total_time = 100  # Total simulation time in seconds
n = int(total_time / dt)  # Number of simulation steps

# PID control gains
Kp = 0.5  # Proportional gain
Kd = 0.05  # Derivative gain

# Initialize the arrays for storing data
time = np.linspace(0, total_time, n)
position_bang = np.zeros(n)
position_p = np.zeros(n)
position_pd = np.zeros(n)
velocity_bang = np.zeros(n)
velocity_p = np.zeros(n)
velocity_pd = np.zeros(n)
acceleration_bang = np.zeros(n)
acceleration_p = np.zeros(n)
acceleration_pd = np.zeros(n)
error_p = np.zeros(n)
error_pd = np.zeros(n)

# Simulation loop
for i in range(1, n):
    error = x_ref - position_bang[i - 1]

    if error > 0:
        acceleration_bang[i] = 1
    else:
        acceleration_bang[i] = -1
    velocity_bang[i] = velocity_bang[i - 1] + acceleration_bang[i] * dt
    position_bang[i] = position_bang[i - 1] + velocity_bang[i] * dt

    error_p[i] = x_ref - position_p[i - 1]
    derivative_of_error = (error_p[i] - error_p[i - 1]) / dt if i > 1 else 0
    acceleration_p[i] = Kp * error_p[i]
    velocity_p[i] = velocity_p[i - 1] + acceleration_p[i] * dt
    position_p[i] = position_p[i - 1] + velocity_p[i] * dt

    error_pd[i] = x_ref - position_pd[i - 1]
    derivative_of_error = (error_pd[i] - error_pd[i - 1]) / dt if i > 1 else 0
    acceleration_pd[i] = Kp * error_pd[i] + Kd * derivative_of_error
    velocity_pd[i] = velocity_pd[i - 1] + acceleration_pd[i] * dt
    position_pd[i] = position_pd[i - 1] + velocity_pd[i] * dt

# Plot the results
plt.figure(figsize=(8, 10))

plt.subplot(6, 1, 1)
plt.plot(time, position_bang, label="Position")
plt.axhline(y=x_ref, color="r", linestyle="--", label="Target Position")
plt.title("Bang_Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()

plt.subplot(6, 1, 2)
plt.plot(time, acceleration_bang, label="Acceleration")
plt.title("Bang_Acceleration Command")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(time, position_p, label="Position")
plt.axhline(y=x_ref, color="r", linestyle="--", label="Target Position")
plt.title("P_Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()

plt.subplot(6, 1, 4)
plt.plot(time, acceleration_p, label="Acceleration")
plt.title("P_Acceleration Command")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.legend()

plt.subplot(6, 1, 5)
plt.plot(time, position_pd, label="Position")
plt.axhline(y=x_ref, color="r", linestyle="--", label="Target Position")
plt.title("PD_Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()

plt.subplot(6, 1, 6)
plt.plot(time, acceleration_pd, label="Acceleration")
plt.title("PD_Acceleration Command")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.legend()

plt.tight_layout()
plt.show()
