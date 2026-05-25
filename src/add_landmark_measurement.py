import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    pose4 = result.atPose2(X(4))
    landmark2 = result.atPoint2(L(2))

    # Vector from pose to landmark in world frame
    dx = landmark2[0] - pose4.x()
    dy = landmark2[1] - pose4.y()

    # Distance
    distance = math.sqrt(dx**2 + dy**2)

    # Bearing: angle to landmark in world frame, minus robot's heading
    angle_world = math.atan2(dy, dx)
    bearing = angle_world - pose4.theta()

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2(bearing), distance, MEASUREMENT_NOISE))
    return graph