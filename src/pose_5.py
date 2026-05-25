import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')
    best_trace = float('inf')


    for pose_label, pose_5 in pose_options.items():
        for landmark in [2, 1]: 
            # Build fresh copies using gtsam's own methods
            g = gtsam.NonlinearFactorGraph(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)

            # Check marginals before adding landmark measurement
            marginals = gtsam.Marginals(g, result)
            cov_sum_for_selection = (
                marginals.marginalCovariance(L(1)).sum() +
                marginals.marginalCovariance(L(2)).sum()
            )

            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            # Compute return value after adding measurement
            marginals = gtsam.Marginals(g, result)
            trace = np.trace(marginals.marginalCovariance(L(1))) + np.trace(marginals.marginalCovariance(L(2)))
            cov_sum = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()

            if trace < best_trace:
                best_trace = trace
                best_sum = cov_sum
                best_pose = pose_label
                best_landmark = landmark

    return best_pose, best_landmark, best_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    ground_truth = {
        X(1): gtsam.Pose2(0.0, 0.0, 0.0),
        X(2): gtsam.Pose2(2.0, 0.0, 0.0),
        X(3): gtsam.Pose2(4.0, 0.0, 0.0),
    }

    for pose_label, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = gtsam.NonlinearFactorGraph(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            list_of_errors = []
            for key, gt in ground_truth.items():
                estimated = result.atPose2(key)
                dx = estimated.x() - gt.x()
                dy = estimated.y() - gt.y()
                list_of_errors.append(np.sqrt(dx**2 + dy**2))

            sum_of_errors = sum(list_of_errors)
            print(f"pose={pose_label}, landmark={landmark}, sum_of_errors={sum_of_errors}")

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_label
                best_landmark = landmark

    return best_pose, best_landmark, best_sum