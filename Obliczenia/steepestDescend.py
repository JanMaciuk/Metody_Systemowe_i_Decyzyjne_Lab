import numpy as np

def get_gradient(a, b):
    return np.array([2*a, 2*b]) # for f(x, y) = x^2 + y^2

def steepest_descent_method(start_point, step_length, min_accuracy):
    current_point = np.array(start_point)
    iteration = 0
    accuracy = min_accuracy + 1 # just to start the loop
    continue_loop = True
    while continue_loop:
        if accuracy < min_accuracy:
            print("Kryterium stopu osiągnięte, " + str(accuracy) + " < " + str(min_accuracy))
            continue_loop = False
        gradient = get_gradient(*current_point)
        next_point = current_point - step_length * gradient
        accuracy = round(np.linalg.norm(next_point - current_point),4)
        print("Iteracja", iteration, ": x =", current_point, "Gradient:", gradient, "Dokładność do następnego:", accuracy)
        iteration += 1
        current_point = next_point
    

# Test
start_point = [4, 4]
step_length = 0.3
min_accuracy = 1e-2
steepest_descent_method(start_point, step_length, min_accuracy)