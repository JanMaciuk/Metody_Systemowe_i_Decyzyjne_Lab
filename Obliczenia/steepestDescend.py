import numpy as np

def gradient_f(a, b):
    return np.array([2*a, 2*b]) # for f(x, y) = x^2 + y^2

def steepest_descent_method(start_point, step_length, min_gradient):
    current_point = np.array(start_point)
    iteration = 0
    
    while True:
        gradient = gradient_f(*current_point)
        print("Iteracja", iteration, ": x =", current_point, "Gradient:", gradient, "Długość gradientu:", np.linalg.norm(gradient))
        
        if np.linalg.norm(gradient) < min_gradient:
            break
        
        next_point = current_point - step_length * gradient
        current_point = next_point
        iteration += 1
    
    print("Minimum point:", current_point)

# Test
start_point = [4, 4]
step_length = 0.3
min_gradient = 1e-2

steepest_descent_method(start_point, step_length, min_gradient)