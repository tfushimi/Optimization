from approximate_method import greedy_knapsack_solver, approximate_knapsack_solver
from branch_and_bound_method import branch_and_bound_knapsack_solver

items = {1, 2, 3, 4, 5}
prices = {1:50, 2:40, 3:10, 4:70, 5:55}
weights = {1:7, 2:5, 3:1, 4:9, 5:6}
capacity = 15

greedy_x, greedy_total = greedy_knapsack_solver(items, capacity, prices, weights)
print("Greedy total price is", greedy_total)
print("Greedy Solution is ", greedy_x)
print()

approx_x, approx_total = approximate_knapsack_solver(items, capacity, prices, weights)
print("Approximate total price is", approx_total)
print("Approximate Solution is ", approx_x)
print()

bb_total, bb_x = branch_and_bound_knapsack_solver(items, capacity, prices, weights)
print("Optimal value = ", bb_total)
print("Optimal solution = ", bb_x)
