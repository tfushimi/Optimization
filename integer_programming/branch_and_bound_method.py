from collections import deque

class KnapsackProblem(object):
    """
    The definition of a Knapsack Problem:
        maximize sum(price_i * x_i)
        subject to sum(weight_i * x_i) <= capacity
        x_i is 0 or 1, i = 1~n
    """
    def __init__(self, name, items, capacity, prices, weights, zeros, ones):
        """
        The constructor

        name: string
        items: set of index
        capacity: scalar value of total capacity
        prices: dictionary of prices, keys are items
        weights: dictionary of weights, keys are items
        zeros: set of index such that x_i = 0
        ones: set of index such that x_i = 1
        """
        # The description of a Knapsack problem
        self.name = name
        self.items = items
        self.capacity = capacity
        self.prices = prices
        self.weights = weights

        # The sorted item list based on price_i / weights_i
        ratio = {i: prices[i] / weights[i] for i in items}
        self.sorted_item_list = [k for k, v in sorted(ratio.items(), key=lambda x:x[1], reverse=True)]

        # A set of variable index such that x_i = 0
        self.zeros = zeros
        # A set of variable index such that x_i = 1
        self.ones = ones

        # The lower and upper bound of the objective function
        self.lower_bound = -float("inf")
        self.upper_bound = -float("inf")

        # solution of a greedy algorithm
        self.x_lower_bound = {i:0 for i in self.items}
        # solution of an approximate algorithm
        self.x_upper_bound = {i:0 for i in self.items}

        # the variable to be used for the next split
        self.fraction_index = None

    def compute_bounds(self):
        """
        Compute the upper and lower bounds
        """
        # fix x_i = 0 for i in self.zeros
        for i in self.zeros:
            self.x_lower_bound[i] = self.x_upper_bound[i] = 0
        # fix x_i = 1 for i in self.ones
        for i in self.ones:
            self.x_lower_bound[i] = self.x_upper_bound[i] = 1
        # do not explore if self.capacity < sum(x_i) for i in self.ones
        if self.capacity < sum(self.weights[i] for i in self.ones):
            self.lower_bound = self.upper_bound = -float("inf")
            return None

        # get the remaining variables and the remaining capacity
        remaining_items = self.items - self.zeros - self.ones
        sorted_remaining_items = [i for i in self.sorted_item_list if i in remaining_items]
        remaining_capacity = self.capacity - sum(self.weights[i] for i in self.ones)

        # compute the lower and upper bound
        stop = False
        for i in sorted_remaining_items:
            if self.weights[i] <= remaining_capacity:
                # include item_i
                remaining_capacity -= self.weights[i]
                self.x_lower_bound[i] = self.x_upper_bound[i] = 1
            elif not stop:
                # include a fraction of item_i for upper bound
                self.x_upper_bound[i] = remaining_capacity / self.weights[i]
                # the index of variable whose approximate solution is fractional
                self.fraction_index = i
                stop = True
        self.lower_bound = sum(self.prices[i] * self.x_lower_bound[i] for i in self.items)
        self.upper_bound = sum(self.prices[i] * self.x_upper_bound[i] for i in self.items)

def branch_and_bound_knapsack_solver(items, capacity, prices, weights):
    """
    Solve a Knapsack problem with Branch-and-Bound method
        maximize sum(price_i * x_i)
        subject to sum(weight_i * x_i) <= capacity
        x_i is 0 or 1, i = 1~n

    Args:
    items: set of index
    capacity: scalar value of total capacity
    prices: dictionary of prices, keys are items
    weights: dictionary of weights, keys are items

    Returns:
    optimal value, optimal solution
    """
    # Initialization
    queue = deque()
    root = KnapsackProblem("KP", items=items, capacity=capacity, prices=prices, weights=weights, zeros=set(), ones=set())
    root.compute_bounds()
    best = root
    queue.append(root)

    # Iteration
    while len(queue) != 0:
        # dequeue a problem
        prob = queue.popleft()
        # compute the upper and lower bounds
        prob.compute_bounds()

        if prob.upper_bound > best.lower_bound: # may need to update best
            if prob.lower_bound > best.lower_bound:
                # update best
                best = prob

            if prob.upper_bound > prob.lower_bound: # the current solution is not optimal
                # choose a variable whose approximate solution is fractional
                k = prob.fraction_index
                p1 = KnapsackProblem(prob.name+'+'+str(k), items=prob.items, capacity=prob.capacity, prices=prob.prices,
                                     weights=prob.weights, zeros=prob.zeros, ones=prob.ones.union({k}))
                # enqueue a sub problem with x_k = 1
                queue.append(p1)
                p2 = KnapsackProblem(prob.name+'-'+str(k), items=prob.items, capacity=prob.capacity, prices=prob.prices,
                                     weights=prob.weights, zeros=prob.zeros.union({k}), ones=prob.ones)
                # enqueue a sub problem with x_k = 0
                queue.append(p2)

    return best.lower_bound, best.x_lower_bound
