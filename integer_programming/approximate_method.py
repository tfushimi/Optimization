def greedy_knapsack_solver(items, capacity, prices, weights):
    """
    a) Solve a Knapsack problem with a greedy algorithm
        maximize sum(price_i * x_i)
        subject to sum(weight_i * x_i) <= capacity
        x_i is 0 or 1, i = 1~n

    b) The greedy algorithm is
        1) Sort items in descending order based on price_i / weight_i
        2) In the order of 1), repeat the following for i = 1~n.
           If item_i is less than the remaining capacity, let x_i = 1.
           Otherwise, let x_i = 0.

    Args
    capacity: total capacity
    prices: dict of prices
    weights: dict of weights

    Returns
    greedy_x: approximate solution
    greedy_total: approximate total price
    """
    ratio = {i: prices[i] / weights[i] for i in items}
    sorted_items = [key for key, val in sorted(ratio.items(), key=lambda x: x[1], reverse=True)]

    greedy_x = {i: 0 for i in sorted_items}
    cap = capacity
    for i in sorted_items:
        if weights[i] <= cap:
            cap -= weights[i]
            greedy_x[i] = 1
    greedy_total = sum(prices[i] * greedy_x[i] for i in sorted_items)
    return greedy_x, greedy_total


def approximate_knapsack_solver(items, capacity, prices, weights):
    """
    a) Solve a Knapsack problem with an approximate algorithm
        maximize sum(price_i * x_i)
        subject to sum(weight_i * x_i) <= capacity
        x_i is 0 or 1, i = 1~n

    b) The approximate algorithm is
        1) Sort items in descending order based on price_i / weight_i
        2) In the order of 1), repeat the following.
           If item_i is less than the remaining capacity, let x_i = 1.
           otherwise, let x_i = the remaining capacity / weights_i and break.

    Args
    capacity: total capacity
    prices: dict of prices
    weights: dict of weights

    Returns
    apx_x: approximate solution
    apx_total: approximate total price
    """
    ratio = {i: prices[i] / weights[i] for i in items}
    sorted_items = [key for key, val in sorted(ratio.items(), key=lambda x: x[1], reverse=True)]

    approx_x = {i: 0 for i in sorted_items}
    cap = capacity
    for i in sorted_items:
        if weights[i] <= cap:
            cap -= weights[i]
            approx_x[i] = 1
        else:
            approx_x[i] = cap / weights[i]
            break
    approx_total = sum(prices[i] * approx_x[i] for i in sorted_items)
    return approx_x, approx_total