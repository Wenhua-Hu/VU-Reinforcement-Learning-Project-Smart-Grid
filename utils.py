import numpy as np

def get_price_bins(values_price, num_prices):
    bin_prices = {
            '3': np.quantile(values_price, [0, 0.5, 0.99]),
            '4': np.quantile(values_price, [0, 0.33, 0.66, 0.99]),
            '5': np.quantile(values_price, [0, 0.25, 0.5, 0.75, 0.99]),
            '7': np.quantile(values_price, [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.99]),
            '11': np.quantile(values_price, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]),   
        }
    return bin_prices[str(int(num_prices))]

def get_action_space(num_actions):
    action_spaces = {
            '3': np.array([-1, 0, 1]),
            '5': np.array([-1, -0.5, 0, 0.5, 1]),
            '11': np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0 ]),
            '21': np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0 , 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0 ])
        }
    return action_spaces[str(int(num_actions))]

