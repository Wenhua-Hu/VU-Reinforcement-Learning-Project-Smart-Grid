from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import utils
import TestEnv


class QAgent():
    
    def __init__(self, env, bin_size = {'battery': 51, 'price': 11,'hour': 24, 'action':3},
                  properties={'reward_shaping':1, 'penalties':1, 'nr_simulations':1000, 'discount_rate':0.95}):
        
        '''
        Params:
        
        env_name = name of the specific environment that the agent wants to solve
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        #create an environment
        self.env = env
        
        # #Set the discount rate
        # self.discount_rate = discount_rate
        
        self.properties = properties
        
        # self.bin_prices = {
        #     '3': np.quantile(values_price, [0, 0.5, 0.99]),
        #     '4': np.quantile(values_price, [0, 0.33, 0.66, 0.99]),
        #     '5': np.quantile(values_price, [0, 0.25, 0.5, 0.75, 0.99]),
        #     '7': np.quantile(values_price, [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.99]),
        #     '11': np.quantile(values_price, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]),   
        # }
        #Set the bin size
        self.bin_size = bin_size
        
        # n different actions
        self.action_space = self.bin_size['action']
        self.actions = utils.get_action_space(num_actions=self.action_space)
        
        # bins of prices
        train_data = pd.read_excel("data/train.xlsx")
        price_values = train_data.iloc[:, 1:25].to_numpy()
        values_price = price_values.flatten()
        self.bin_price = utils.get_price_bins(values_price, self.bin_size["price"])
        print(f"Prices ({str(self.bin_size['price'])} bins): {self.bin_price}")

        # bins of battery levels
        self.battery_low = 0
        self.battery_high = 50
        self.bin_battery = np.linspace(self.battery_low, self.battery_high, self.bin_size['battery'])
        print(f"Battery Levels ({self.bin_size['battery']} bins): {self.bin_battery}")

        # bins of hours
        self.hour_low = 0
        self.hour_high = 24
        self.bin_hour = np.linspace(self.hour_low, self.hour_high, self.bin_size['hour'] + 1)
        print(f"Hours ({self.bin_size['hour']} bins): {self.bin_hour}")
        '''
        ToDo:
        
        Please create the bins for the velocity feature in the same manner and call this variable self.bin_velocity!
        '''
        
        #Solution
        # self.bin_hour = np.linspace(self.hour_low, self.hour_high, self.bin_size['hour']) 
        
        #Append the two bins
        self.bins = [self.bin_battery, self.bin_price, self.bin_hour]
        
        # get all properties for file loading
        self.property_name = f"disc_{self.properties['discount_rate']}_shap_{self.properties['reward_shaping']}_pen_{self.properties['penalties']}_sims_{self.properties['nr_simulations']}"
        # load Qtable
        filename = f"qtable_{self.property_name}_batt_{self.bin_size['battery']}_price_{self.bin_size['price']}_hour_{self.bin_size['hour']}_action_{self.bin_size['action']}.npy"
        self.Qtable = np.load(f"data/{filename}")

        # initialize filename var
        self.k = filename
        
    
    def discretize_state(self, state):
        
        '''
        Params:
        state = state observation that needs to be discretized
        
        
        Returns:
        discretized state
        '''
        #Now we can make use of the function np.digitize and bin it
        self.state = state
        # print(f"self.state: {self.state}")
        
        #Create an empty state
        digitized_state = []
    
        # (-inf, 0) [0, 1)    [50, )  # 52 states
        digitized_state.append(np.digitize(self.state[0], self.bins[0], right=False) -1) 
        digitized_state.append(np.digitize(self.state[1], self.bins[1], right=False) -1)
        digitized_state.append(np.digitize(self.state[2], self.bins[2], right=True) -1)
        

        #Returns the discretized state from an observation
        return digitized_state
    
    
    # def create_Q_table(self):
    #     # self.state_space = self.bin_size - 1
    #     #Initialize all values in the Q-table to zero
    #     self.state_battery_space = self.bin_size['battery']
    #     self.state_hour_space = self.bin_size['hour']
    #     self.state_price_space = self.bin_size['price']

    #     '''
    #     ToDo:
    #     Initialize a zero matrix of dimension state_space * state_space * action_space and call it self.Qtable!
    #     '''
        
    #     #Solution:
    #     # self.Qtable = np.zeros((self.state_space, self.state_space, self.action_space))
    #     self.Qtable = np.zeros((self.state_battery_space, 
    #                             self.state_price_space,
    #                             self.state_hour_space,
    #                             self.action_space))
    #     # print(self.Qtable.shape)
        
    # def save_Q_table(self):
    #     table_shape = self.Qtable.shape
        
    #     num_battery_levels = table_shape[0]
    #     num_price_levels = table_shape[1]
    #     num_hours = table_shape[2]
    #     num_actions = table_shape[3]
        
    #     filename = f"qtable_{num_battery_levels}_{num_price_levels}_{num_hours}_{num_actions}.npy"
    #     np.save(f"data/{filename}", self.Qtable)
    #     print(f"{filename} is saved ...")
        
    def reload_Q_table(self, bin_size, properties):
        self.bin_size = bin_size
        self.properties = properties
        self.property_name = f"disc_{self.properties['discount_rate']}_shap_{self.properties['reward_shaping']}_pen_{self.properties['penalties']}_sims_{self.properties['nr_simulations']}"

        # get all variables for file name
        filename = f"qtable_{self.property_name}_batt_{self.bin_size['battery']}_price_{self.bin_size['price']}_hour_{self.bin_size['hour']}_action_{self.bin_size['action']}.npy"
        self.k = filename

        self.Qtable = np.load(f"data/{filename}")
        return self.Qtable
    
    def act(self, state):
        state = self.discretize_state(state)
        action_i = np.argmax(self.Qtable[state[0],
                                         state[1],
                                         state[2],
                                        :])
        action = self.actions[action_i]
        
        return action
        
    def visualize_rewards(self):
        plt.figure(figsize =(7.5,7.5))
        plt.plot(self.num_simulations, self.average_rewards)
        plt.axhline(y = -110, color = 'r', linestyle = '-')
        plt.title('Average reward over the past 100 simulations', fontsize = 10)
        plt.legend(['Q-learning performance','Benchmark'])
        plt.xlabel('Number of simulations', fontsize = 10)
        plt.ylabel('Average reward', fontsize = 10)
        
            
if __name__ == "__main__":
    properties = {'reward_shaping':1, 'penalties':1, 'nr_simulations':10, 'discount_rate':0.0}
    # num_battery_levels can be 6 11 26 51         => not test for now : 26, 51
    # num price can be 3, 4, 5, 7, 11              => not test for now : 11
    # num_hours can be 3, 4, 6, 12, 24             => not test for now : 12, 24
    # num_actions can be 3, 5, 11, 21              => not test for now : 11, 21
    bin_size = {'battery': 6, 'price': 3,'hour': 3, 'action': 3 }
    
    test_env = TestEnv.Electric_Car(path_to_test_data="data/validate.xlsx")
    RL_agent = QAgent(env=test_env, bin_size=bin_size, properties=properties)
    
    print(RL_agent.Qtable.shape)
    print(RL_agent.Qtable)
    
    qtable = RL_agent.reload_Q_table(bin_size=bin_size)
    print(qtable.shape)
    print(qtable)
    
