from TestEnv import Electric_Car
import argparse
import matplotlib.pyplot as plt
import numpy as np
import utils
import agent 
import json
import os
import pandas as pd


# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='data/validate.xlsx') # Path to the excel file with the test data
parser.add_argument('--config_file', type=str, default='config.json') # Path to the excel file with the test data

args = parser.parse_args()

env = Electric_Car(path_to_test_data=args.excel_file)

# create file to store all rewards in if it doesnt exist yet
if not os.path.isfile('test_rewards.txt'):
    rewards = {}

else:
    # reading the rewards data from the file
    with open('test_rewards.txt', 'r') as f:
        data = f.read()
        
    # reconstructing the data as a dictionary 
    rewards = json.loads(data)

# define the actions, need to be inline with agent
# num_actions can be 3, 5, 11, 21              => not test for now : 11, 21
# num_battery_levels can be 6 11 26 51         => not test for now : 26, 51
# num price can be 3, 4, 5, 7, 11              => not test for now : 11
# num_hours can be 3, 4, 6, 12, 24             => not test for now : 12, 24
config_file = open(args.config_file)
config = json.load(config_file)

bin_size, properties = config['bin_size'], config['properties']
#We can also train the Qagent with a decaying epsilon schedule

RL_agent = agent.QAgent(env=env, bin_size=bin_size, properties=properties)

# initialize logging vars
cumulative_reward = []
battery_level = []

# for final output dict
total_reward = []
actions = []
battery_levels = []
prices = []
hours = []

observation = env.observation()
for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    # Choose a random action between -1 (full capacity sell) and 1 (full capacity charge)
    # action = env.continuous_action_space.sample()
    # Only choose randomly 1 or -1 or 0
    # action = np.random.choice([-1, 0, 1])
    # action = np.random.choice(actions)
    # Or choose an action based on the observation using your RL agent!:
    action = RL_agent.act(observation)
    # The observation is the tuple: [battery_level, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    # log all desired vars, transform from numpy dtypes to native python types if necessary (to allow JSON encoding) 
    actions += [float(action)] if not isinstance(action, float) else [action]
    battery_levels += [int(observation[0])] if not isinstance(observation[0],int) else [observation[0]]
    prices += [float(observation[1])] if not isinstance(observation[1],float) else [observation[1]]
    hours += [int(observation[2])] if not isinstance(observation[2],int) else int([observation[2]])
    
    total_reward += [float(reward)] if isinstance(reward,float) else [reward]

    cumulative_reward.append(sum(total_reward))

    done = terminated or truncated
    observation = next_observation

    if done:
        # saving all rewards and actions
        out = {'rewards': total_reward, 'actions': actions, 'battery_levels': battery_levels, 'price': prices, 'hours':hours}

        # add current run to rewards dictionary
        rewards[RL_agent.k] = out

        # save new rewards dictionary to file
        with open('test_rewards.txt', 'w') as f: 
            f.write(json.dumps(rewards))

        print('Total reward: ', sum(total_reward))
        # # Plot the cumulative reward over time
        # plt.plot(cumulative_reward)
        # plt.xlabel('Time (Hours)')

        # plt.ylabel('Cumulative reward')
        # # plt.title(f"Total Reward {RL_agent.k}")
        # plt.show() 
        
        





hours = np.arange(1, len(hours)+1, 1)
out['hours'] = hours

# print(len(out['hours']))
# print(len(out['actions']))
# print(len(out['price']))
# Example data (replace this with your own data)
# i for i in range(730*24 -1)
# data = {
#     'hour': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
#     'price': [10, 15, 12],
#     'charging_amount': [100, 200, 50]  # Example charging amounts (replace with your data)
# }

# # Convert the data to a Pandas DataFrame
num = 150
df = pd.DataFrame.from_dict(out)[num:num+200]

# # Convert the timestamp column to a datetime object
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# # Extract hours from the timestamp
# df['hour'] = df['timestamp'].dt.hour

# # Plot the time series

plt.figure(figsize=(10, 6))

# # Plot the time series line
# plt.plot(df['hours'], df['price'], linestyle='-', label='Price', color='gray',alpha=0.1)
plt.plot(df['hours'], df['price'], linestyle='-', label='Price', color='gray')
# # Scatter plot for charging or discharging
scatter = plt.scatter(df['hours'], df['price'], c=df['actions'], cmap='coolwarm', s=50, label='Action')

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_label('Actions')

# Add labels and title
plt.xlabel('Hours')
plt.ylabel('Price')
plt.title('Electricity Price vs Actions evolved along the hours')

# Add legend
plt.legend()

# Show the plot
plt.show()
    
    

