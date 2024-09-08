import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from datetime import datetime

# Load the data
config_use_cases = {
    "telesales": {"identifier": "company_campaign_id", "num_customers_1": 5262, "num_customers_0": 9663},
    "print": {"identifier": "person_campaign_id", "num_customers_1": 6285, "num_customers_0": 7349},
    "webinar": {"identifier": "", "num_customers_1": 36352, "num_customers_0": 81648}
}

use_cases = ["telesales", "print","webinar"]
targets = [0, 1]

for use_case in use_cases:
    for target in targets:
        df = pd.read_parquet(f'res/{use_case}_target_{target}.parquet')

        # Convert 'date' to datetime and then to numerical format (days since the start of the dataset)
        df['date'] = pd.to_datetime(df['date'])
        start_date = df['date'].min()
        df['event_time'] = (df['date'] - start_date).dt.days

        # Ensure 'TYPE' is a categorical variable
        df['TYPE'] = pd.Categorical(df['TYPE'])
        df['event_type'] = df['TYPE'].cat.codes

        # Select relevant columns
        events_data = df[['event_type', 'event_time']].astype(float).to_numpy()

        # Analyze the real dataset to determine the number of events per customer
        events_per_customer = df[config_use_cases[use_case]['identifier']].value_counts().values

        # Define a more complex ODE function
        class ODEFunc(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(ODEFunc, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),  # Using Tanh for smoother transitions
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, t, x):
                return self.net(x)

        # Define the Neural ODE model
        class NeuralODE(nn.Module):
            def __init__(self, ode_func):
                super(NeuralODE, self).__init__()
                self.ode_func = ode_func
            
            def forward(self, x, t):
                out = odeint(self.ode_func, x, t)
                return out

        input_dim = 2  # event_type and event_time
        hidden_dim = 30  # Hidden dimension size
        ode_func = ODEFunc(input_dim, hidden_dim)
        model = NeuralODE(ode_func)

        # Prepare data for training
        inputs = torch.tensor(events_data, dtype=torch.float32)

        # Define training parameters
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        # Training loop
        num_epochs = 300  # Number of epochs
        t = torch.linspace(0, 1, steps=2)  # Dummy time points for ODE

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred = model(inputs, t)[-1]  # Get the final time step prediction
            loss = criterion(pred, inputs)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Set the number of synthetic customers
        num_customers = config_use_cases[use_case][f"num_customers_{target}"]

        # Generate new events for each customer
        synthetic_data_list = []

        # Sample event counts for synthetic customers based on the real dataset's distribution
        sampled_event_counts = np.random.choice(events_per_customer, num_customers, replace=True)

        for customer_id in range(num_customers):
            num_events = sampled_event_counts[customer_id]

            # Generate synthetic inputs based on the distribution of the original dataset
            mean_input = df[['event_type', 'event_time']].mean().values
            std_input = df[['event_type', 'event_time']].std().values
            synthetic_inputs = torch.tensor(np.random.normal(mean_input, std_input, size=(num_events, 2)), dtype=torch.float32)

            with torch.no_grad():
                synthetic_events = model(synthetic_inputs, t)[-1]

            # Convert synthetic events to DataFrame
            synthetic_df = pd.DataFrame(synthetic_events.numpy(), columns=['event_type', 'event_time'])
            synthetic_df['event_type'] = pd.Categorical.from_codes(
                synthetic_df['event_type'].astype(int) % len(df['TYPE'].cat.categories), df['TYPE'].cat.categories
            )
            synthetic_df['date'] = pd.to_datetime(start_date) + pd.to_timedelta(synthetic_df['event_time'], unit='D')
            synthetic_df['customer_id'] = f'synthetic_{customer_id}'  # Ensure unique customer IDs

            # Add additional columns as needed
            synthetic_df['amount_net'] = np.random.normal(
                loc=df['amount_net'].mean(), scale=df['amount_net'].std(), size=len(synthetic_df)
            )

            synthetic_data_list.append(synthetic_df)

        # Combine all synthetic data into a single DataFrame
        all_synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)

        print(all_synthetic_data.head())

        # Evaluate the synthetic data
        print(all_synthetic_data.describe())
        print(all_synthetic_data[['event_type', 'event_time', 'amount_net']].describe())
        all_synthetic_data.to_csv(f'res/synthetic_selected_customers_events_{use_case}_{target}.csv', index=False)
