import pandas as pd
import plotly.graph_objects as go
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


# Define event groups
event_groups = {
    'activity': 'engagement',
    'bank_charge_booked': 'financial',
    'cancellation': 'status_change',
    'cancellation_booked': 'status_change',
    'contract_created': 'transaction',
    'conversion': 'status_change',
    'dunning_booked': 'financial',
    'invoice_received': 'financial',
    'letter': 'communication',
    'payment_booked': 'financial',
    'return_debit_booked': 'financial',
    'return_remittance_booked': 'financial',
    'revenue_booked': 'financial',
    'tm_called_on': 'communication',
    'write_off_booked': 'financial',
    'email_click': 'engagement',
    'email_subscribe': 'engagement',
    'email_unsubscribe': 'engagement',
    'webinar_attendee': 'engagement',
    'webinar_registration': 'engagement'
}

# Define colors for each group
color_map = {
    'engagement': 'skyblue',
    'financial': 'lightgreen',
    'status_change': 'salmon',
    'transaction': 'gold',
    'communication': 'lightcoral'
}

# List of all event types
event_types = [
    'activity', 'bank_charge_booked', 'cancellation', 'cancellation_booked', 
    'contract_created', 'conversion', 'dunning_booked', 'invoice_received', 
    'letter', 'payment_booked', 'return_debit_booked', 'return_remittance_booked', 
    'revenue_booked', 'tm_called_on', 'write_off_booked', 'email_click', 
    'email_subscribe', 'email_unsubscribe', 'webinar_attendee', 'webinar_registration'
]

mode = 'synthetic'
if 'live' in mode:
    # Load contract_created data
    contract_created_df = pd.read_csv('data/prefiltered/webinar/train/contract_created/contract_created.csv')

    # Identify customers based on number of contracts
    contracts_per_customer = contract_created_df['email_webinar_id'].value_counts()

    # Select customers
    customer_1 = contracts_per_customer[contracts_per_customer == 1].index[0]
    customer_2 = contracts_per_customer[(contracts_per_customer >= 5) & (contracts_per_customer <= 10)].index[0]
    customer_3 = contracts_per_customer[contracts_per_customer > 50].index[0]
    customer_4 = contracts_per_customer[contracts_per_customer == 1].index[1]
    customer_5 = contracts_per_customer[(contracts_per_customer >= 5) & (contracts_per_customer <= 10)].index[2]
    customer_6 = contracts_per_customer[contracts_per_customer > 50].index[3]

    selected_customers = [customer_1, customer_2, customer_3, customer_4, customer_5, customer_6]

    print(f"Selected Customers: {selected_customers}")

    # Initialize an empty dataframe to collect all events
    all_events = pd.DataFrame()
    all_events_max = pd.DataFrame()


    # Loop through all event types and collect events for selected customers
    for event_type in event_types:
        file_path = f'data/prefiltered/webinar/train/{event_type}/{event_type}.csv'
        event_df = pd.read_csv(file_path)
        event_df['date'] = pd.to_datetime(event_df['date'])

        all_events_max = pd.concat([all_events, event_df], ignore_index=True)

        # Filter events for the selected customers
        event_df_filtered = event_df[event_df['email_webinar_id'].isin(selected_customers)]
        all_events = pd.concat([all_events, event_df_filtered], ignore_index=True)
    # Order all events by time
    all_events = all_events.sort_values(by='date')
    all_events_max = all_events_max.sort_values(by='date')
else:
    all_events = pd.read_csv('data/prefiltered/customer_journey/synthetic_selected_customers_events.csv')
    all_events.rename(columns={'event_type': 'TYPE', 'customer_id': 'email_webinar_id'}, inplace=True)
    all_events['date'] = pd.to_datetime(all_events['date'])

    all_events = all_events.sort_values(by='date')
    selected_customers = all_events['email_webinar_id'].unique()
    print (f"selected_customers: {selected_customers}")


# Map the event types to their groups

all_events['event_group'] = all_events['TYPE'].map(event_groups)


# Extract the year and month for grouping
all_events['year_month'] = all_events['date'].dt.to_period('M')


# Save the ordered events for reference
if 'live' in mode:
    all_events.to_csv('data/prefiltered/customer_journey/selected_customers_events.csv', index=False)
    all_events_max.to_csv('data/prefiltered/customer_journey/selected_customers_events_max.csv', index=False)

# Group by customer and count the number of events, and find the first and last event
summary = all_events.groupby(['email_webinar_id', 'TYPE']).agg(
    event_count=('TYPE', 'count'),
    event_sum=('amount_net', 'sum'),
    first_event=('date', 'min'),
    last_event=('date', 'max')
).reset_index()

# Save the summary table
print ("SAVE SUMMARY")

summary.to_csv(f'data/prefiltered/customer_journey/{mode}_customer_event_summary.csv', index=False)

print (summary)

# Ensure timestamp is in datetime format
all_events['date'] = pd.to_datetime(all_events['date'])

# Create a new column for the next event type for transitions
all_events['next_event_type'] = all_events.groupby('email_webinar_id')['TYPE'].shift(-1)

# Filter out rows where the next_event_type is NaN (end of the sequence for a customer)
transitions = all_events.dropna(subset=['next_event_type'])

# Create a table of transitions
transition_counts = transitions.groupby(['TYPE', 'next_event_type']).size().reset_index(name='count')

### TIMELINE ####
print ("TIMELINE")
# Function to plot timeline for a customer
def plot_timeline(customer_id, customer_events):
    if customer_events.empty:
        print(f"No events for customer {customer_id}")
        return
    
    plt.figure(figsize=(15, 8))
    
    # Sort events by timestamp
    customer_events = customer_events.sort_values('date')
    
    # Plot each event type on the y-axis
    event_types = customer_events['TYPE'].unique()
    event_type_map = {event: idx for idx, event in enumerate(event_types)}
    
    # Plot each event
    for idx, row in customer_events.iterrows():
        plt.plot([row['date'], row['date']], [event_type_map[row['TYPE']], event_type_map[row['TYPE']] + 0.4], 
                 color=color_map[row['event_group']], alpha=0.6, linewidth=3)
    
    # Add event labels
    plt.scatter(customer_events['date'], 
                [event_type_map[event] for event in customer_events['TYPE']], 
                color=customer_events['event_group'].map(color_map), s=100, zorder=5)
    
    # Add text labels
    # for idx, row in customer_events.iterrows():
    #     plt.text(row['date'], event_type_map[row['TYPE']] + 0.45, row['TYPE'], rotation=45, ha='right', va='bottom', fontsize=10)
    
    # Format the plot
    plt.title(f"Customer Journey Timeline for {customer_id}")
    plt.xlabel('Time')
    plt.yticks(range(len(event_types)), event_types)
    plt.grid(axis='x')
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Add legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in color_map.values()]
    labels = list(color_map.keys())
    plt.legend(handles, labels, title='Event Groups')
    
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs('data/prefiltered/customer_journey/', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'data/prefiltered/customer_journey/{mode}_{customer_id}_timeline.png')

# Generate and save timeline for each customer
for customer_id in selected_customers:
    customer_events = all_events[all_events['email_webinar_id'] == customer_id]
    plot_timeline(customer_id, customer_events)



#### GRAPH ######
print ("GRAPH")
# Create a directed graph
# Create a directed graph
# Function to plot graph layout for a customer
def plot_graph(customer_id, customer_events):
    if customer_events.empty:
        print(f"No events for customer {customer_id}")
        return

    # Create a new column for the next event type
    customer_events['next_event_type'] = customer_events.groupby('email_webinar_id')['TYPE'].shift(-1)
    
    # Filter out rows where the next_event_type is NaN (end of the sequence for a customer)
    transitions = customer_events.dropna(subset=['next_event_type'])
    
    # Create a table of transitions
    transition_counts = transitions.groupby(['TYPE', 'next_event_type']).size().reset_index(name='count')
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges with weights
    for _, row in transition_counts.iterrows():
        G.add_edge(row['TYPE'], row['next_event_type'], weight=row['count'])
    
    # Apply the Fruchterman-Reingold layout
    pos = nx.spring_layout(G)
    
    # Map each node to its group color
    node_colors = [color_map[event_groups[node]] for node in G.nodes]
    
    # Draw the graph
    plt.figure(figsize=(14, 10))
    
    # Draw nodes with colors
    nx.draw_networkx_nodes(G, pos, node_size=7000, node_color=node_colors, node_shape='o', alpha=0.7)
    
    # Draw edges with weights
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, alpha=0.5, edge_color='gray')
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): d['weight'] for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Create a legend
    for group, color in color_map.items():
        plt.scatter([], [], c=color, label=group)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Event Groups')
    
    plt.title(f"Customer Journey Graph for {customer_id}")
    plt.axis('off')
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(f'data/prefiltered/{mode}_customer_journey/', exist_ok=True)
    
    # Save the graph as a PNG file
    plt.savefig(f'data/prefiltered/customer_journey/{mode}_{customer_id}_graph.png')
    plt.close()

for customer_id in selected_customers:
    customer_events = all_events[all_events['email_webinar_id'] == customer_id]
    plot_graph(customer_id, customer_events)


def create_sankey(customer_id, customer_events):
    # Create a new column for the next event type
    customer_events['next_event_type'] = customer_events.groupby('email_webinar_id')['TYPE'].shift(-1)
    
    # Filter out rows where the next_event_type is NaN (end of the sequence for a customer)
    transitions = customer_events.dropna(subset=['next_event_type'])
    
    # Create a table of transitions
    transition_counts = transitions.groupby(['TYPE', 'next_event_type']).size().reset_index(name='count')
    
    # Prepare the data for the Sankey diagram
    labels = list(set(transition_counts['TYPE']).union(set(transition_counts['next_event_type'])))
    label_indices = {label: i for i, label in enumerate(labels)}

    source = [label_indices[event] for event in transition_counts['TYPE']]
    target = [label_indices[event] for event in transition_counts['next_event_type']]
    value = transition_counts['count']

    # Create the Sankey diagram
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightblue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(150, 0, 90, 0.4)'
        )
    )])

    fig_sankey.update_layout(
        title_text=f"Customer Journey Sankey Diagram for {customer_id}",
        font_size=12,
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Ensure the directory exists
    os.makedirs('data/prefiltered/customer_journey/', exist_ok=True)
    
    # Save the Sankey diagram as a PNG file
    fig_sankey.write_image(f'data/prefiltered/customer_journey/{mode}_{customer_id}_sankey.png', engine='kaleido')


# Generate and save Sankey diagram for each selected customer
for customer_id in selected_customers:
    customer_events = all_events[all_events['email_webinar_id'] == customer_id]
    create_sankey(customer_id, customer_events)