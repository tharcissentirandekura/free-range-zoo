import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import ast
from pathlib import Path

def load_and_process_data(csv_path):
    """Load and process the wildfire simulation data"""
    data = pd.read_csv(csv_path)
    # Convert string representations to Python objects
    def safe_literal_eval(val):
        if pd.isna(val) or val == 'NULL':
            return None
        try:
            return ast.literal_eval(val)
        except:
            return val
    
    # Convert key columns
    list_columns = ['fires', 'intensity', 'fuel', 'suppressants', 'capacity', 
                   'equipment', 'agents', 'firefighter_1_action', 'firefighter_2_action', 
                   'firefighter_3_action']
    
    for col in list_columns:
        if col in data.columns:
            # evaluate 
            data[col] = data[col].apply(safe_literal_eval)
    return data

# Count fires in the environment
# At each step
def count_fires_and_fuel(data):
    """Extract fire counts and fuel totals from each step"""
    fire_counts = []
    fuel_totals = []
    burnt_counts = []
    
    for idx in range(len(data)):
        fires_grid = data['fires'].iloc[idx]
        fuel_grid = data['fuel'].iloc[idx]
        
        if fires_grid and fuel_grid:
            # Count active fires (positive values)
            active_fires = sum(1 for row in fires_grid for cell in row if cell > 0)
            # Count burnt areas (negative values)
            # burnt_areas = sum(1 for row in fires_grid for cell in row if cell < 0)
            burnt_areas = 0

            for i,fire_row in enumerate(fires_grid):
                for j, fire_cell in enumerate(fire_row):
                    fuel_cell = fuel_grid[i][j]
                    if fire_cell < 0 and fuel_cell == 0:
                        # burned out
                        burnt_areas += 1
            fire_counts.append(active_fires)
            burnt_counts.append(burnt_areas)
        else:
            fire_counts.append(0)
            burnt_counts.append(0)
        
        if fuel_grid is not None:
            # Sum total fuel available
            total_fuel = sum(cell for row in fuel_grid for cell in row if cell >= 0)
            fuel_totals.append(total_fuel)
        else:
            fuel_totals.append(0)
    
    return fire_counts, fuel_totals, burnt_counts

def create_comprehensive_analysis(csv_path):
    """Create comprehensive analysis and visualization"""
    # Load data
    data = load_and_process_data(csv_path)
    fire_counts, fuel_totals, burnt_counts = count_fires_and_fuel(data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Wildfire Simulation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Fire progression over time
    axes[0, 0].plot(data['step'], fire_counts, 'r-o', label='Active Fires', linewidth=2)
    axes[0, 0].plot(data['step'], burnt_counts, 'k-s', label='Burnt Areas', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Fire Progression Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Fuel consumption
    axes[0, 1].plot(data['step'], fuel_totals, 'g-^', linewidth=2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Total Fuel')
    axes[0, 1].set_title('Fuel Consumption Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Agent rewards
    axes[1, 0].plot(data['step'], data['firefighter_1_rewards'], 'b-o', label='Firefighter 1')
    axes[1, 0].plot(data['step'], data['firefighter_2_rewards'], 'orange', marker='s', label='Firefighter 2')
    axes[1, 0].plot(data['step'], data['firefighter_3_rewards'], 'purple', marker='^', label='Firefighter 3')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Agent Rewards Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative rewards
    cum_rewards_1 = data['firefighter_1_rewards'].fillna(0).cumsum()
    cum_rewards_2 = data['firefighter_2_rewards'].fillna(0).cumsum()
    cum_rewards_3 = data['firefighter_3_rewards'].fillna(0).cumsum()
    
    axes[1, 1].plot(data['step'], cum_rewards_1, 'b-o', label='Firefighter 1')
    axes[1, 1].plot(data['step'], cum_rewards_2, 'orange', marker='s', label='Firefighter 2')
    axes[1, 1].plot(data['step'], cum_rewards_3, 'purple', marker='^', label='Firefighter 3')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Agent Rewards')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Burnouts and putouts
    axes[2, 0].bar(data['step'] - 0.2, data['burnouts'], width=0.4, label='Burnouts', alpha=0.7, color='red')
    axes[2, 0].bar(data['step'] + 0.2, data['putouts'], width=0.4, label='Putouts', alpha=0.7, color='blue')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].set_title('Burnouts vs Putouts per Step')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Suppressant levels
    suppressant_data = []
    for idx in range(len(data)):
        suppressants = data['suppressants'].iloc[idx]
        if suppressants is not None:
            suppressant_data.append(suppressants)
        else:
            suppressant_data.append([0, 0, 0])
    
    suppressant_array = np.array(suppressant_data)
    axes[2, 1].plot(data['step'], suppressant_array[:, 0], 'b-o', label='Agent 1')
    axes[2, 1].plot(data['step'], suppressant_array[:, 1], 'orange', marker='s', label='Agent 2')
    axes[2, 1].plot(data['step'], suppressant_array[:, 2], 'purple', marker='^', label='Agent 3')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Suppressant Level')
    axes[2, 1].set_title('Agent Suppressant Levels')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, data



def print_simulation_summary(data):
    """Print a summary of the simulation"""
    print("="*50)
    print("WILDFIRE SIMULATION SUMMARY")
    print("="*50)
    print(f"Total Steps: {data['step'].max()}")
    print(f"Simulation Completed: {data['complete'].iloc[-1]}")
    print(f"Total Burnouts: {data['burnouts'].sum()}")
    print(f"Total Putouts: {data['putouts'].sum()}")
    print()
    
    # Agent performance
    total_rewards = {
        'Firefighter 1': data['firefighter_1_rewards'].fillna(0).sum(),
        'Firefighter 2': data['firefighter_2_rewards'].fillna(0).sum(),
        'Firefighter 3': data['firefighter_3_rewards'].fillna(0).sum()
    }
    
    print("AGENT PERFORMANCE:")
    for agent, reward in total_rewards.items():
        print(f"{agent}: {reward:.2f} total reward")
    
    print(f"\nBest performing agent: {max(total_rewards, key=total_rewards.get)}")
    print("="*50)

if __name__ == "__main__":
    # Update this path to your actual CSV file
    csv_path = "/Users/tharcisse/Desktop/AdamResearch/Codes/free-range-zoo/free_range_zoo/envs/wildfire/configs/outputs/wildfire_logging_test_0/done.csv"
    
    
    # Create comprehensive analysis
    fig1, data = create_comprehensive_analysis(csv_path)
    plt.show()
    
    
    # Print summary
    print_simulation_summary(data)
    
    # Save figures
    fig1.savefig('wildfire_analysis.png', dpi=300, bbox_inches='tight')
    print("Analysis complete! Figures saved as PNG files.")