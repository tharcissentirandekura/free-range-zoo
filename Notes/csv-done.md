# interpreting done.csv and mapping it to the code using wildfire.py and many more to come

1. Pristine Equipment System

# Equipment: Equipment states: 0 (damaged) → 1 (intermediate) → 2 (pristine)
pristine = state.equipment == (self.equipment_states.shape[0] - 1)  # Equipment value 2

2. Fire State Transitions & Burnout Logic
# From CSV: fires go from positive values to -2 (burned out or no fire anymore/yet and this need to be analyzed together with internsity)
# Step 12: fires = "[[0, -2, 1], [-2, 0, -2], [-1, -2, 0]]"
# Step 12: burnouts = 1, putouts = 0

# Fire intensity reaches 4 → automatic burnout
- The code to check that I think explains it:      
    just_burned_out = self.fire_increase_transition(...)
    fire_rewards[just_burned_out] = self.reward_config.burnout_penalty
    - So, because there is burned out, we should apply pernalty

3. Suppressant Depletion & Refill Mechanics

# From CSV: suppressants go from [2.0, 2.0, 2.0] → [0.0, 0.0, 0.0]
# Action [-1, -1] = refill action
refills[agent_index] = agent_actions[:, 1] == -1
# When suppressant = 0, agents can't fight fires
has_suppressants = self._state.suppressants > 0

4. Dynamic Action Space Changes

# How action_maps change in CSV:
# Step 1: firefighter_1_action_map = [0]     # Can fight 1 fire which is fire 0
# Step 8: firefighter_1_action_map = []      # Can't fight any fires  
# Step 13: firefighter_1_action_map = [0]    # Can fight again

# This happens because the code is like this:
checks = torch.logical_and(in_range, has_suppressants): Determined by being this task being in the range and the suppressant being sufficient enough

5. Fire Spread Mechanics

# New fires appear: Step 7 vs Step 8
# Step 7: fires = "[[0, -2, -1], [-2, 0, -2], [-1, 2, 0]]"  # 1 fire at (2,1)
# Step 8: fires = "[[0, -2, 1], [-2, 0, -2], [1, 2, 0]]"    # Fire spread to (0,2) and (2,0)

self._state = self.fire_spread_transition(state=self._state, randomness_source=field_randomness[2]) 

6. Reward System Components

# Multiple reward sources:
rewards[agent_name][bad_users] = self.reward_config.bad_attack_penalty  # Bad actions
rewards[agent] += fire_rewards_per_batch                                # Fire rewards
rewards[agent][newly_terminated] += termination_reward                  # Episode completion

# Termination reward decreases with burnouts:
termination_penalty = self.reward_config.termination_kappa * torch.log(self.num_burnouts + 1.0)


7. Environment Termination Conditions
# When do we end the episode:
fires_are_out = self._state.fires.flatten(start_dim=1).max(dim=1)[0] <= 0

if self.stochastic_config.fire_fuel:
    depleted_fuel = self._state.fuel.sum(dim=(1, 2)) <= 0
    batch_is_dead = torch.logical_and(depleted_fuel, fires_are_out)
else:
    batch_is_dead = fires_are_out


8. Stochastic vs Deterministic Elements
# We have many configurable randomness sources which applies to equipment, fire, suppressant, environment grid:
stochastic_repair, stochastic_degrade, critical_error
stochastic_decrease, stochastic_increase, stochastic_burnouts  
fire_fuel, suppressant_decrease, suppressant_refill

