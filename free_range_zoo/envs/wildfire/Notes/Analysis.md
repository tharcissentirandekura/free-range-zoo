## Fires: Straight forward
- **0**: agent position, can't catch fire
- **1**: small fire
- **2**: Medium/large fire
- **-1**: No fire currently but if it were to start would be small
- **-2**: No fire currently but if it were to start would be small
    - Usually fire intensity will be 0
    - Same with fuel
- **-2**: Can mean burned out and this happens when
    - fire intensity is 4 
    - So, it is helpful to look those two parts of the log files 
## Agent Action:
    - The action is a list of what action an agent takes at step i. This list consists of 2 things:
        - At index 0, the value is a fire index (usually a fire index since all fires are indexed from 0) from the agent action map
        - And at index 1, the value represent the power (for now not sure if this power if agent power suppressant power,..) but it looks like it is 0 or -1 always. I have looked in the rendering file that the dev implemented, the conditions only wether the power is 0 or -1, for -1 meaning the agent is a NOOP (no operation) and 0 indicating that the agent is active.
    - The action of the agent is pulled from agent's action map which is a list of all possible actions for the agent (details below).


## Action map:
- List of actions available for the agent
- **[]** : No action available
- It can have **[0,1,2]**: Are those fire index? It turns out to be yes
- The File: `free_range_zoo/envs/wildfire/env/wildfire.py`
- This file shows which agent can fight which fires: inrange of the fire task position
- Numbers represent fire task indices - NOT fire positions directly
- Empty list **[]** = No valid actions available for that agent
- **[0, 1, 2]** = Agent can fight fire tasks indexed 0, 1, and 2
    - We don't know which environment this fire is, just index: Bad naming and they should name fires based on ids instead so that each new fire comes with its unique id (my thoughts)
- All fires in environment get indexed sequentially and this actions are pulled from the observation map since action map is a subset of all actions avaible (my thinking because I haven't validated it but I am confident). An agent observes the environment and returns a map of actions observed which can be the entire tasks avaiable in its environment, and among those tasks, the agent filters out tasks it can operate on based on its ability, location, and the results is teh action map, and finally from this action map, the agent choose 1 task to operate on at step i (I guess based on its policy).
    ```
    fires = self._state.fires > 0
    fire_positions = fires.nonzero(as_tuple=False)
    ```

So, in our csv log:
**Step 1:**
```
fires = "[[0, 2, -1], [-2, 0, -2], [-1, 2, 0]]"
firefighter_1_action_map = [0]
firefighter_2_action_map = [0, 1]
firefighter_3_action_map = [1]
```

**Means:**
- There are 2 active fires at positions (0,1) and (2,1)
- So, indexed as :
    - Task 0: (0,1)
    - Task 1: (2,1)
- For agents:
    - fighter1: can only fight task  at index 0 at (0,1) from action map
    - fighter2: can fight task at index 0 and task 1 at (0,1) and (2,1) from action map
    - fighter3: can only fight task at index 1 at (2,1) from action map

- Action map depends on these functions:
    - **range change**: in_range_check.chebyshev()
        - File `free_range_zoo/envs/wildfire/env/utils/in_range_check.py` to check if the agent is close enough to fight
    - **suppressant check**: does agent have suppressant available/suffient enough to fight such fire in the range
    - **equipment bonus**: Does the equipment bonus allows the agent to attack extra range or distance? It turns out to be true as far as we know
    - **observation map** : The action map is like a subset for observation map (not too sure)

- The task in range check is calculated using chebyshev or euclidean distance but in our case we are using chebyshev which is fast and better to calculate the neighbor distance for instance.
- We then update the agent action map:
    - Using the logical and for `in_range` and `has_suppressant` `torch.logical_and(in_range, has_suppressants)`

- So, this is what fires the agent can fight rather than what fires the agent can see which is opposite to observation map.

## Observation map:
- List of observed entities in the environment
    - Each list at step i indicates action to be taken for step i + 1 (next step)
    - Are these observed the agents or fires indices
    - each number is agent id or agent index (Look into the file)
    - File: `free_range_zoo/envs/wildfire/env/wildfire.py`
    - Very similar to action map but instead of actions available, it shows all fires available in the environment
    - So, the agent can observe all fires regardless of ability to fight them
    - Uses `task_indices_nested` for all tasks

- In csv logs:
    ```
    firefighter_1_action_map = [0]
    firefighter_1_observation_map = [0, 1]
    firefighter_2_action_map = [0, 1] 
    firefighter_2_observation_map = [0, 1]
    firefighter_3_action_map = [1]
    firefighter_3_observation_map = [0, 1]
    ```

**Means:**
- All agents can observe both fires (tasks 0 and 1)
- fighter1 can see both fires/tasks 0 and 1 but only fight fire 0
- fighter2 can see both fires and can fight fires available
- fighter3 can see both fires but can only fight fire 1

- Observation map depends on function:
    - `update_actions()` and `update_observations`
    - For actions map:
        - `self.agent_action_mapping[agent] = batchwise_indices` which filters actions 
    - For observation map:
        - `self.agent_observation_mapping[agent] = task_indices_nested` which takes all observed tasks for the agent  
    - It defines `self.observations = {}` on line 606 and this defines a Tensor for:
        ```
        'self': agent_observations[:, agent_index],
        'others': agent_observations[:, agent_mask][:, :, observation_mask],
        'tasks': fire_observations
        ```
        for agent in self.agents

- So, this is what fires the agent can see rather than what fires the agent can fight.

## Equipments:
- File `free_range_zoo/envs/wildfire/env/transitions/equipment.py`
- **0**: damaged
- **1**: intermediate which is when the equipment is neither damaged nor pristine
- **2**: pristine equipment
- Depending on which equipment an agent has, there can be a bonus to that so that the agent is enabled to fight fires in longer distance. So, equipment bonuses can extend attack range of an agent.
- in `wildfire.py` line 515 in the `update_actions` function and calculates the true range for each of the agents in each environment

- **From our csv**:
    - equipment = [2, 2, 2] : All agents have pristine equipment
    - equipment = [1, 1, 1]: All agents have intermediate equipment
    - equipment = [0, 0, 0] → All agents have damaged equipment, they need to repair
    - It looks like equipment degrade probability of 1, so each time equipment 

- **Calculate the new modified equipment conditions**
    - if the equipment is pristine, it can degrade or have a critical error
    - if the equipment is damaged, it can be repaired
    - if the equipment is in an intermediate state, it can degrade

- **Equipment repairs**:
    - Equipment randomly repairs, degrades, or has critical failures and creates dynamic agent capabilities that change overtime
    - Agents must adapt to changing equipment conditions? I guess

- So, the equipment affects which fire an agent can fight

## Issues spotted: Details in the progress.md