## Nov 3, 2025

### Error Log
```bash
Traceback (most recent call last):
  File "free-range-zoo/free_range_zoo/envs/wildfire/configs/run_wildfire.py", line 57, in <module>
    render("outputs/wildfire_logging_test_0/0.csv", render_mode="human", frame_rate=15)
  File "free-range-zoo/free_range_zoo/envs/wildfire/env/utils/rendering.py", line 583, in render
    fire_row = fire_to_supress['y']
               ~~~~~~~~~~~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
(three12) ➜  configs git:(main) ✗ 
```

**Cause:** fire_to_suppress is None and we can't get the key since it is not subscriptable

**Notes:**
- Cases: The problem is not related to rendering, but the log files not being the correct things we want to pass to the render
- Try to run the different log file and it may not crash
---

## Nov 8, 2025

### Running agents with Strongest baseline and Weakest baseline

**Error:**
```python
self.observation, self.t_mapping = observation #not running 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 1)
```

**Steps:** I printed the agent observation and it printed a tensor dict with self,others and tasks as keys
- in wildfire.py: agent_action_mapping exists set to a tupple

**Fix:** instead of passing observation[agent_name[0]], change it to observation[agent_name] which is a tuple.


## Dec 12, 2025

## Logging inconsistency with rendering

1. In the wildfire logging, we build the agent action by indexing the tasks but the agent action contains the fire index in this agent action map and also the power, representing whether or not the fire should fight or be a NOOP.

So we need a way to trace which fire is really being mapped to an agent and then when we render, .... [For me to understand better how to deal with this]

## January 12, 2026: Updating error logs + proposed fixes

- Running the most recent changes from main repository
Unresolved issues:
1. quickstart docs:
**Error 1**
  ```bash
  Traceback (most recent call last):
    File "./free-range-zoo/free_range_zoo/envs/wildfire/configs/run.py", line 33, in <module>
      agent_name:agents[agent_name].act(action_space = env.action_space(agent_name))
                ~~~~~~^^^^^^^^^^^^
   KeyError: 'firefighter_3'
  ```
  **Proposed Solution**
  The environment configuration has 3 agents initially but in the quickstart, the code is defining only two agents, so the mapping ended up searching for third agent but it doesn't exists in the agents keys.
   
   **Fix**: in the agents map definition, add a third agent of any kind
    e.g: env.agents[2]: StrongBaseline(agent_name = "firefighter_3", parallel_envs = 1)

**Error 2**
```bash
FileExistsError: The logging output directory already exists. Set override_initialization_check or rename.
```
This is not really a serious issue but the docs should mention this or provide easy fix solution. Other future solution would be to allow a `dynamic logging` where each time we run the environment, it generates a new log file instead of overwitting the existing one and render that.

**Fix**: Add `override_initialization_check = True` in the env definiton
  ```bash
  env = wildfire_v0.parallel_env(
      max_steps = 700,
      parallel_envs = 1,
      configuration = wildfire_configuration,
      device=torch.device('cpu'),
      log_directory = log_dir,
      override_initialization_check = True
  )
  ```

**Error 3: New error**
```bash
Traceback (most recent call last):
  File "/Users/tharcisse/Desktop/AdamResearch/Codes/codebase/free-range-zoo/free_range_zoo/envs/wildfire/configs/run.py", line 41, in <module>
    observations, rewards, terminations, truncations, infos = env.step(agent_actions)

  ...
TypeError: list indices must be integers or slices, not tuple
```
From the print, the agent actions seems to be a nested list of 2. 

**Fix**: This error is happening because the `agent_actions` is a python list. So, `agent_actions[:,1]` is not a valid tensor slicing. We need to convert agent_actions to a tensor slicing/indexing in the `wildfire.py` in the `step_environment` in the loop.

```bash 
  if isinstance(agent_actions,list):
      agent_actions = torch.tensor(agent_actions,device=self.device)
  ```


**Error 4**
```bash
  File "free-range-zoo/free_range_zoo/utils/logging_handlers.py", line 88, in log_environment
    new_cols[f'{agent}_action'] = [str(action) for action in actions[agent].cpu().tolist()]
                                                             ^^^^^^^^^^^^^^^^^^
AttributeError: 'list' object has no attribute 'cpu'
```

**Fix**: actions[agent] is a python list for some agents, not a PyTorch tensor. The code tries to call .cpu() on a list, which fails. So, similar case as the above previous error, we need to convert it to a tensor. There may be a better ways to globally update the action[agent] to be a tensor but I tried different ways but only one worked. So, feel free to try your way.


```bash
  #update the for loop in the logging_handlers.py on on line 88 or around that
    for agent in agents:
        action = actions[agent] # initialize the agent action
        if isinstance(action, list):
            action = torch.tensor(action) # convert to tensor if it is not already
        new_cols[f'{agent}_action'] = [str(action) for action in action.cpu().tolist()]
        new_cols[f'{agent}_rewards'] = rewards[agent].cpu().tolist()
        new_cols['step'] = num_moves.cpu().tolist()
        new_cols['complete'] = finished.cpu().tolist()
```

**Error 5: Not enough values to unpack**

```bash
  File "free-range-zoo/free_range_zoo/envs/wildfire/configs/run.py", line 34, in <module>
    agent.observe(observations[agent_name][0])  # Policy observation 

    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  self.observation, self.t_mapping = observation
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ValueError: not enough values to unpack (expected 2, got 1)
```
The issue if because the agent observation has two items.
1. The TensorDict with free fields: self, others,tasks defining what the agent observed from itself, other agents and environment tasks.
2. The dication of agent_action_mapping defining what actions are assigned to an agent based on its fighting power.

So, in the baselines, when we do `self.observation, self.t_mapping = observation` is an error because observation is only one value not two. 

**Fix**: in the quickstart file, in the while loop:
  ```bash
      for agent_name, agent in agents.items():
          print(f"agent Observation: {observations[agent_name]}")
          agent.observe(observations[agent_name][0])  # Policy observation
          # replace:
          agent.observe(observations[agent_name][0]) 
          # with:
           agent.observe(observations[agent_name]) # to pass both agent tensor dic and agent action mapping


      # So, new updated code will be:
      for agent_name, agent in agents.items():
        agent.observe(observations[agent_name])  # Policy observation
  ```