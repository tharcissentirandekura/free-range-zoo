## Nov 3, 2025

### Error Log
```bash
(three12) ➜  configs git:(main) ✗ python3 run_wildfire.py 
Episode: 0.csv, Total steps: 15
Traceback (most recent call last):
  File "/Users/tharcisse/Desktop/AdamResearch/Codes/free-range-zoo/free_range_zoo/envs/wildfire/configs/run_wildfire.py", line 57, in <module>
    render("outputs/wildfire_logging_test_0/0.csv", render_mode="human", frame_rate=15)
  File "/Users/tharcisse/Desktop/AdamResearch/Codes/free-range-zoo/free_range_zoo/envs/wildfire/env/utils/rendering.py", line 583, in render
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

