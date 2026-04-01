import asyncio
from env import Env
from multi_agent import get_multi_agent_action

# Step 12: Limit Steps (3-5)
MAX_STEPS = 3 

async def run_episode(task_id: str) -> float:
    """Executes a full episode simulation looping through agent steps."""
    env = Env()
    
    # Reset Environment
    observation = env.reset(task_id)
    print(f"\n--- Running Task: {task_id.upper()} ---")
    print(f"Scenario: {observation['scenario']}")
    
    total_reward = 0.0
    
    # Loop max steps
    for step_num in range(MAX_STEPS):
        # 1. State goes to Multi-Agent cluster, combines votes, returns Action
        action = await get_multi_agent_action(env.state())
        
        # 2. Step Environment
        next_state, reward, done, info = env.step(action)
        print(f"  Step {step_num+1}: Chosen [{action}] -> Step Reward: {reward:.4f}")
        
        # Track Rewards structurally
        total_reward += reward
        
        if done:
            break
            
    # Compute Final Average over iterations
    return total_reward / MAX_STEPS

async def main():
    print("Testing Simulation Pipeline...")
    tasks = ["easy", "medium", "hard"]
    results = {}
    
    # Inference loop all tasks
    for task_id in tasks:
        score = await run_episode(task_id)
        results[task_id] = score
        
    # Step 10 target format output
    print("\n" + "="*20)
    print("RESULT SCORES")
    print("="*20)
    
    for idx, (t_id, score) in enumerate(results.items(), 1):
        print(f"Task {idx}: {score:.2f}")
    
    final_avg = sum(results.values()) / len(results) if results else 0.0
    print(f"Final Avg: {final_avg:.2f}")
    print("="*20)

if __name__ == "__main__":
    asyncio.run(main())
