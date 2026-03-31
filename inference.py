import asyncio
import json
from env import Env

async def run_task(task_id: str):
    """Executes a full episode in the environment for a specific task."""
    print(f"\n--- Running Task: {task_id.upper()} ---")
    
    env = Env()
    observation = env.reset(task_id)
    
    print(f"Scenario: {observation.scenario[:80]}...")
    
    done = False
    total_reward = 0.0
    
    # Step 1 & 2: Run deliberation Rounds
    for round_idx in range(2):
        response = await env.step({"action_type": "step"})
        total_reward += response.reward
        print(f"Step {round_idx+1} [Reward: {response.reward:.2f}]: {response.info['status']}")
        
    # Step 3: Run Final Decision Agent
    response = await env.step({"action_type": "finalize"})
    total_reward += response.reward
    
    print(f"Decision: {response.info['final_decision']['final_decision'].upper()}")
    print(f"Final Reward: {response.reward:.2f} | Cumulative: {total_reward:.2f}")
    
    return float(total_reward)

async def main():
    tasks = ["easy", "medium", "hard"]
    results = {}
    
    # Running all tasks in sequence to avoid hitting AI rate limits too hard
    # while still enabling deterministic evaluation across the suite
    for task_id in tasks:
        try:
            score = await run_task(task_id)
            results[task_id] = score
        except Exception as e:
            print(f"Error executing task {task_id}: {e}")
            results[task_id] = 0.0
            
    print("\n" + "="*30)
    print("FINAL EVALUATION SCORES")
    print("="*30)
    for task_id, score in results.items():
        print(f"{task_id.ljust(10)}: {score:.4f}")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(main())
