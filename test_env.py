from openenv import OpenEnv, AgentAction, Metrics

def main():
    print("Testing OpenEnv...")
    env = OpenEnv(max_rounds=2)
    state, info = env.reset()
    print(f"Scenario: {state.scenario}")
    
    agents = ["Profit", "PR", "Ethics"]
    
    for round_num in range(2):
        print(f"\n--- Round {round_num + 1} ---")
        for expected_agent in agents:
            current = env.get_expected_agent()
            print(f"Expected: {current}")
            
            action = AgentAction(
                agent_name=current,
                action_content=f"This is {current}'s recommendation for round {round_num+1}.",
                metrics_update=Metrics(expected_profit=10.0) if current == "Profit" else None,
            )
            
            state, reward, done, info = env.step(action)
            print(f"Action processed. Done: {done}. Next agent: {info['next_expected_agent']}")

    print(f"\n--- Final Decision ---")
    current = env.get_expected_agent()
    print(f"Expected: {current}")
    
    final_action = AgentAction(
        agent_name=current,
        action_content="This is the final decision."
    )
    state, reward, done, info = env.step(final_action)
    print(f"Action processed. Done: {done}. Reward: {reward}")
    
    print("\n--- Final State History ---")
    for a in state.history:
        print(f"{a.agent_name}: {a.action_content}")
        
    print("\n--- Final Metrics ---")
    print(state.metrics)

if __name__ == "__main__":
    main()
