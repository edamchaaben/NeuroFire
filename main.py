from agent import DoubleDQNAgent, ReplayBuffer
from fire_env import FireEnv
from helper import plot
import numpy as np

def train():
    env = FireEnv()
    agent = DoubleDQNAgent()
    
    scores = []
    mean_scores = []
    losses = []
    total_score = 0
    record = 0
    
    print("🔥 Starting NeuroFire Training (Double DQN)...")
    print("Press Ctrl+C to stop training and save the model.\n")
    
    try:
        while True:
        # Get old state
        state_old = agent.get_state(env)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = env.play_step(final_move)
        state_new = agent.get_state(env)

        # Store in Memory
        # convert final_move (one-hot) to index? No, our memory push expects raw
        # But wait, replay buffer usually stores (state, action, reward, next_state, done)
        # Our buffer implementation is generic.
        agent.memory.push(state_old, final_move, reward, state_new, done)

        # Train network
        loss = agent.train_step() # Trains on batch from memory

        if done:
            env.reset()
            agent.n_games += 1
            
            # Update target network every 5 episodes
            if agent.n_games % 5 == 0:
                agent.update_target_network()
            
            if score > record:
                record = score
                agent.save()
                print(f'🎉 New Record! Model saved.')

            print(f'Game {agent.n_games} Score {score} Record {record} Epsilon {agent.epsilon:.2f}')

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            
            # Only append loss if training actually happened
            if loss > 0:
                losses.append(loss)
            
            plot(scores, losses, mean_scores)

    except KeyboardInterrupt:
        print("\n\n⏸️  Training stopped by user.")
        print(f"Final Stats: Games={agent.n_games}, Best Score={record}")
        agent.save(filename='final_model.pth')
        print("✅ Final model saved as 'final_model.pth'")

if __name__ == '__main__':
    train()
