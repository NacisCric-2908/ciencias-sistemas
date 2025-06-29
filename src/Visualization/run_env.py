import time
from prey_predator_Env import PreyPredatorEnv

def main():
    env = PreyPredatorEnv(render_mode="human")
    obs = env.reset()
    
    done = {agent: False for agent in env.agents}

    for step_num in range(200):  # 200 pasos máximos
        # Generar acciones aleatorias válidas
        actions = {
            agent: env.action_spaces[agent].sample()
            for agent in env.agents
            if not done[agent]
        }

        # Paso en el entorno
        try:
            obs, rewards, terminations, truncations, infos = env.step(actions)
            env.render()
        except Exception as e:
            print(f"[ERROR EN STEP {step_num}] {e}")
            break

        # Mostrar info útil
        print(f"Step {step_num}")
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("Terminations:", terminations)

        # Chequear si todos terminaron
        done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
        if all(done.values()):
            print("Todos los agentes terminaron el episodio.")
            break

        time.sleep(0.1)  # Visualmente legible

    env.close()

if __name__ == "__main__":
    main()
