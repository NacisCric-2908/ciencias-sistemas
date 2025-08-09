import os
import json
import pygame

"""
This file allows you to see the results of the episode in a graphical and interactive way.
"""

def show_episode_outcome(prey_reward, predator_reward, project_root, final_infos):
    pygame.init()

    # === Adjust window ===
    screen_width, screen_height = 800, 540
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Episode Outcome")

    # === Colors ===
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (30, 144, 255)
    ORANGE = (255, 140, 0)
    RED = (220, 20, 60)
    GREEN = (0, 200, 0)
    GRAY = (100, 100, 100)

    # === Load images ===
    prey_img_path = os.path.join(project_root, "src", "assets", "images", "prey", "Prey_0.png")
    predator_img_path = os.path.join(project_root, "src", "assets", "images", "predator", "Predator_0.png")
    prey_img = pygame.transform.scale(pygame.image.load(prey_img_path), (150, 150))
    predator_img = pygame.transform.scale(pygame.image.load(predator_img_path), (150, 150))

    # === Fonts ===
    title_font = pygame.font.SysFont(None, 60)
    result_font = pygame.font.SysFont(None, 48)
    reward_font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 28)
    button_font = pygame.font.SysFont(None, 28)

    # === Determine outcome ===
    if final_infos and isinstance(final_infos, dict) and "prey" in final_infos:
        termination_reason = final_infos["prey"].get("termination_reason", "unknown")
        prey_status = predator_status = termination_reason
    else:
        prey_status = predator_status = "unknown"

    if prey_status == "caught":
        prey_result = ("Loser", RED)
        predator_result = ("Winner", GREEN)
    elif prey_status == "trapped":
        prey_result = ("Loser", RED)
        predator_result = ("Loser", RED)
    elif prey_status == "escaped":
        prey_result = ("Winner", GREEN)
        predator_result = ("Loser", RED)
    else:
        prey_result = predator_result = ("Unknown", BLACK)

    # === Draw background ===
    screen.fill(WHITE)

    # === Título ===
    title_text = title_font.render("EPISODE OUTCOME", True, BLACK)
    screen.blit(title_text, ((screen_width - title_text.get_width()) // 2, 25))

    # === Resultados ===
    left_center_x = screen_width // 4
    right_center_x = 3 * screen_width // 4
    result_y = 90
    img_y = 130
    reward_y = 290
    chart_y = 340

    screen.blit(result_font.render(prey_result[0], True, prey_result[1]), (left_center_x - 60, result_y))
    screen.blit(result_font.render(predator_result[0], True, predator_result[1]), (right_center_x - 60, result_y))

    screen.blit(prey_img, (left_center_x - 75, img_y))
    screen.blit(predator_img, (right_center_x - 75, img_y))

    screen.blit(reward_font.render(f"Reward: {prey_reward:.2f}", True, BLUE), (left_center_x - 75, reward_y))
    screen.blit(reward_font.render(f"Reward: {predator_reward:.2f}", True, ORANGE), (right_center_x - 75, reward_y))

    # === Gráfica de barras ===
    termination_json_path = os.path.join(project_root, "jsons", "termination_stats.json")
    if os.path.exists(termination_json_path):
        try:
            with open(termination_json_path, "r") as f:
                stats = json.load(f)

            labels = ["caught", "escaped", "trapped"]
            colors = [ORANGE, BLUE, GRAY]
            values = [stats.get(label, 0) for label in labels]

            bar_width = 60
            spacing = 100
            start_x = screen_width // 2 - (len(labels) * spacing) // 2
            max_height = 100
            max_value = max(values) if max(values) > 0 else 1

            for i, (label, value) in enumerate(zip(labels, values)):
                height = int((value / max_value) * max_height)
                x = start_x + i * spacing
                y = chart_y + (max_height - height)
                pygame.draw.rect(screen, colors[i], (x, y, bar_width, height))
                label_text = small_font.render(label, True, BLACK)
                value_text = small_font.render(str(value), True, BLACK)
                screen.blit(label_text, (x + (bar_width - label_text.get_width()) // 2, chart_y + max_height + 5))
                screen.blit(value_text, (x + (bar_width - value_text.get_width()) // 2, y - 20))

        except Exception as e:
            print(f"⚠️ Error al cargar termination_stats.json: {e}")
    else:
        print("❌ termination_stats.json no encontrado")

    # === Botones ===
    otro_ep_button = pygame.Rect(140, screen_height - 50, 200, 40)
    finalizar_button = pygame.Rect(screen_width - 340, screen_height - 50, 200, 40)

    pygame.draw.rect(screen, (0, 150, 0), otro_ep_button)
    pygame.draw.rect(screen, (150, 0, 0), finalizar_button)

    otro_text = button_font.render("Otro episodio", True, WHITE)
    final_text = button_font.render("Finalizar", True, WHITE)

    screen.blit(otro_text, (otro_ep_button.x + (200 - otro_text.get_width()) // 2,
                            otro_ep_button.y + (40 - otro_text.get_height()) // 2))
    screen.blit(final_text, (finalizar_button.x + (200 - final_text.get_width()) // 2,
                             finalizar_button.y + (40 - final_text.get_height()) // 2))

    pygame.display.flip()

    # === Event Loop ===
    running = True
    action = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                action = "exit"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if otro_ep_button.collidepoint(mouse_pos):
                    running = False
                    action = "next"
                elif finalizar_button.collidepoint(mouse_pos):
                    running = False
                    action = "exit"

    pygame.quit()
    return action
