#!/usr/bin/env python
"""Debug pygame controller input."""

import pygame

pygame.init()
pygame.joystick.init()

print(f"Joysticks found: {pygame.joystick.get_count()}")

if pygame.joystick.get_count() == 0:
    print("No controller found!")
    exit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Name: {joystick.get_name()}")
print(f"Axes: {joystick.get_numaxes()}")
print(f"Buttons: {joystick.get_numbuttons()}")
print(f"Hats: {joystick.get_numhats()}")

print("\nMove sticks and press buttons. Ctrl+C to exit.\n")

clock = pygame.time.Clock()

try:
    while True:
        # MUST call this for joystick to update!
        pygame.event.pump()

        # Print all axes
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]

        # Print all buttons
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        # Format output
        axes_str = " ".join([f"A{i}:{v:+.2f}" for i, v in enumerate(axes)])
        buttons_pressed = [i for i, b in enumerate(buttons) if b]

        print(f"\r{axes_str} | Buttons: {buttons_pressed}       ", end="", flush=True)

        clock.tick(20)

except KeyboardInterrupt:
    print("\nDone!")
finally:
    pygame.quit()
