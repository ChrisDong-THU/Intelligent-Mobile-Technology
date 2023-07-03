from vision import Vision
from action import Action
import time

if __name__ == '__main__':
    vision = Vision()
    action = Action()
    while True:
        action.controlObs(vision)
        time.sleep(0.2)
