from controllers.main_controller import MainController
from utils.logging_config import setup_logging


setup_logging()

if __name__ == "__main__":
    controller = MainController()
    controller.run()
