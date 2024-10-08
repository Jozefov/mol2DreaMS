import sys
import yaml
from mol2dreams.utils.parser import build_trainer_from_config



def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Build the trainer
    trainer = build_trainer_from_config(config)

    # Start training
    trainer.train()

    # Optionally test the model
    trainer.test()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)