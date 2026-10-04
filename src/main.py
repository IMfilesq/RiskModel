import hydra

from src.schemas.config import Config


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    print(cfg)


if __name__ == "__main__":
    main()
