from pathlib import Path


def read_env_file(env_path):
    values = {}
    env_path = Path(env_path)
    if not env_path.exists():
        print(f"No .env file found at: {env_path}")
        return values

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values
