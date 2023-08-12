import os

if __name__ == "__main__":
    curr_dict = os.getcwd()
    os.environ["TORCH_HOME"] = curr_dict + "/code/results/cache/xdg"
    os.environ["XDG_CACHE_HOME"] = curr_dict + "/code/results/cache/torch"
