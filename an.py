import time
from tqdm import tqdm

for i in tqdm(range(10), desc="aaa"):
    time.sleep(0.1)