import os
import re

result_dir = "./local/outputs/qps"

for exp in os.listdir(result_dir):
    exp_path = os.path.join(result_dir, exp)
    latencies = []
    with open(exp_path) as f:
        for line in f.readlines():
            pattern = r"(?<=Average latency: )\d+\.\d+"

            # Search for the pattern in the text
            match = re.search(pattern, line)

            # Extract the number if a match is found
            if match:
                number = match.group()
                latencies.append(number)
    
    print(exp)
    print(','.join(latencies))
