import temp_plot_util as tpu
import json

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()



with open(args.filepath, 'r') as file:
    cases:list[dict] = json.load(file)["cases"]
print(f"Number of mismatches: {len(cases)}")
toroid = cases[0]["toroid"]
toroids = []
toroids_used = [] # index of toroid corresponding to each case
for case in cases:
    tor = case['toroid']
    duplicate = False
    for i, t in enumerate(toroids):
        if tor["rab"][0] == t["rab"][0]: 
            duplicate = True
            toroids_used.append(i)
            break
    if not duplicate:
        toroids_used.append(len(toroids)) 
        toroids.append(tor)
print(f"Number of toroids: {len(toroids)}")
# print(toroids_used)
rays = [it["ray"] for it in cases]
tpu.plot_tor_rays_standalone(toroids, rays, tors_used=toroids_used)



inf_cases:list = [] #cases where one thinks there's no real root
inf_ten = [] # alg1010 thought no good root
inf_fer = [] # ferrari thought no good root
dif_cases:list = [] #cases where both are real, but disagreee

for i, case in enumerate(cases):
    fer = case['ferrari']
    ten = case['alg1010']
    if fer[0] is None:
        inf_cases.append(i)
        inf_fer.append(i)
    elif ten[0] is None:
        inf_cases.append(i)
        inf_ten.append(i)
    else:
        dif_cases.append(i)

plt.hist([inf_ten, inf_fer, dif_cases], 50, histtype='bar', density=True, stacked=True, label=["1010 inf", 'ferrari inf', 'both finite'])
plt.xlabel('Calling order (approximate)')
plt.ylabel('count')
plt.legend()
plt.show()

# tpu.plot_tor_rays_standalone(toroids, [rays[i] for i in inf_ten if i > 3500])