import sys
import subprocess
import numpy as np

from simulation2x2x2 import run_sim
from visualization import generate_invasion_network, eval_matrix

def animate(v0, iterations, fitness_array, matrix, system="2x2x2"):
    data, meanfitness, training_data = run_sim(v0=v0, iterations=iterations, fitness_array=fitness_array, format="invasion")
    for i in range(iterations):
        print(f"{i}/{iterations}")
        holobiont_frequencies = [data[x][i] for x in range(len(data))]
        print(holobiont_frequencies)
        generate_invasion_network(matrix=matrix, index=f"animation_{str(i).zfill(3)}", holobiont_frequencies=holobiont_frequencies)
    subprocess.run(["ffmpeg", "-pattern_type", "glob", "-i", 'animation_frames/*.png', \
  "-c:v", "libx264", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "animation.mp4"])
#     subprocess.run(["ffmpeg", "-framerate", "4", "-pattern_type", "glob", "-i", 'animation_frames/*.png', \
#   "-c:v", "libx264", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "animation.mp4"])
#     subprocess.run(["ffmpeg", "-framerate", "10", "-pattern_type", "glob", "-i", 'animation_frames/*.png', \
#   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "animation.mp4"])
    
if __name__ == "__main__":
    index = 238
    if len(sys.argv) > 1:
        index = sys.argv[1]
    with open(f"2x2x2simulations/{index}/W.txt", "r") as f:
        W = eval(f.read())
    with open(f"2x2x2simulations/{index}/eig.txt", "r") as f:
        data = f.read()
    x = np.array(eval("["+",".join(data[data.find("Invasion matrix:"):].split("\n")[1:])+"]")) 
    # runplot("invasion", W, index=index)
    v0 = [1, 1, 1, 1]
    # v0 = [10, 1, 10, 1]
    # v0 = [1, 2, 1, 2]
    animate(v0, 50, fitness_array=W, matrix=x)