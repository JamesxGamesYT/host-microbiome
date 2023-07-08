# TODO List
- [ ] Compare the smeared (sequentially modelled) matrix to the real simulations and see how long it takes to break down
    - Key questions: How long does transient behavior last?
- [ ] Analyze leading/secondary eigenvectors and eigenvector centrality of the smeared models
- [ ] More tests to break down the two classes more rigorously
- [ ] What fitness matrix parameters create complete dominance, and others only partial dominance based on the initial conditions? Play around with simulation

# Concerns:
- Lack of robustness - changes in the fitness matrix can lead to unpredictable changes in the smeared matrix
- Approximation - very close, but clearly not completely accurate
- Depedency on data - does the matrix only work on 100 timesteps? Does it do enough to capture the cycling dynamics around the corners of a frequency square? As opposed to the random sampling throughout that space? 