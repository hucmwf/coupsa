# Solving coupled partial differential equations using improved PINNs with a self-adaptive mechanism

The ISA-PINNs achieve high precision for solving the 2D Burgers' equations, demonstrating a 2-3 orders of magnitude improvement compared to the original SA-PINNs and outperforming DeepXDE with the residual-based adaptive refinement method.

For the 2D Burgers' equation, the data file for achieving prediction solutions is shared on Google Drive due to file size limitations.
The exact and learned results are combined to generate two videos demonstrating the high precision.

https://drive.google.com/file/d/1hA2Lo914UjzwCzJkG5Gyy78kzFFAuiKK/view?usp=sharing

# The following two videos show the animations of the u(t,x,y) and v(t,x,y) solutions of Burgers' equation over the time domain.
[![YouTube Thumbnail](https://img.youtube.com/vi/_1qK4ejEQnw/hqdefault.jpg)](https://www.youtube.com/watch?v=_1qK4ejEQnw)

[![YouTube Thumbnail](https://img.youtube.com/vi/VCSHgUi42OU/sddefault.jpg)](https://www.youtube.com/watch?v=VCSHgUi42OU)

The Schr\"{o}dinger equation, decomposed into real and imaginary parts as coupled equations, is solved with high prediction accuracy by the model.

# The following two videos show the evolution of self-adaptive weights for solving the first-order rogue wave of the Schrödinger equation.
![Video](https://github.com/hucmwf/coupsa/blob/main/sa-sch1st-animation.gif)
![Video](https://github.com/hucmwf/coupsa/blob/main/sa-sch1st-v-animation.gif)

# The following two videos show the evolution of self-adaptive weights for solving the second-order rogue wave of the Schrödinger equation.
![Video](https://github.com/hucmwf/coupsa/blob/main/sa-sch2nd-animation.gif)
![Video](https://github.com/hucmwf/coupsa/blob/main/sa-sch2nd-v-animation.gif)

The number of epochs in the Adam optimizer is set sufficiently high to ensure the convergence of self-adaptive weights.

# Installation on RTX 4070 in Linux

PyTorch version 2.0.0 or higher

CUDA versions 11 or 12
