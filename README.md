# Coupled partial differential equations solved by the improved self-adaptive PINNs

The ISA-PINNs achieve high precision for solving the 2D Burgers' equations, demonstrating a 2-3 orders of magnitude improvement compared to the original SA-PINNs and outperforming DeepXDE with the residual-based adaptive refinement method.

For Burgers' equation, the data used to generate the following video are shared on Google Drive.
https://drive.google.com/file/d/1hA2Lo914UjzwCzJkG5Gyy78kzFFAuiKK/view?usp=sharing

# The following two videos show the animations of the u(t,x,y) and v(t,x,y) solutions of Burgers' equation over the time domain.
[![YouTube Thumbnail](https://img.youtube.com/vi/_1qK4ejEQnw/hqdefault.jpg)](https://www.youtube.com/watch?v=_1qK4ejEQnw)

[![YouTube Thumbnail](https://img.youtube.com/vi/VCSHgUi42OU/sddefault.jpg)](https://www.youtube.com/watch?v=VCSHgUi42OU)

The Schr\"{o}dinger equation, decomposed into real and imaginary parts as coupled equations, is solved with high prediction accuracy by the model.

# The following two videos show the evolution of self-adaptive weights for solving the second-order rogue wave of the Schrödinger equation.
![Video](https://github.com/hucmwf/coupsa/blob/main/sa-sch2nd-animation.gif)
![Video](https://github.com/hucmwf/coupsa/blob/main/sa-sch2nd-v-animation.gif)

ISA-PINNs achieve 1–2 orders of magnitude higher precision than causal PINNs with transfer learning, addressing the limitations of prior SA-PINNs that struggled to produce reliable solutions for EB and TB dynamics.

# The following two videos show the evolution of self-adaptive weights for solving the EB equation, with and without data normalization.
![Video](https://github.com/hucmwf/coupsa/blob/main/EB-animation-norm.gif)
![Video](https://github.com/hucmwf/coupsa/blob/main/EB-animation.gif)

# Installation on RTX 4070 in Linux

PyTorch version 2.0.0 or higher

CUDA versions 11 or 12
