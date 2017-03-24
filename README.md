# Tutorial in Diffraction Imaging

Python version of the following tutorial, which is realized in Matlab.

http://xray.bmc.uu.se/~huldt/molbiofys/tutorial.pdf

## Contents

Chapter 1. Projection images [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Chapter%201.%20Projection%20images.ipynb)]

Chapter 2. Diffraction concepts [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Chapter%202.%20Diffraction%20concepts.ipynb)]

Chapter 3. Resolution [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Chapter%203.%20Resolution.ipynb)]

Chapter 4. A look at 3D reconstructions [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Chapter%204.%20A%20look%20at%203D%20reconstructions.ipynb)]

Chapter 5. Sampling, periodicity and crystals [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Chapter%205.%20Sampling%2C%20periodicity%20and%20crystals.ipynb)]

Chapter 6. Phase retrieval [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Chapter%206.%20Phase%20retrieval.ipynb)]

Appendix. Q&A [[ IPython nbviewer ](http://nbviewer.jupyter.org/github/wang-zy/Tutorial-in-Diffraction-Imaging-Python/blob/master/notebook/Q%26A.ipynb)]

## About Mayavi

Mayavi is a scientific data visualizer written in Python, which uses VTK and provides a GUI via Tkinter. In this tutorial, Mayavi is used for 3D data visualization, especially its mlab module.

### Installation of Mayavi

The latest version of Mayavi, called Mayavi2, is a component of the [ Enthought Canopy ](https://www.enthought.com/products/canopy/). For those who use [ anaconda ](https://www.continuum.io/why-anaconda) instead of Enthought Canopy, Mayavi can also be easily installed using conda (check the latest version avaiable before running it):

```
conda install -c anaconda mayavi=4.5.0
```

If you want to compile the files and install Mayavi manualy, see this [ official documentation ](http://docs.enthought.com/mayavi/mayavi/installation.html) for more details.

### Use Mayavi in Jupyter notebook

Mayavi provides two different ways of displaying in Jupyter notebook, static images and [ X3D ](https://www.x3dom.org/) elements, which produces fully interactive 3D scene.

The X3D feature is quite easy to use and has nice display outputs. The only requirement is to install the x3dom Javascript and CSS files locally, which is already shipped with Mayavi package. Simply run the following command should work.

```
jupyter nbextension install --py mayavi --user
```

Generating a static image is much more complicated than it looks. Off screen rendering needs to be enabled for this to work. More details about using Mayavi in Jupyter notebook and how to set off screen rendering can be found [ here ](http://docs.enthought.com/mayavi/mayavi/tips.html).
