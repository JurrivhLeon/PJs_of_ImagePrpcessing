# Project 3: Spatial Filters
Follow this instruction to reproduce the result of my experiment. Make sure this repo is under the directory ${MY_DIR}, and ensure you have installed libraries listed below in your running environment:
```
numpy==1.20.1
opencv-python=4.6.0.66
matplotlib==3.3.4
```

Now create a new terminal in an IDE (e.g. PyCharm, Visual Studio Code, etc.),
and check if the current working directory is ${MY_DIR}. If not, change it to this directory.

To smooth an image (using Gaussian & median filters), run this command:

```
python smooth.py -i ${IMAGE_DIR} [-s ${KERNEL_SIZE} -v ${VARIANCE}]
```

Change the content of ```${...}```, and content in brackets is optional. For example:

```
python smooth.py -i images/suomi.jpg -s 3 -v 9.0
```

Example:<br>




To sharpen an image with Laplacian filter, run this command:

```
python sharpen_laplacian.py -i ${IMAGE_DIR} [-w ${WEIGHT}]
```

Change the content of ${...}, and content in brackets is optional. For example:

```
python sharpen_laplacian.py -i images/suomi.jpg -w 1.0
```

Example:<br>

<p align="center">
  <img src='images/suomi.jpg' width='250'/> &nbsp;&nbsp;&nbsp;
  <img src='images/suomi_7x7_81.0_gaussian_smoothed.jpg' width='250'/> &nbsp;&nbsp;&nbsp;
  <img src='images/suomi_5x5_median_smoothed.jpg' width='250'/>
</p>


To sharpen an image with highboost, run this command:

```
python sharpen_highboost.py -i ${IMAGE_DIR} [-k ${WEIGHT}]
```

Change the content of ${...}, and content in brackets is optional. For example:

```
python sharpen_highboost.py -i images/suomi.jpg -k 2.0
```

Example:<br>

