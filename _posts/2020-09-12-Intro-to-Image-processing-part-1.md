---
toc: true
layout: post
description: Image processing with OpenCV
categories: [learning]
comments: true
title:  "Image processing with OpenCV: Part 1"
---
# Image processing with OpenCV: Part 1

## Disclaimer

The notebook is prepared on initial repo: https://github.com/MakarovArtyom/OpenCV-Python-practices

**Acknowledgement**

The code from Jupyter notebooks follows [**Udacity Computer Vision nanodegree**](https://www.udacity.com/course/computer-vision-nanodegree--nd891) logic.<br>
More Computer Vision exercises can be found under Udacity CVND repo: https://github.com/udacity/CVND_Exercises

## Libraries


```python
# uncomment in case PyQt5 is not installed
#! pip install PyQt5
```


```python
import numpy as np
import matplotlib.pyplot as plt
# for image reading
import matplotlib.image as mpimg
import cv2

# makes image pop-up in interactive window
%matplotlib qt
```

## Image reading

Given an **RGB** image, let's read it using matplotlib (`mpimg`) .

> The RGB color model is an additive color model in which red, green, and blue light are added together in various ways to reproduce a broad array of colors. The name of the model comes from the initials of the three additive primary colors, red, green, and blue
<br>

Source: https://en.wikipedia.org/wiki/RGB_color_model


```python
image = mpimg.imread('images/robot_blue.jpg')
```


```python
print('Shape of the image:', image.shape)
```

    Shape of the image: (720, 1280, 3)


Per single image, we can retrieve the following info:
* `height = 720`
* `width = 1280`
* `number of channels = 3`

Later on, we will see that for many "classical" computer vision algorithms the color information is redundant. <br>
For solving problems like edge, corner or blob detection, the grayscal image is enough.<br>

Let's use `cvtColor` function to convert RGB image to gray.


```python
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img, cmap = 'gray')
```




    <matplotlib.image.AxesImage at 0x7fdd7c819828>




![png](https://github.com/MakarovArtyom/mlstudygroup/tree/master/images/output_11_1.png)



```python
print('Shape of the image after convertation:', gray_img.shape)
```

    Shape of the image after convertation: (720, 1280)


It's important to understand an image is just a matrix (tensor) with values, representing the **intensity** of pixels (from 0 to 255).<br>
Each pixel has it's own coordinates.


```python
print('Max pixel value:', gray_img.max())
print('Min pixel value:', gray_img.min())
```

    Max pixel value: 255
    Min pixel value: 0


Use the pair of (x,y) coordinates to access particular pixel's value.


```python
x = 45
y = 52

pixel_value = gray_img[x,y]
print(pixel_value)
```

    28


Note, it's possible to **manipulate pixel values** by scalar multiplication and augmentation in different formats.

Example: let's create a 6x6 image with random pixel values.


```python
pixel = abs(np.random.randn(6,6)) * 10
pixel.astype(int)
```




    array([[15,  8,  3, 14,  3,  3],
           [ 6,  9, 10,  7,  0,  5],
           [11, 18, 10,  4, 14,  5],
           [ 9,  6,  0,  1,  1, 16],
           [18,  9,  6, 29,  8,  7],
           [12,  0,  2,  0,  9, 12]])



Next, display the results with grayscale colormap.


```python
plt.matshow(pixel, cmap = 'gray')
```




    <matplotlib.image.AxesImage at 0x7fdd7acd96d8>




![]({{ site.baseurl }}/images/output_20_1.png "grayscale image")





```python

```
