---
layout: post
title: Training on FLIC Dataset with Limited Memory in Tensorflow
date: 2023-03-28 10:38 -0500
categories: CV_Notes
tag: CV
---
# Introduction
I attempted to run the code from the [joint_cnn_mrf](https://github.com/max-andr/joint-cnn-mrf), which is the first and only implementation of the paper "Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation." However, I encountered several challenges due to the version difference between Tensorflow 1.x and Tensorflow 2.x, as well as memory limitations. The code requires loading the entire training and testing datasets into memory, which proved to be a challenge for my computer with only 32GB of RAM. I even tried to migrate it to Colab and upgraded to Colab Pro, but the data processing consumed more than 50GB of memory and used up the compute unit immediately. It seems the authors must have had a powerful computer even in 2018. As recommended by the authors, I have decided to implement the code from scratch.


```python
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import cv2
import math
from scipy.io import loadmat
import imageio
```

# Download Data

To download file from google drive, share the file by link and set the access permission to everyone who knows the link. For example, my shared link is https://drive.google.com/file/d/16o7zXFl2PsFHXf9OVEc7OVJrQZ6ru1Mv/view?usp=share_link, copy the magic string '16o7zXFl2PsFHXf9OVEc7OVJrQZ6ru1Mv'


```python
!gdown 16o7zXFl2PsFHXf9OVEc7OVJrQZ6ru1Mv
```

    Downloading...
    From: https://drive.google.com/uc?id=16o7zXFl2PsFHXf9OVEc7OVJrQZ6ru1Mv
    To: /content/FLIC.zip
    100% 300M/300M [00:02<00:00, 138MB/s]


Unzip the file


```python
!unzip '/content/FLIC.zip'
```

# Generate Ground Truth Heatmaps

The code is adapted from https://github.com/max-andr/joint-cnn-mrf


```python
#load the annotation data from the annotation file path
data_FLIC = loadmat('/content/FLIC/examples.mat')
data_FLIC = data_FLIC['examples'][0]

#path to the image file
images_dir = '/content/FLIC/images/'

```


```python
#define joint ids
joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose']  # , 'leye', 'reye',
dict = {'lsho':   0, 'lelb': 1, 'lwri': 2, 'rsho': 3, 'relb': 4, 'rwri': 5, 'lhip': 6,
            'lkne':   7, 'lank': 8, 'rhip': 9, 'rkne': 10, 'rank': 11, 'leye': 12, 'reye': 13,
            'lear':   14, 'rear': 15, 'nose': 16, 'msho': 17, 'mhip': 18, 'mear': 19, 'mtorso': 20,
            'mluarm': 21, 'mruarm': 22, 'mllarm': 23, 'mrlarm': 24, 'mluleg': 25, 'mruleg': 26,
            'mllleg': 27, 'torso': 28}

#get train and test data index
is_train = [data_FLIC[i][7][0, 0] for i in range(len(data_FLIC))]
is_train = np.array(is_train)
train_index = list(np.where(is_train == 1))[0]
test_index = list(np.array(np.where(is_train == 0)))[0]
print('# train indices:', len(train_index), '  # test indices:', len(test_index))
```

    # train indices: 3987   # test indices: 1016


Define the Gaussian Kernel for generating the heatmaps.


```python
orig_h, orig_w = 480, 720
coefs = np.array([[1, 2, 1]], dtype=np.float32) / 4  # maximizes performance
kernel = coefs.T @ coefs
temp = round((len(kernel) - 1) / 2)
pad = 5  # use padding to avoid the exceeding of the boundary
```

Create a list to store train filenames and also a list to store test filenames.
In the meantime, generate groud-truth heatmap for train and test dataset.


```python
train_filenames = []
test_filenames = []
train_heatmaps = []
test_heatmaps = []
for filenames, hmaps, indices in zip([train_filenames,test_filenames], [train_heatmaps,test_heatmaps],[train_index, test_index]):
  for i in indices:
    filenames.append(data_FLIC[i][3][0])
    flic_coords = data_FLIC[i][2]
    heatmaps = []
    torso = (flic_coords[:, dict['lsho']] + flic_coords[:, dict['rhip']] + flic_coords[:, dict['rsho']] +
                     flic_coords[:, dict['lhip']]) / 4
    flic_coords[:, dict['torso']] = torso
    for joint in joint_ids + ['torso']:
        coords = np.copy(flic_coords[:, dict[joint]])
        # there are some annotation that are outside of the image (annotators did a great job!)
        coords[0], coords[1] = max(min(coords[1], orig_h), 0), max(min(coords[0], orig_w), 0)

        coords /= 8
        heat_map = np.zeros([60, 90], dtype=np.float32)
        heat_map = np.lib.pad(heat_map, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
        coords = coords + pad
        h1_k, h2_k = int(coords[0] - temp), int(coords[0] + temp + 1)
        w1_k, w2_k = int(coords[1] - temp), int(coords[1] + temp + 1)
        heat_map[h1_k:h2_k, w1_k:w2_k] = kernel
        heat_map = heat_map[pad:pad + 60, pad:pad + 90]
        heatmaps.append(heat_map)
        hmap = np.stack(heatmaps, axis=2)
    hmaps.append(hmap)
```

Check the length and shape of the data created.


```python
print(len(train_filenames))
print(len(test_filenames))
print(len(train_heatmaps))
print(train_heatmaps[0].shape)
print(len(test_heatmaps))
print(test_heatmaps[0].shape)
```

    3987
    1016
    3987
    (60, 90, 10)
    1016
    (60, 90, 10)


We can save the numpy file for latter usage. We don't have to do the data process every time.


```python
np.save('x_train_filenames.npy',train_filenames)
np.save('x_test_filenames.npy',test_filenames)
np.save('y_train.npy',train_heatmaps)
np.save('y_test.npy',test_heatmaps)
```

# Create Custom Data Generator for Training in Tensorflow

Create a custom batch data generator, each time the generator will help us to load the images from the disk.


```python
class Flic_Generator(keras.utils.Sequence):
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size

  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    #load the images and normalized it, in our case, the model takes the original size of the image as input
    #no need to resize
    x_array = np.array([imageio.v2.imread('/content/FLIC/images/' + str(file_name)) for file_name in batch_x])/255
    y_array = np.array(batch_y)
    return x_array,y_array
```

Create train and test data generator


```python
batch_size = 32

training_batch_generator = Flic_Generator(train_filenames,train_heatmaps,batch_size)
test_batch_generator = Flic_Generator(test_filenames,test_heatmaps,batch_size)
```


```python
print(training_batch_generator.image_filenames[0])
```

    12-oclock-high-special-edition-00006361.jpg


Show one image


```python
img = imageio.v2.imread('/content/FLIC/images/' + str(training_batch_generator.image_filenames[0]))
import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis('off')
plt.show()
```


    
![png](/assets/images/FLIC_Load_to_memory/output_26_0.png)
    


Show the heatmaps


```python
heatmaps = training_batch_generator.labels[0]
heatmaps.shape
```




    (60, 90, 10)




```python
#transpose the heatmaps array
heatmaps = np.transpose(heatmaps,(2,0,1))
heatmaps.shape
```




    (10, 60, 90)




```python
combined_heatmap = heatmaps[0]
for i in range(1,10):
  combined_heatmap += heatmaps[i]

plt.imshow(combined_heatmap)
plt.axis('off')
plt.show()
```


    
![png](/assets/images/FLIC_Load_to_memory/output_30_0.png)
    


# Train and Test the model

We will use fit_generator to train the model, and predict_generator to test the test dataset. Tensorflow will call the "__getitem__" in the background for each iteration.


```python
history = model.fit_generator(generator=training_batch_generator,epochs=30)
predicted = model.predict_generator(test_batch_generator)

```
