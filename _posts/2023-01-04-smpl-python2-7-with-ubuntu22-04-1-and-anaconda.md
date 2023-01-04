---
layout: post
title: SMPL python2.7 with Ubuntu22.04.1 and Anaconda
date: 2023-01-04 13:43 -0600
categories: Technical-Notes
---
After some effort in trying to run SMPL python on macOS and failing, I decided to use Ubuntu.
Install Ubuntu dual system with Windows on my PC in my lab is another story.
Here I wrote down the process for future usage. I referenced a lot from this post in Chinese is very detailed, however, need some updates on some detail. [](https://blog.csdn.net/weixin_42145554/article/details/111381447)

## Environment:
Ubuntu22.04.1
Anaconda 
python2.7

## Step1
Create a new environment through anaconda with python 2 and activate
```
conda create -n smpl python=2
conda activate smpl
```

## Step2
Install required dependencies
```
pip install numpy
pip install scipy
pip install chumpy
```
The trick for installing the correct version of opencv is using the following command. After running render_smpl.py found out the opencv I installed is not correct. It has to be 4.2.0.32, the last version supporting python2.7. I also tried pip, but not correct.

```
pip2 install opencv-python==4.2.0.32
```

For the installation of opendr, is much easier than the thing on macOS.

```
sudo apt install libosmesa6-dev
sudo apt-get install build-essential
sudo apt-get install libgl1-meas-dev
sudo apt-get install libglu1-meas-dev
sudo apt-get install freeglut3-dev
pip install opendr
```

## Step3
Set the python package path for smpl.
1. Go to the python installed path, mine is anaconda3/envs/smpl/lib/python2.7/site-packages
2. Create .pth file
```
touch xxx.pth
```
3. Add the absolute path of smpl folder and smpl_websuer folder which contain __init__.py file.
```
/home/jiechang/Documents/SMPL/SMPL/smpl
/home/jiechang/Documents/SMPL/SMPL/smpl/smpl_webuser
```

## Step4
Test hello_smpl.py
need to change the model's name in the script If successful there will be a hello_smpl.obj model in the path.
line 48
```
m = load_model( '../../models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl' )
```

## Step5
Test render_smpl.py.
Still need to change the name of the model accordingly.

Here, another things need to fix is,
if encounter 
ImportError: /home/jiechang/anaconda3/envs/smpl/lib/python2.7/site-packages/../../libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /lib/x84_64-linux-gnu/libLLVM-13.so.1)

Solution:
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```
The version of 3.4.30 should be shown, the lib is exist in our system, but the anaconda don't know, we need to create a shortcut to let anaconda find the lib.
```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/jiechang/anaconda3/envs/smpl/bin/../lib/libstdc++.so.6
```

Finally get hello smpl to work. So I can go and learn Smplify.
![Result](/assets/images/render_smpl.png)