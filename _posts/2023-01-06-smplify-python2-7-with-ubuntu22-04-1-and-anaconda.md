---
layout: post
title: Smplify python2.7 with Ubuntu22.04.1 and Anaconda
date: 2023-01-06 13:56 -0600
categories: Technical-Notes
---

According to the README Getting Started, after extracting the code, we should get LSP data, but the LSP dataset cannot be downloaded from the given path, I found a source from the other website. Get if from here [LSP](https://drive.google.com/file/d/1MhC1v-D_8UZwkuuUONCMwvKQj1r5Jf5o/view?usp=sharing)


The other dependencies are the same as the SMPL.

Follow the README, created a symbolic link to LSP images, and a symlink to the SMPL model from the SMPL package.
Before running the fit_3d.py, the name of the model should be changed.

line 665-669, the model path should be changed to the name of SMPL models.

Finally, smplify is good to go.

![Result](/assets/images/smplify_result.png)
