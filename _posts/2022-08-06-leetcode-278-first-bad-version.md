---
layout: post
title: Leetcode 278 First Bad Version
date: 2022-08-06 12:22 -0500
categories: Leetcode-Diary
---
# First Bad Version problem results

[First Bad Version](https://leetcode.com/problems/first-bad-version/)

![Result](/assets/images/first_bad_version.png)

# First Blood
```
{
    // The API isBadVersion is defined for you.
// bool isBadVersion(int version);

class Solution {
public:
    int firstBadVersion(int n) {
        int left = 0;
        int right = n-1;
        while(left <= right)
        {
            int pivot = left + (right - left)/2;
            if(isBadVersion(pivot + 1))
            {
                //check if is first bad version
                if(isBadVersion(pivot))
                {
                    right = pivot - 1;
                }
                else
                {
                    return pivot + 1;
                }
            }
            else
            {
                left = pivot + 1;
            }
        }
        
        return -1;
        
    }
};
}
```