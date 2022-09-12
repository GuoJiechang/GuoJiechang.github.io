---
layout: post
title: Leetcode 704 Binary Search
date: 2022-08-06 12:24 -0500
categories: Leetcode-Diary
---
# Binary Search problem results

[Binary Search](https://leetcode.com/problems/binary-search/)
Given an array of integer nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If the target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

![Result](/assets/images/binaray_search.png)

Keywords for binary search: ordered array, O(log n) runtime complexity.

Algorithm:
left pointer points to the first element in the array, and the right pointer points to the last element in the array. Check if the target matches the middle element of left and right (temp = left + (right-left)/2), if the target is less than the temp element, the target must be on the right side of the temp element, so move the left pointer to the next element of temp. else if the target is bigger than the temp element, the target must be on the left side of the temp element, so move the right pointer to the prior element of temp, until we find target == temp element or the left pointer meet the right pointer.


# First Blood
```
{
    class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size()-1;
        while(left<=right)
        {
            int temp = (left+right)/2;
            if(nums[temp] == target)
            {
                return temp;
            }
            else if(nums[temp] < target)
            {
                left = temp + 1;
            }
            else if(nums[temp] > target)
            {
                right = temp - 1;
            }
        }
        
        return -1;
    }
};
}
```

# Double Kill
```
{
    class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size()-1;
        while(left<=right)
        {
            //(left+right)/2 should be care, if right and left is too big
            int temp = left + (right - left)/2;
            if(nums[temp] == target)
            {
                return temp;
            }
            else if(nums[temp] < target)
            {
                left = temp + 1;
            }
            else if(nums[temp] > target)
            {
                right = temp - 1;
            }
        }
        
        return -1;
    }
};
}
```