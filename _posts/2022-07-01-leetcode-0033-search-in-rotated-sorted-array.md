---
layout: post
title: Leetcode 0033 Search in Rotated Sorted Array
date: 2022-07-01 10:40 -0500
categories: Leetcode-Diary
---
# Search in Rotated Sorted Array problem results

[Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

![Result](/assets/images/search_in_rotated_sorted_array.png)

# First Blood

I got a wrong answer because I forget to check if the pivot number is equal to the target.

# Double Kill

First, find the pivot index, if no pivot index was found, the array is not rotated, do a binary search on the whole array.
Then check the target with the pivot number, to decide to do a binary search on which part of the array.

```
{
    class Solution {
public:
    int search(vector<int>& nums, int target) {
              int pivot = nums[0];
        int size = nums.size();
        if(size == 1)
        {
            return target == pivot ? 0 : -1;
        }

        int i = 0;
        //find pivot index
        for(i = 0; i < size; i++)
        {
            if(nums[i] < pivot)
            {
                break;
            }
        }
        int low = 0;
        int high = 0;
        //the array is not rotated
        if(i == size)
        {
            //Do binary search on whole array
            low = 0;
            high = size -1;
            
        }
        else
        {
            //pivot index is i-1;
            //check target on which side of pivot index
            if(target >= pivot)
            {
                low = 0;
                high = i-1;
            }
            else
            {
                low = i;
                high = size-1;
            }
        }

        int j = 0;
        while(low<=high)
        {
            j = (low + high)/2;

            if(nums[j] == target)
            {
                return j;
            }
            else if(nums[j] < target)
            {
                low = j + 1;
            }
            else
            {
                high = j - 1;
            }
        }

        return -1;  
    }
};
}
```

