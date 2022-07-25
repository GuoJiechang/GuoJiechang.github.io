---
layout: post
title: Leetcode 0724 Find Pivot Index
date: 2022-07-25 14:52 -0500
categories: Leetcode-Diary
---
# Find Pivot Index problem results

[Find Pivot Index](https://leetcode.com/problems/find-pivot-index/)

![Result](/assets/images/find_pivot_index.png)

The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right.

This problem is related to the running sum of 1d array.

# First Blood
I used a silly brute force method but failed to be accepted. 
For each element in the vector, I calculated the left sum and right sum to check the equality.
```
{
    int pivotIndex(vector<int>& nums) {
        int left_sum = 0;
        int right_sum = 0;
        for(int i = 0; i < nums.size(); i++)
        {
            //calculate left sum
            for(int j = 0; j < i; j++)
            {
                left_sum += nums[j];
            }
            
            //calculate right sum
            for(int k = i+1; k < nums.size(); k++)
            {
                right_sum += nums[k];
            }
            
            if(left_sum == right_sum)
            {
                return i;
            }
            
            left_sum = 0;
            right_sum = 0;
        }
        
        return -1;
    }
}
```
# Double Kill
My second try was calculating the sum of the given array, for each element in the array, calculate the left sum, then right sum = sum - left_sum - nums[i].
This passed the test but still with a bad performance.

```
{
int pivotIndex(vector<int>& nums) {
        int left_sum = 0;
        int right_sum = 0;
        int sum = 0;
        //calculate sum
        for(int i = 0; i < nums.size(); i++)
        {
             sum += nums[i];
        }
        
        for(int i = 0; i < nums.size(); i++)
        {
            //calculate left sum
            for(int j = 0; j < i; j++)
            {
                left_sum += nums[j];
            }
            
            right_sum = sum - left_sum - nums[i];
            
            if(left_sum == right_sum)
            {
                return i;
            }
            
            left_sum = 0;
            right_sum = 0;
        }
        
        return -1;
    }
}
```

# Triple Kill
After checking the Discussion, I got an acceptable submission.
By calculating the left_sum as the running sum of the given array.
```
{
     int pivotIndex(vector<int>& nums) {
        int left_sum = 0;
        int right_sum = 0;
        int sum = 0;
        //calculate sum
        for(int i = 0; i < nums.size(); i++)
        {
             sum += nums[i];
        }
        
        for(int i = 0; i < nums.size(); i++)
        {
            right_sum = sum - left_sum - nums[i];
            
            if(left_sum == right_sum)
            {
                return i;
            }
            
            //calculate running sum
            left_sum += nums[i];
        }
        
        return -1;
    }

}
```
