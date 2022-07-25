---
layout: post
title: Leetcode 1480 Running Sum of 1d Array
date: 2022-07-25 14:54 -0500
categories: Leetcode-Diary
---
# Running Sum of 1d Array problem results

[Running Sum of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/)

![Result](/assets/images/running_sum_of_1d_array.png)

# First Blood
My first solution is so straightforward, that I created a new vector to save the running sum
```
{
    vector<int> runningSum(vector<int>& nums) 
    {
        vector<int> runningSum(nums.size());
        runningSum[0] = nums[0];
        for(int i = 1; i < nums.size(); i++)
        {
            runningSum[i] = runningSum[i-1] + nums[i];
        }
        
        return runningSum;
    }
}
```

# Double Kill
It could be better, I can return the result with the input array.
```
{
    vector<int> runningSum(vector<int>& nums) 
    {
       
        for(int i = 1; i < nums.size(); i++)
        {
            nums[i] += nums[i-1];
        }
        
        return nums;
    }
}
```

