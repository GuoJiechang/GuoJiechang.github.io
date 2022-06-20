---
layout: post
title:  "Leetcode Day1 Two Sum"
categories: Leetcode-Diary
---

# TWO SUM problem results

[Two_Sum](https://leetcode.com/problems/two-sum/)

![Result](/assets/images/two_sum.png)

## First Blood
Using "two loops" to solve the problem is not efficient.
```
{
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> indices(2);
        for(int i = 0; i < nums.size(); i++)
        {
            for(int j = 0; j < nums.size(); j++)
            {
                if((nums[i] + nums[j] == target) && (i != j))
                {
                    indices[0] = i;
                    indices[1] = j;
                    return indices;
                }
            }
        }
        return indices;
    }
};
}
```

## Double Kill
use std::map to record "key-value", in order to speed up the searching process.
key: the value of the element in nums array
value: the index of the element in nums array
```
{
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> key_value;

        //loop array nums, record value_index pair
        for(int i = 0; i < nums.size(); i++)
        {
            key_value[nums[i]] = i;
        }
        
        vector<int> indices(2);
        
        //loop array nums, find y
        for(int i = 0; i < nums.size(); i++)
        {
            int y = target - nums[i];
          
            if(key_value.find(y) != key_value.end())
            {
                int y_index = key_value[y];
                if(i != y_index)
                {
                    indices[0] = i;
                    indices[1] = y_index;
                    return indices;
                }
            }
        }
        return indices;
    }
};
}
```

# Triple Kill
some minor update
```
{
    class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> key_value;
        vector<int> indices(2);
        int i = 0;
        
        for(i = 0; i < nums.size(); i++)
        {
            key_value[nums[i]] = i;
        }
        
        //Declare and define y here, avoid define y multiple times inside the loop
        int y = 0;
        
        for(i = 0; i < nums.size(); i++)
        {
            y = target - nums[i];
          
            if(key_value.find(y) != key_value.end())
            {
                if(i != key_value[y])
                {
                    indices[0] = i;
                    indices[1] = key_value[y];
                    return indices;
                }
            }
        }
        return indices;
    }
};
}
```

# Quadra Kill
Using one loop is enough to find the pair
```
{
    class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> key_value;
        vector<int> indices(2);
        int i = 0;
        int y = 0;
        
        for(i = 0; i < nums.size(); i++)
        {
            y = target - nums[i];
            if(key_value.find(y) != key_value.end())
            {
                if(i != key_value[y])
                {
                    indices[0] = i;
                    indices[1] = key_value[y];
                    return indices;
                }
            }
            else
            {
                key_value[nums[i]] = i;
            }
        }
        
        return indices;
    }
};
}
```

# Penta Kill
Use unordered_map instead of map
![Differences between map and unordered map](/assets/images/map_vs_unordered_map.png)
See more detail in [The Image is from](https://www.geeksforgeeks.org/map-vs-unordered_map-c/)