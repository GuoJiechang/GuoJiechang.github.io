---
layout: post
title: Leetcode 0004 Median of Two Sorted Arrays
date: 2022-06-24 11:19 -0500
categories: Leetcode-Diary
---
# Median of Two Sorted Arrays problem results

[Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

![Result](/assets/images/median_of_two_sorted_array.png)

# First Blood
My first solution is so brute force and the run time complexity is not O(log(m+n))。But I think this solution is more straightforward to come up with. It could be better because there's no need to sort the whole array, it can be stopped when half of the elements are sorted.
In other words, stop looping when the median is found.

```
{
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
       
        int num1_size = nums1.size();
        int num2_size = nums2.size();
        int size = num1_size + num2_size;
       
        int vec[size];

        int i = size-1;
        int m = num1_size-1;
        int n = num2_size-1;

        while(i >= 0)
        {
            if(m<0)
            {
                vec[i] = nums2[n];
                n--;
            }
            else if(n<0)
            {
                vec[i] = nums1[m];
                m--;
            }
            else
            {
                if(nums1[m] > nums2[n])
                {
                    vec[i] = nums1[m];
                    m--;
                }
                else
                {
                    vec[i] = nums2[n];
                    n--;
                }
            }
            i--;
        }

        if(size%2 == 0)
        {
            return (vec[size/2-1]+vec[size/2])/2.0;
        }
        else
        {
            return vec[size/2];
        }

        return 0.0;
    }
}
```

# Double Kill
After I checked with the Discussion part, I learned that this problem is a typical "Binary Search" problem for the keywords "Sorted Array" and "O(log())". 

The main idea for this solution is the concept of the median. The median is the middle number of an odd amount of number sorted array, and half of the middle pair numbers' sum when it is an even amount of number array. The median number can divide the array into two parts, the elements in the left part of the array are less or equal to the median, while the elements in the right part are large or equal to the median. 

Imagine that if we merge the given two sorted arrays into one sorted array, the elements on the left of the median will consist of the number from both given arrays. So instead of merging two arrays, what we can do is find the right position to divide the given two arrays. In other words, we can divide each sorted array into two parts - the elements in the left part of the array are less than the median and the right part of the array is big than the median. Combining the left parts of the two sorted arrays will form the left part of the merged array, and so is the right part.

The question becomes finding the right position to divide the array. 
Since we know we need to find the middle number to divide the array into the left and right parts. The amount of numbers in the left and right parts is the same. We only need to find the cut position of one array, and then calculate the cut position of the other array. And we will do the binary search to find the correct cut position on the smaller array to speed up the process.

Let m be the size of nums1, n be the size of nums2,
len = m + n is the total amount of numbers in the two arrays.

If len is odd, we assume that the median is on the left part, so there's one more number on the left than the right.
If len is even, the median is half of the sum of the max number in the left part and the min number in the right part, and the number of element in the left and right is the same.

left_side_size = (len + 1)/2
if len = 3, left_side_size = (3+1)/2 = 2, if len = 2, left_size_size = 1

Let i be the number of elements in the left part of nums1
j be the number of elements in the right part of nums2
j = left_side_size - i;

The binary search is to find the correct cut position or the correct size of the left part.

We start at the middle of the nums1 array:
low = 0;
high = m;
i = (low+high)/2; // (0+m)/2
j = left_side_size - i;


To check if we find the right position:

//i is the size of left part, the index of max number of left part i-1;
//the edge case is that, if there's no element in the nums1 array's left part, the max number will be set to INT_MIN
num1_left_max = (i == 0) ? INT_MIN: nums1[i-1];

//the index of min number of the right part is i
//the edge case is that, the elements in the nums1 are all in the left side of the merged array, the min number of the right part will be set to INT_MAX
num1_right_min =  (i == m_size) ? INT_MAX: nums1[i];

num2_left_max = (j == 0) ? INT_MIN: nums2[j-1];
num2_right_min = (j == n_size) ? INT_MAX: nums2[j];


remember that the elements in the left part is less than the right part
to check if we find the correct cut position:
num1_left_max <= num2_right_min && num2_left_max <= num1_right_min
if so, we can return the median:
if the len is even, the median will be (std::max(num1_left_max,num2_left_max)+std::min(num1_right_min,num2_right_min))/2.0
if the len is odd, the median will be (std::max(num1_left_max,num2_left_max))

if not, we do the binary search, 
if num1_left_max > num2_right_min, the num1_left_max should be in the right side of the nums1, in other words, we need to contract the size of left side of the nums1 array, the possible cut position should be on the left of current cut postion.
let high = i - 1;

if num2_left_max > num1_right_min, num2_left_max should be in the right side fo nums2，we need to expand the size of left side of the nums1 array, so that the left side of nums2 will be contract.
let low = i + 1;




```
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m_size = nums1.size();
        int n_size = nums2.size();
        if(n_size < m_size)
        {
            return findMedianSortedArrays(nums2,nums1);
        }

        int size = m_size + n_size;
        
        //pointer low high for binary search
        int low = 0;
        int high = m_size;

        //i for cut nums1, j for cut nums2
        int i = 0;
        int j = 0;

        //do binary search on nums1
        while(high>=low)
        {
            i = (low+high)/2;
            j = (size+1)/2 - i;

            int num1_left_max = (i == 0) ? INT_MIN: nums1[i-1];
            int num1_right_min =  (i == m_size) ? INT_MAX: nums1[i];

            int num2_left_max = (j == 0) ? INT_MIN: nums2[j-1];
            int num2_right_min = (j == n_size) ? INT_MAX: nums2[j];

            //find right i and j to divide two arrays
            if((num1_left_max <= num2_right_min) && (num2_left_max <= num1_right_min))
            {
                if((m_size+n_size)%2 == 0)
                {
                    return (std::max(num1_left_max,num2_left_max)+std::min(num1_right_min,num2_right_min))/2.0;
                }
                else
                {
                    return (double)(std::max(num1_left_max,num2_left_max));
                }
            }
            //left part of num1 should be contract
            else if(num1_left_max > num2_right_min)
            {
                high = i-1;
            }
            else//left part of num1 should be expand
            {
                low = i+1;
            }
        }

        return 0.0;
    }
```




