---
layout: post
title: Leetcode 409 Longest Palindrome
date: 2022-08-06 12:28 -0500
categories: Leetcode-Diary
---
# Longest Palindrome problem results

[Longest Palindrome](https://leetcode.com/problems/longest-palindrome/)
Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

Example 1:
Input: s = "abccccdd"
Output: 7
Explanation: One longest palindrome that can be built is "dccaccd", whose length is 7.

Example 2:
Input: s = "a"
Output: 1
Explanation: The longest palindrome that can be built is "a", whose length is 1.

![Result](/assets/images/longest_palindrome.png)

# First Blood
A palindrome string can be formed by several pairs of a character and one single character which is optional.
So we can simply calculate the number of each character to check how many pairs and if there is a single character that can be served as the palindrome top.

```
{
    class Solution {
public:
    int longestPalindrome(string s) {
        unordered_map<char,int> counter;

        //count the number of each character occured in the given string
        for(int i =0; i<s.size(); i++)
        {
            if(counter.find(s[i]) != counter.end())
            {
                counter[s[i]]++;
            }
            else
            {
                counter[s[i]] = 1;
            }
        }
        
        int paired_count = 0;
        int single_count = 0;

        //traverse each pair in the map, to check each character's number is odd or even.
        for(unordered_map<char,int>::iterator iter = counter.begin(); iter != counter.end(); iter++)
        {
            int val = iter->second;
            if(val%2 == 0)
            {
                paired_count += val;
            }
            else
            {
                paired_count += val-1;
                single_count++;
            }
        }
        
        if(single_count != 0)
            return paired_count+1;
        else
            return paired_count;
    }
};
}
```

# Double Kill
```
{
class Solution {
public:
    int longestPalindrome(string s) {
        int counter[128] = {};
        for(int i =0; i<s.size(); i++)
        {
            counter[s[i]]++;
        }
        
        int paired_count = 0;
        int single_count = 0;
        for(int i = 0; i < 128; i++)
        {
            int val = counter[i];
            if(val%2 == 0)
            {
                paired_count += val;
            }
            else
            {
                paired_count += val-1;
                single_count++;
            }
        }
        
        if(single_count != 0)
            return paired_count+1;
        else
            return paired_count;
    }
};
}
```

# Triple Kill
```
{
class Solution {
public:
    int longestPalindrome(string s) {
        int counter[58] = {0};
        for(char c : s)
        {
            counter[c -'A']++;
        }
        
        int paired_count = 0;
        bool is_odd = false;
        for(int val : counter)
        {
            if(val%2 == 0)
            {
                paired_count += val;
            }
            else
            {
                paired_count += val-1;
                is_odd = true;
            }
        }
        
        return is_odd ? paired_count+1 : paired_count;
    }
};
}
```