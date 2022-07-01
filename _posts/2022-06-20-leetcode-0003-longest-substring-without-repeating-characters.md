---
layout: post
title: Leetcode 0003 Longest Substring Without Repeating Characters
date: 2022-06-20 11:04 -0500
categories: Leetcode-Diary
---
# Longest Substring Without Repeating Characters problem results

[Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

![Result](/assets/images/longest_substring_without_repeating_character.png)

When I was working, I occasionally heard about the common interview question about the substring, but never bother to think about it. Because, in my daily work, these sorts of questions may not show up and bother me.
But this is really interesting. I find out that by solving these questions, I really can learn something interesting and also enhance my logical ability.

## First Blood
By myself, I only came up with the brute force approach. 
At first, I tried to record every substring and then compare each other's lengths to get the size of the longest substring. This approach undoubtedly failed to be accepted.
Finally, I got an acceptable answer which is bad of course.
```
{
class Solution {
public:
    static int lengthOfLongestSubstring(string s) {
        
         if(s.size() == 0)
            return 0;

        int j = 0;
        int count = 0;
        int max = 0;

        set<char> substr;

        for(int i = 0; i < s.size(); i++)
        {
            substr.insert(s[i]);
           
            j = i + 1;
            while(j < s.size())
            {
                if(substr.find(s[j]) != substr.end())
                {
                    break;
                }
                else
                {
                    substr.insert(s[j]);
                }
                j++;
            }

            if(max < substr.size())
                max = substr.size();
                
            substr.clear();
        }
       
        
        return max;
    }
};

}
```

## Double Kill
Then, I read the "Solution" to learn the idea of "Sliding Window". I was like: "Oh, I see, here it is".
After three attempts I finally got an acceptance.
```
{class Solution {
public:
    static int lengthOfLongestSubstring(string s) {
        
        if(s.size() <= 1)
            return s.size();

        //sliding window
        int win_left = 0;
        int win_right = 0;
        
        int max = 0;
        int cur_str_size = 0;

        //store visited char
        unordered_set<char> substr;

        while(win_right < s.size() && win_left < s.size())
        {
           char r = s[win_right];
           if(substr.find(r) != substr.end())
           {
               substr.erase(s[win_left]);
               win_left++;
           }
           else
           {
                substr.insert(r);
                win_right++;
                cur_str_size = win_right - win_left;
                max = max <  cur_str_size? cur_str_size: max;
           }
        
        }
        
        return max;
    }
};}
```

## Sliding Window Algorithm
Here are some notes about the "Sliding Window Algorithm" when solving the longest substring problem.

Traversing the given string by a changeable size of the window.

First move the right pointer of the window which expands the size of the sliding window, until meeting a repeating character, calculate the size of the current substring, and record the max length.

Then move the left pointer of the window which contracts the size of the sliding window, until there is no repeating character in the current substring.

The traversing will be stopped until the pointer reaches the end of the given string.
