---
layout: post
title: Leetcode 0205 Isomorphic Strings
date: 2022-07-28 11:52 -0500
categories: Leetcode-Diary
---
# Isomorphic String problem results

[Isomorphic String](https://leetcode.com/problems/isomorphic-string/)

![Result](/assets/images/isomorphic_string.png)

I can find an excuse this time for my multiple failures. I had a fever that time after getting the booster of Covid19. My mind is not clear.

# First Blood
My first accepted solution is straightforward.
s1=s,t1=t
Mapping each character in "s" to "t" and converting s1 with the map.
Mapping each character in "t" to "s" and converting t1 with the map.
compare s1 with t, t1 with s.
Simply test two ways.
```
{
    bool isIsomorphic(string s, string t) {
        int size = s.length();
        map<char,char> replaced_char_s;
        map<char,char> replaced_char_t;
        
        string s1 = s;
        string t1 = t;
        
        for(int i = 0; i < size; i++)
        {
            if(replaced_char_s.find(s[i]) == replaced_char_s.end())
            {
               replaced_char_s[s[i]] = t[i];
            }

            if(replaced_char_t.find(t[i]) == replaced_char_t.end())
            {
               replaced_char_t[t[i]] = s[i];
            }
            
            if(replaced_char_s.find(s[i]) != replaced_char_s.end())
            {
               s1[i] = replaced_char_s[s[i]];
            }
            
            if(replaced_char_t.find(t[i]) != replaced_char_t.end())
            {
               t1[i] = replaced_char_t[t[i]];
            }
        }
        
        
        if(s1 == t && t1 == s)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}
```

# Double Kill
My second try was smarter.
There's no need for me to convert the string to check the equality, what I can do is simply map and check if the two-way mapping is correct.
If no mapping exists in either of the map, map it.
If a mapping exists in the s_to_t map, check if the existing mapping is equal to the current t[i], if not return false and vice versa.

```
{
     bool isIsomorphic(string s, string t) {
        int size = s.length();
        map<char,char> s_2_t;
        map<char,char> t_2_s;
        
        for(int i = 0; i < size; i++)
        {
            if(s_2_t.find(s[i]) == s_2_t.end() && t_2_s.find(t[i]) == t_2_s.end())
            {
               s_2_t[s[i]] = t[i];
               t_2_s[t[i]] = s[i];
            }
            else if(s_2_t.find(s[i]) != s_2_t.end() && t_2_s.find(t[i]) == t_2_s.end())
            {
                if(s_2_t[s[i]] != t[i])
                    return false;
            }
            else if(s_2_t.find(s[i]) == s_2_t.end() && t_2_s.find(t[i]) != t_2_s.end())
            {   if(t_2_s[t[i]] != s[i])
                    return false;
            }
            else
            {
                 if(s_2_t[s[i]] != t[i] || t_2_s[t[i]] != s[i])
                    return false;
            }
        }
           
        return true;
    }
}
```

# Triple Kill
The last solution I tried is to transform both string into a template string, and check if the template string is the same.
For each character in the given string, we replace it with the index of that character's first occurrence in the string.

```
{
    bool isIsomorphic(string s, string t) {
        unordered_map<char,int> s_2_format;
        unordered_map<char,int> t_2_format;
        
        for(int i = 0; i < s.length(); i++)
        {
           if(s_2_format.find(s[i]) == s_2_format.end())
           {
               s_2_format[s[i]] = s_2_format.size();
           }
           s[i] = s_2_format[s[i]];
           
           if(t_2_format.find(t[i]) == t_2_format.end())
           {
               t_2_format[t[i]] = t_2_format.size();
           }
          
           t[i] = t_2_format[t[i]];
            
           if(s[i] != t[i])
           {
                return false;
           }
        }
        
           
        return true;
    }
}
```

