---
layout: post
title: Leetcode 0021 Merge Two Sorted Lists
date: 2022-08-01 15:11 -0500
categories: Leetcode-Diary
---
# Merge Two Sorted Lists problem results

[Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

![Result](/assets/images/merge_two_sorted_lists.png)


I should try a recursive method later.


# First Blood
Iterative method.
```
{
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(list1 == nullptr)
        {
            return list2;
        }
        
        if(list2 == nullptr)
        {
            return list1;
        }
        
        
        ListNode* result = nullptr;
        if(list1->val <= list2->val)
        {
            result = list1;
            list1 = list1->next;
        }
        else
        {
            result = list2;
            list2 = list2->next;
        }
        
        ListNode* temp = result;
        
        
        while(list1 != nullptr && list2 != nullptr)
        {
            if(list1->val <= list2->val)
            {
                temp->next = list1;
                list1 = list1->next;
            }
            else
            {
                temp->next = list2;
                list2 = list2->next;
            }
            temp = temp->next;
        }
        
        if(list1 != nullptr)
        {
            temp->next = list1;
        }
        
        if(list2 != nullptr)
        {
            temp->next = list2;
        }
        
        return result;
    }
}
```
