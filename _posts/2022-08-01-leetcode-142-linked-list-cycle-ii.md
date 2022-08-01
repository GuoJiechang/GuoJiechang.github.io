---
layout: post
title: Leetcode 142 Linked List Cycle II
date: 2022-08-01 15:12 -0500
categories: Leetcode-Diary
---
# Linked List Cycle II problem results

[Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

![Result](/assets/images/linked_list_cycle_ii.png)


# First Blood
Saving the visited node in an unordered_set, stop when meeting the same node.
```
{
        ListNode *detectCycle(ListNode *head) {
        if(head == nullptr)
            return head;
        
        unordered_set<ListNode*> visited_node;
        
        while(head->next != nullptr && visited_node.find(head->next) == visited_node.end())
        {
            visited_node.insert(head);
            head = head->next;
        }
        
        return head->next;
    }
}
```