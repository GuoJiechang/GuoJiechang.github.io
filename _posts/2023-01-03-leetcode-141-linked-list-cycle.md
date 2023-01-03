---
layout: post
title: Leetcode 141 Linked List Cycle
date: 2023-01-03 15:42 -0600
categories: Leetcode-Diary
youtubeId: pKO9UjSeLew
---
[141-Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)
Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

# method 1
Store visited node in hash-table
```
{
    class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head)
            return false;

        unordered_set<ListNode*> visited;
        ListNode* node = head;
        visited.insert(node);
        while(node->next)
        {
            if(visited.find(node->next)!=visited.end())
            {
                return true;
            }
            node = node->next;
            visited.insert(node);
        }

        return false;
    }
};
}
```

# method 2
This method is genius. Called "Floyd's Tortoise and Hare", an algorithm using two pointers with different speeds to check a cycle. The idea is simple, image Hare and Tortoise are racing, they must meet each other at some point if they are running in a circle.
And this video is so funny. 
{% include youtubePlayer.html id=page.youtubeId %}

```
{
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head)
            return false;

        ListNode* slow = head;
        ListNode* fast = head;
        
        while(slow && fast)
        {
            //slow go one step
            slow = slow->next;

            //fast go two step
            fast = fast->next;
            if(fast)
            {
                fast = fast->next;
            }

            if(!slow || !fast)
            {
                return false;
            }

            if(slow == fast)
            {
                return true;
            }

        }

        return false;
    }
};
}
```