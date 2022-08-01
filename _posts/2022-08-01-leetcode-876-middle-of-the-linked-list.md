---
layout: post
title: Leetcode 876 Middle of the Linked List
date: 2022-08-01 15:12 -0500
categories: Leetcode-Diary
---
# Middle of the Linked List problem results

[Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)

![Result](/assets/images/middle_of_the_linked_list.png)

# First Blood
Saving each node in a vector, calculate the middle index, and get the node in the vector.
```
{
    ListNode* middleNode(ListNode* head) {
        vector<ListNode*> node_list;
        node_list.push_back(head);
        while(head->next != nullptr)
        {
            head = head->next;
            node_list.push_back(head);
        }
        
        int index = node_list.size()/2;
       
        return node_list[index];
    }
}
```