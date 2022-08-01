---
layout: post
title: Leetcode 0206 Reverse Linked List
date: 2022-08-01 15:11 -0500
categories: Leetcode-Diary
---
# Reverse Linked List problem results

[Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

![Result](/assets/images/reverse_linked_list.png)

# First Blood
Iterative Method.

```
{
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr)
            return head;
        
        ListNode* curr = head->next;
        ListNode* temp = curr->next;

        curr->next = head;
        head->next = nullptr;
        head = curr;
        curr = temp;
        
        while(curr != nullptr)
        {
            temp = curr->next;
            curr->next = head;
            head = curr;
            curr = temp;
        }
        
        return head;
    }
}
```
# Double Kill
Iterative Method with simpler code.

```
{
    ListNode* reverseList(ListNode* head) {
        ListNode* temp = nullptr;
        ListNode* prev = nullptr;

        while(head != nullptr)
        {
            temp = head->next;
            head->next = prev;
            prev = head;
            head = temp;
        }
        
        return prev;
    }
}
```

# Triple Kill

Recursive Method.
```
{
    ListNode* reverseList(ListNode* head) {
       if(head == NULL || head -> next == NULL){
            return head;
        }
    
        ListNode* reversedListHead = reverseList(head -> next);
    
        head -> next -> next = head;
        head -> next = NULL;
    
        return reversedListHead;
    }
}
```
