---
layout: post
title: Leetcode 0002 Add Two Numbers
date: 2022-06-16 12:08 -0500
categories: Leetcode-Diary
---
# ADD TWO NUMBERS problem results

[Add_Two_Numbers](https://leetcode.com/problems/add-two-numbers/)

![Result](/assets/images/add_two_numbers.png)


## First Blood
I feel very ashamed when solving this problem. How can I not even think about the Integer overflow issue? This should be a very basic trap.
My lessons learned from this problem:
Use all the test cases...

I don't want to publicize my stupid code...

## Double Kill
The second submission took me almost one hour... Integrating C++ extension with VS Code on my brand new Mac pro. Adding testbed. It's hard for me to get used to a brand new OS. But I did it!

This answer is somehow not very clear and has some redundant "if-else", could be more clear.

```
{class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        
        ListNode* result_list = new ListNode();
        
        int a = l1->val;
        int b = l2->val;
        int carry = (a+b)/10;

        result_list->val = (a+b)%10;

        ListNode* ptr = result_list;
        
        if(l1->next != nullptr)
        {
            l1 = l1->next;
        }
        else
        {
            l1 = nullptr;
        }
        
        if(l2->next != nullptr)
        {
            l2 = l2->next;
        }
        else
        {
            l2 = nullptr;
        }

        while((l1 != nullptr) || (l2 != nullptr))
        {
           if(l1 != nullptr)
           {
                a = l1->val;
           }
            else
            {
                a = 0;
            }

           if(l2 != nullptr)
           {
                b = l2->val;
           }
            else
            {
                b = 0;
            }

           int r = a + b + carry;

           ListNode* node = new ListNode();
           node->val = r%10;
           ptr->next = node;
           ptr = ptr->next;
         
           carry = r/10;

           if(l1 != nullptr && l1->next != nullptr)
           {
                l1 = l1->next;
           }
           else
           {
                l1 = nullptr;
           }
           if(l2 != nullptr && l2->next != nullptr)
           {
                l2 = l2->next;
           }
           else
           {
                l2 = nullptr;
           }

        }
       
        if(carry != 0)
        {
           ListNode* node = new ListNode();
           node->val = carry;
           ptr->next = node;
        }
        return result_list;
    }
};}
```
## Triple Kill

## Quadra Kill

## Penta Kill
On my last try, return result_list->next instead of result_list, in order to make the code clear.
```
{
    class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        
        ListNode* result_list = new ListNode();
        
        int a = 0;
        int b = 0;
        int carry = 0;

        ListNode* ptr = result_list;
        
        while((l1 != nullptr) || (l2 != nullptr))
        {
           a = 0;
           b = 0;
            
           if(l1 != nullptr)
           {
                a = l1->val;
           }

           if(l2 != nullptr)
           {
                b = l2->val;
           }

           int r = a + b + carry;

           ListNode* node = new ListNode();
           node->val = r%10;
           ptr->next = node;
           ptr = ptr->next;
         
           carry = r/10;

           if(l1 != nullptr)
           {
                l1 = l1->next;
           }
           
           if(l2 != nullptr)
           {
                l2 = l2->next;
           }

        }
       
        if(carry != 0)
        {
           ListNode* node = new ListNode();
           node->val = carry;
           ptr->next = node;
        }
        return result_list->next;
    }
};

}
```


For this problem, or simply for Leetcode, is it ok for us to ignore the memory leaking problem? Like using new inside the class function but never releasing them. Some awful memory of "memory leaking" started to attack me...