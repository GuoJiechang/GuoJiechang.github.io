---
layout: post
title: Leetcode 589 N-ary Tree Preorder Traversal
date: 2022-08-06 12:25 -0500
categories: Leetcode-Diary
---
# N-ary Tree Preorder Traversal problem results

[N-ary Tree Preorder Traversal](https://leetcode.com/problems/n-ary-tree-preorder-traversal/)
Given the root of an n-ary tree, return the preorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

![Result](/assets/images/n_ary_tree_preorder_traversal.png)

# First Blood
Visit each node's children recursively.
 
```
{
    /*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
public:
    void preorder(Node* head, vector<int>& op)
    {
        op.push_back(head->val);
        for(int i = 0; i < head->children.size(); i++)
        {
            preorder(head->children[i],op);
        }
    }
    vector<int> preorder(Node* root) {
        vector<int> op;
        
        if(root == nullptr)
            return op;
        
        op.push_back(root->val);
        for(int i = 0; i < root->children.size(); i++)
        {
            preorder(root->children[i],op);
        }
        
        return op;
    }
};
}
```

# Double Kill
The code could be simpler.

```
{

class Solution {
public:
    void pre(Node* head, vector<int>* op)
    {
        if(head)
        {
            op->push_back(head->val);
            for(int i = 0; i < head->children.size(); i++)
            {
                pre(head->children[i],op);
            }
        }
    }
    
    vector<int> preorder(Node* root) {
        vector<int> op;
        
        if(root)
        {
            pre(root,&op);
        }
       
        
        return op;
    }
};
}
```