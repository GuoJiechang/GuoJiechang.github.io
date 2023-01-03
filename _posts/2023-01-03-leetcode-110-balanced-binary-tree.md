---
layout: post
title: Leetcode 110 Balanced Binary Tree
date: 2023-01-03 15:18 -0600
categories: Leetcode-Diary
---
[110-Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/description/)
Given a binary tree, determine if it is height-balanced.

Height-Balanced
A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.

# Method 1
Intuitively check each node and calculate the height of the left and the right subtree. This method is a Top-down method, containing duplicate calculations for the height of subtrees.
I was stuck at the calculation of the height of the tree. For calculating the height, the height of an empty tree is -1. 
For each node, we check its left and right to get the largest height from the subtree, and take the node itself into height.
```
{
class Solution {
public:
    int depth(TreeNode* root)
    {
        if(root == nullptr)
        {
           return -1;
        }

        return 1 + max(depth(root->left), depth(root->right));
    }
    
    bool isBalanced(TreeNode* root) {
        if(!root)
            return true;
        if(abs(depth(root->left) - depth(root->right)) > 1)
            return false;
        return isBalanced(root->left) && isBalanced(root->right);
    }
};
}
```

# Method 2
To avoid extra computation on the height of the subtrees, we can remember the height of the child and check if a child's subtree is balanced.

```
{
class Solution {
public:
    bool isBalancedHelper(TreeNode* root, int &height)
    {
        //empty tree is balanced
        if(root == nullptr)
        {
            height = -1;
            return true;
        }

        //check children
        int left, right = -1;

        //left subtree and right subtree should be balanced
        if(isBalancedHelper(root->left, left) && isBalancedHelper(root->right, right) 
        && abs(left-right)<=1)
        {
            //update current node height
            height = max(left,right) + 1;
            return true;
        }

        return false;
    }
    
    bool isBalanced(TreeNode* root) {
        int height = -1;
        return isBalancedHelper(root,height);
    }
};

}
```