---
layout: post
title: Leetcode 98 Validate Binary Search Tree
date: 2022-08-06 12:22 -0500
categories: Leetcode-Diary
---
# Validate Binary Search Tree problem results

[Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

![Result](/assets/images/validate_binary_search_tree.png)

# First Blood
```
{
    /**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
  
    
    bool isBST(TreeNode* node,TreeNode* min_node,TreeNode* max_node)
    {
        if(!node)
            return true;
        
        if(min_node)
        {
            if(node->val <= min_node->val)
            {
                return false;
            }
        }
        
        if(max_node)
        {
            if(node->val >= max_node->val)
            {
                return false;
            }
        }
        
        return isBST(node->left,min_node,node) && isBST(node->right,node,max_node);
    }
    
    bool isValidBST(TreeNode* root) {
        return isBST(root,nullptr,nullptr);
    }
};
}
```