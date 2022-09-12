---
layout: post
title: Leetcode 102 Binary Tree Level Order Traversal
date: 2022-08-06 12:24 -0500
categories: Leetcode-Diary
---
# Binary Tree Level Order Traversal problem results

[Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

![Result](/assets/images/binary_tree_level_order_traversal.png)

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> node_queue;
        vector<vector<int>> result;
        
        if(root)
        {
            node_queue.push(root);
        }
        
        while(node_queue.size())
        {
            vector<int> cur;
            int level = node_queue.size();
            
            for(int i = 0; i < level; i++)
            {
                TreeNode* node = node_queue.front();
                cur.push_back(node->val);
                node_queue.pop();
                if(node->left)
                {
                    node_queue.push(node->left);
                }
                if(node->right)
                {
                    node_queue.push(node->right);
                }
            }
            result.push_back(cur);
        }
        
        return result;
    }
};
}
```