---
layout: post
title: Leetcode 235 Lowest Common Ancestor of a Binary Search Tree
date: 2023-01-03 14:51 -0600
categories: Leetcode-Diary
---
[235-Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

## Notes
Binary Search Tree(BST)
Binary Search Tree is a node-based binary tree data structure which has the following properties:

The left subtree of a node contains only nodes with keys lesser than the node’s key.
The right subtree of a node contains only nodes with keys greater than the node’s key.
The left and right subtree each must also be a binary search tree.[](https://www.geeksforgeeks.org/binary-search-tree-data-structure/)

For a problem related to a tree, it is intuitive to think about divide and conquer and solve the problem recursively.

The lowest common ancestor should have the property that p and q are on different sides of the node. for simplicity, let p be smaller than q, if the node's value satisfies p <= node <= q, then the node is the lowest common ancestor.

Three conditions:
1. The value of root is smaller than p and q, check the left subtree
2. The value of root is bigger than p and q, check the right subtree
3. The value of root satisfies p <= node <= q

```
{
    class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if((p->val <= root->val && q->val >= root->val) ||
        (q->val <= root->val && p->val >= root->val))
        {
            return root;
        }
        else if(p->val > root->val && q->val > root->val)
        {
            return lowestCommonAncestor(root->right,p,q);
        }
        else if(p->val < root->val && q->val < root->val)
        {
            return lowestCommonAncestor(root->left,p,q);
        }
        return nullptr;
    }
};
}
```
Actually, as the description, p and q must exist in the tree, so the LCA must exist too, the three conditions can change to
1. The value of root is smaller than p and q, check the left subtree
2. The value of root is bigger than p and q, check the right subtree
3. The root must be LCA

The code can be cleaner.
```
{
    class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(p->val > root->val && q->val > root->val)
        {
            return lowestCommonAncestor(root->right,p,q);
        }
        else if(p->val < root->val && q->val < root->val)
        {
            return lowestCommonAncestor(root->left,p,q);
        }
        else
        {
            return root;
        }
    }
};
}
```
