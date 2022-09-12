---
layout: post
title: Leetcode 121 Best Time to Buy and Sell Stock
date: 2022-08-06 12:29 -0500
categories: Leetcode-Diary
---
# Best Time to Buy and Sell Stock problem results

[Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

![Result](/assets/images/best_time_to_buy_and_sell_stock.png)

# First Blood
```
{
    class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int lessest = INT_MAX;
        int max = 0;
        int curr = 0;
        
        for(int i = 0; i < prices.size(); i++)
        {
            if(prices[i] < lessest)
            {
                lessest = prices[i];
            }
            
            curr = prices[i] - lessest;
            
            if(curr > max)
            {
                max = curr;
            }
        }
        
        return max;
    }
};
}
```

# Double Kill
```
{
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int lessest = INT_MAX;
        int max_profit = 0;
        
        for(int i = 0; i < prices.size(); i++)
        {
            lessest = min(lessest,prices[i]);
            max_profit = max(max_profit,prices[i] - lessest);
        }
        
        return max_profit;
    }
};
}
```
