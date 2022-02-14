#!/usr/bin/env python
# coding: utf-8

# Segment 5.1


import pandas as pd


df = pd.read_csv('~/Desktop/STAT_418/stat418_w22/hw1/q5/tvsales.csv')



# Creating Table 1: Stores Table 
stores_list = ['S1'], ['S2'], ['S3'], ['S4'], ['S5'], ['S6'], ['S7'], ['S8'], ['S9'], ['S10']
stores = pd.DataFrame(stores_list, columns = ['Store Name'])
stores.to_csv("stores.csv")



# Creating Table 2: Sales Table 
sales = df.copy()
sales
sales.to_csv("sales.csv")

