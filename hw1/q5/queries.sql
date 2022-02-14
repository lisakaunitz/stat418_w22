
# Lisa Kaunitz 
# STAT 418 - q5


# 5.1
select * from hw1.sales; # sales table
select * from hw1.stores; # stores table 

# 5.2 Which store makes the max sales on sundays?
select sum(s1), sum(s2) , sum(s3) , sum(s4) , sum(s5), sum(s6) , sum(s7) , sum(s8) , sum(s9) , sum(s10) 
from hw1.sales
where DAYOFWEEK(Date) = 1 ;
# Answer: S2 has makes the max sales on sundays 

# 5.3 Find all stores with total sales in December lower than those of s5. 
select sum(s1), sum(s2) , sum(s3) , sum(s4) , sum(s5), sum(s6) , sum(s7) , sum(s8) , sum(s9) , sum(s10) 
from hw1.sales
where MONTH(Date) = 12;
# Answer: S8, S9, S10

# 5.4 Which store recorded the highest number of sales for the largest number of days
select sum(s1), sum(s2) , sum(s3) , sum(s4) , sum(s5), sum(s6) , sum(s7) , sum(s8) , sum(s9) , sum(s10)
from hw1.sales;
# Answer: S2 

# 5.5 What week in 2019 has the highest total sales across all the stores
select WEEK(Date) as week_date, sum(s1) as s1, sum(s2) as s2, sum(s3) as s3, sum(s4) as s4, sum(s5) as s5, sum(s6) as s6, sum(s7) as s7, sum(s8) as s8, sum(s9) as s9 , sum(s10) as s10, sum(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10) as total_sales
from hw1.sales 
where YEAR(Date) = 2019 
group by week_date
order by total_sales desc;
# Answer: Week 37
