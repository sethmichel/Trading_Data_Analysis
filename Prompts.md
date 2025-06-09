DO NOT DELETE
timestamp: 7:36:00 (5/6 trading)


right now track_val_avg_crosses() iterates over a csv file containing these columns: Ticker,Price,Val,Avg,Atr14,Atr28,Rsi,Time, where time is hour:minute:second. it tracks stock market data for 9 tickers. it finds the timestamps where Val and Avg cross each other for more than 1 minute and writes that info a text file (timestamps grouped together by ticker symbol)

Let's expand on this functionality in the track_val_avg_crosses() function
1) currently: it only records the cross start times. start is defined as the moment the val and avg values cross each other, it then tracks that those values stay crossed for 1 minute, after the 1 minute the start time is confirmed to be the orignal val and avg cross time, or the current rows time - 1 minute. So, when I refer to a "cross", I'm refering to the val and avg values crossing each other and having stayed crossed for the 1 minute check period. It's possible for them to cross and cross back after less than 1 minute - this is not a valid cross.

Task: Please also record that crosses end times. End time is very similar to start time, during a valid cross if the val and avg values cross each other that timestamp is the end time. This ending cross needs to hold for 1 minute, so after val and avg cross they can't cross back again for 1 minute. After 1 minute the cross is confirmed to have ended and the end time is the time that the cross happend (also known as current rows time - 1 minute).

Challenging part: end time overlaps with the next crosses start time. Because end time and start time both need to be confirmed for 1 minute, the end time of the current cross and start time of the next testing cross overlap every time. In that case we'll simply track one 1 minute timer and at the end of the timer record the end time of the current cross and the start time of the new cross. This is the hardest part of this task.

write the results to a csv file. write hte results like this

COIN,,
start_time,end_time
06:33:55,06:46:23
06:46:23,07:01:08
...

this format will group results by ticker symbol with each ticker having it's own start_time and end_time columns. Skip 1 line for the next ticker symbol and repeat this format for every unique ticker symbol.

-check this, go second by second and track 2 trades. I think it's wrong somehow. use a stock that gets consolidated (multiple buys in a row), how does that happen?


2) goal: add a direction column to the output csv file. for each valid cross determine whether it represents a "buy" or "short" opportunity (buy and short are refering to stock market trades). you might be able to do this based on the current_direction variable in the function.

What 'buy' and 'short' are: you can tell which it is by if Val is less than or greater than the Avg during the cross. If Val > Avg during the cross it's a buy (when current_direction = 'up'). if Val < avg during the cross it's a short (when current_direction = 'down'). If Val = Avg, look ahead (row-by-row) until they differ. Then use the first non-equal comparison to decide.

specifics: This determination needs to happen when a cross is detected, not after the 1-minute validation window. More specifically, determine this using the val and avg values at the first row that a cross happens, not after the 1 minute trial period. So, you'll detect the val and avg crossing each other, then do 2 things: 1: determine if it's a buy or short, 2: do the current logic where you check if the cross is valid for x minutes (CROSS_DURATION_THRESHOLD variable). Only store the buy or short label if the cross is later validated (i.e., it lasts for CROSS_DURATION_THRESHOLD, usually 1 minute). If the cross is found to be invalid (does not persist long enough), then forget the newly found buy or short variable.

Implementation Steps:
While iterating through the rows:
Detect when Val and Avg cross (i.e., relative comparison flips).

Immediately determine direction (buy or short) using the current row.

Store this direction temporarily.

Begin the 1-minute validation window (as current logic does).

If the cross passes the validation window: 
    Record: start_time, end_time, and direction

If the cross fails the validation:
    Discard this cross entirely, including its direction label.

output csv format (for each ticker):
COIN
start_time,end_time,direction
06:33:55,06:46:23,buy
06:46:23,07:01:08,short
...


3) right now track_crosses() iterates over a csv file containing these columns: Ticker,Price,Val,Avg,Atr14,Atr28,Rsi,Time, where time is hour:minute:second. it tracks stock market data for 9 tickers, and all 9 tickers are written 1 after the other, so they're grouped in groups of 9 rows, for example, at timestamp 06:45:15 the csv has 9 rows each row being 1 tickers info that timestamp; the next 9 rows are the tickers info for the next timestamp. it finds the timestamps where Val and Avg cross each other for more than 1 minute and writes that info a csv file (timestamps grouped together by ticker symbol)

goal: add 4 new columns to the output csv file: best possible exit timestamp, best possible exit percent ROI, worst possible exit timestamp, worst possible exit percent ROI. These columns will be the return on investment (ROI) if I exited the trade position at certain times. This means tracking the ROI of each row. Start tracking when a cross is detected, and stop tracking when a exit cross is detected.

variables: 
-entry_price: when a cross is detected, not after the 1 minute confirmation period, record the price column value as "entry_price". this is the entry price for a stock market trade and will be referenced in calculations. 

-best_exit_price: this is set equal to the entry price at first, then for each row for the ticker, if exiting at that rows price would result in a greater ROI than exiting at the current best_exit_price, then set best_exit_price = current rows price

-worst_exit_price: this is set equal to the entry price at first, then for each row for the ticker, if exiting at that rows price would result in a worse ROI than exiting at the worst_exit_price, then set worst_exit_price = current rows price

-best_exit_percent: each time best_exit_price is updated, compute the percent change of entering a trade at entry price, and exiting the trade at best_exit_price. This is the ROI percent. round this percent to 2 decimal places

-worst_exit_percent: each time worst_exit_price is updated, compute the percent change of entering a trade at entry price, and exiting the trade at worst_exit_price. This is the ROI percent. round this percent to 2 decimal places

For simplicity, I'll refer to these 4 variables as the "price tracking variables"

equations to find percents: for buy: (exit_price - entry_price) / entry_price * 100
for short: (entry_price - exit_price) / entry_price * 100

current state: track_crosses() curr_cross_states is the dictionary you should use to track the price tracking variables for the current cross. this is where things get tricky: if curr_state['start_detected_time'] != None and curr_state['end_detected_time'] == None, you should be tracking those variables saved in curr_state. but notice that the price tracking variables are also in  next_cross_data. if curr_state['end_detected_time'] == None, then you should stop tracking the price tracking variables in curr_state, and start tracking them in next_cross_data. I've added as much of this code as I can to the function already, you can see the price tracking variables are moved to curr_state and reset starting line 201. also see that the else statement at line 214 is an edge case, you'll need to track the data in that edge case and in the if statement before that, because of these 2 sections it might be better to write your solution in a helper function.

edge case: 
-If a row produces the same ROI as we've found in another row, ignore it. we only care about the earlist row in that case
-Best/Worst ROI cannot Be Updated During the Exit Confirmation Period

challenge: best exit price/percent, worst exit price/percent, are determined by the direction (buy or short). if it's a buy then the best exit would be a higher value than the entry price, the worst exit would be a lower value than the entry price. the opposite is true if the cross is a short. I'm referring to stock trading buys and shorts so buys make money if the stock goes up, and shorts make money if the stock goes down.

Treat these variables the same way you treat the buy and short direction variables from my previous requests, that is determine them when the cross is detected, not after the 1 minute confirmation period. if the 1 minute confirmation period fails, meaning if the cross is found to be invalid (does not persist long enough), then forget the newly found variables.

stopping tracking: record these variabels while a cross is active, meaning the cross was detected and the 1 minute confirmation period is ongoing, post- 1 minute confirmation period, and stop tracking while the exit cross 1 minute confirmation period is ongoing. if that exit cross is found to be invalid (1 minute confirmation period fails) continue tracking until the next exit cross is found.

format
COIN
start_time,end_time,direction,best possible exit timestamp,best possible exit percent ROI,worst possible exit timestamp,worst possible exit percent ROI
06:30:00,06:33:55,short,09:56:20,0.4,06:34:20,-0.23
...




4) lets add a new feature to track_crosses()
Current functionality: track_crosses() iterates over a csv file containing these columns: Ticker,Price,Val,Avg,Atr14,Atr28,Rsi,Time, where time is hour:minute:second. it tracks stock market data for 9 tickers. This data is grouped mostly by Time in the csv, meaning at each timestamp it shows 1 row for each stock data, meaning 9 rows per timestamp. After all the tickers have been listed the next row is the new data for the enxt timestamp, and so on. The code finds all instances of val crossing over avg and decides if it's a "cross", meaning the values stay crossed for 1 minute. it then records various data points about this cross, and it finds when the cross ends, which is when the val and avg re-cross each other. When a cross ends it's also the start of a new cross. It outputs this data to a csv.

new feature: I want to see price movement as a percent change from the entry price and from the cross start time to the cross end time, this is particular to track though. it needs to be checked every loop for the duration of the trade. Each calculation takes into account if curr_state['direction'] is 'buy' or 'short' and if next_cross_data['direction'] is 'buy' or 'short' to determine the formula to find the percent change. similar to how the 'best and worst variables' are tracked (for example, best_exit_percent,and worst_exit_percent) if the direciton being used is 'buy' then positive percents will be if the price is higher than the entry price, and negative percents are if the price is lower than the entry price, shorts are the opposite - just like real stock market buys and shorts. We want to track every 0.1% price movement but only 1 time for each 0.1%, and we exclude 0.0%, this means if the price rises 0.3%, then falls 0.5% then rises 0.2%, we will record only the unique percents 0.1, 0.2, 0.3, -0.1, -0.2. notice that -0.1 for example is reached multiple times but we only record it the first time it's reached. those numbers represent a simple view of price movement during a time period. Let's call this variable curr_state['price_movement'] and next_cross_data['price_movement']. organize the data in 1 variable using '|' as a separateor. so my example would be written to the file (which I talk about in a moment) as "0.1|0.2|0.3|-0.1|-0.2".

how to track this new variable: it will be tracked very similar to how the best/worst variables are tracked ('best_exit_timestamp', 'best_exit_percent', 'best_exit_price', 'worst_exit_timestamp', 'worst_exit_percent', 'worst_exit_price'), in that it needs to checked every loop, will sometimes be cacluated in curr_state and sometimes in next_cross_data, and sometimes that data will be earased (set to None). do the calculation every time the best/worst variables are updated using the same dictionary that the best/worst variables are using at that time. It's probably better to do the calculation in a helper function. the data will also be handled in an indentical way to the best/worst variables, for example, you'll want to move the data from next_cross_data to curr_data like how the best/worst variables do in line 330, and you'll set it to None whenever the best/worst variabels are set to None. 

When the cross ends, add the data to results, and write it as a new column to the end of the csv file.
















- compare the results to best/worst exit function results

- Then use google sheets to group by time, atr...


















