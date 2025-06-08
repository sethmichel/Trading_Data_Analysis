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

equations to find percents: for buy: (exit_price - entry_price) / entry_price * 100
for short: (entry_price - exit_price) / entry_price * 100

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

4.1) We'll make another feature in a few parts. write the feature in Parameter_Testing(), read the trade_csv_name file which is formatted like this: 

COIN
start_time	end_time	type
6:33:55	6:46:23	buy
6:46:23	7:01:08	short

where COIN isn't a header, but rather a row - that's the ticker symbol for the following data. The next row is the "headers" for that ticker. start_time, end_time... are column names for the following data. The rows under that are the data for their headers. for example: 6:33:55 is start_time, while 6:46:23 is the end_time. After that data, there's another ticker section in the same format, this repeates 9 times. The values for start_time and end_time are timestamps in hour:minute:second format. file_name is another csv file with headers Ticker,Price,Val,Avg,Atr14,Atr28,Rsi,Time, where time is hour:minute:second. it tracks stock market data for 9 tickers. An important note is that the timestamps in each csv file may or may not have leader 0's for single digits, as in one csv might write 6:15:9 while the writes that same timestamp as 06:15:09.

Using trade_csv_name's start_time, end_time, type column values for each ticker symbol, We're going to look at price data in file_name between each start_time and end_time. we'll use the price columns value at the start_time row as the entry price, and look for if the following rows would reach a 'target' percent (positive) of profit if we sold at that point, or reach a 'stop loss' percent (negative) based on the type value ('buy' or 'short').

the function has target and stop_loss variables which are floats representing percents. For each ticker in trade_csv_name, note each rows start_time, end_time, and type values, then find the row in file_name that has that start_time timestamp for that ticker (look at file_name's ticker and time columns). Note that rows price value, that's the entry price. 

Now print the row number and entry price for each start_time

before you write code, do you understand and do you have questions?

4.2) Now we're going to iterate over the following rows of file_name to see if the precent change in price compared to entry price reaches 'target' or 'stop loss' based on the type value. This depends on the type value from trade_csv_name; if it's 'buy' then higher prices are used to find the target value, if it's 'short' then lower values than the entry price are used to find the target value - this is because buy orders make money when the price increases, and short orders make money when the price decreases. Same logic for stop loss, if the stop loss is -0.5 and the type is 'buy' and the current rows price is -0.55% lower than the entry price, it has triggered the stop loss. If a row triggers a target or stop loss, print the row number, timestamp of the triggering, and percent change in price to the terminal (print()), then move on to the next start_time. Continue looking for target and stop loss until you reach a row with the same end_time timestamp, this means the cross didn't reach a target or stop loss, write that it didn't reach either to the terminal and move on to the next start_time

so, using the start_time and type, find the start_time in file_name's csv time column. use the price column as the entry price, then for each row after that compute the percent change of that rows price compared to the entry price based on the 'type'. if the percent change reaches target or stop loss, stop, report the results, move on to the next start_time

before you write code, do you understand and do you have questions?


- to add to 4.1 and 4.2, I actually want to know the best/worst exit before the target/sl was hit. if target was hit, what was the largest loss to taht point?

- compare the results to best/worst exit function results

- Then use google sheets to group by time, atr...





















