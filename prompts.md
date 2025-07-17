START - my redesign idea
don't code, tell me any questions you have about this solution

Regarding the process in Grid_Search_Parameter_Optimization() which calls process_volatility_chunk():
I think there's a huge efficiency oversight. currently it processes every index of volailities by finding every 
combination of the other lists for that volaility and running those combinations of parameters over a data set. 
However, for each volatility it's looking at data in the data set where the volatility is greater than or equal to 
that volatility. this means for example that if it processes volatility = 0.6 then it processes volatility = 0.5, 
the 0.5 volatility will process the same combinations as the 0.6 volatility because it's looking for all volatilities 
greater than or equal to 0.5 (which includes 0.6). this is hugly inefficient

I think the best way to fix this is to 
1. have the volatilities list in descending order (which it is)

2. valid_combinations could overflow my ram if we're not careful. so it must now group combinations together by sub parameters instead of by volatility. currently it finds all the volatility = 0.9 combinations before finding any volatilities = 0.8 combinations for example; but it should find the first combination for the first volatility, then the first combination for the next volatility and so on until it has the first combination for each volatility. Then process this small chunk of combinations and then delete the list of combinations once it's processed (to save ram).

3. how to process the new combination list: the only difference in each combination is the volatility- the other parameters are the same between the lists. 

3a. Currently each combination is using a subset of the data which is greater than or equal to that volatility, we need to change this to only use data which has volatility in a range greater than or equal to volatility but less than the previous volatility (for the first calculation it won't have a previous volatility). So, if we calculate 0.9 volatility, then we're on 0.8 volatility, we'll only calculate data where volatility is between 0.8 (inclusive) and 0.9 (exclusive).

3b. Next, since the sublists volatility is in descending order, we take all the sublists between the previous volatility and the current volatility that are identical parameters (excluding volatility since that's the 1 difference parameter). This is hard, but we can see all the parameters in the sublists sublist_key which is created at line 275. the format of the sublist_key is this:

(volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss) as a tuple.

We add the sublist attributes sum, count, wins, loss, neither from the 1 higher volatility sublist to the current volatility sublist. the reason we do this is those 2 lists have the same parameters except for volatility, so the current implementation is for example calculating all the 0.9 volatility combinations for both 0.9 and 0.8, but this will only calculate the 0.9 combos for only 0.9 then then 0.8 will add those results to itself. this logic of adding previous results to itself should only extend 1 volatility previously. so if we're working on 0.5 volatility, it should add the results of 0.6 volatility but not for any other volatilities; the reason is 0.6 has already added 0.7, and 0.7 has already added 0.8 and so on.

if we find sublists with identical parameters (excluding volatility), raise an error because that shouldn't happen. If there's no data for a combination then the sublist should still have been created it just won't have any results. This brings up an edge case of when a combo has no data from the data source.

the upper bound for the highest volaility (index 0 of volatilities list) is nothing; for the first calculation for each group there is no adding from another list that happens since there is no other list

This is the only idea that removes redundancy and allows for pruning to save RAM. Please tell me if you have any questions, if not tell me what your implemtation plan is

additional clarification: 
when I say When I say "find the first combination for the first volatility, then the first combination for the next volatility"
what I mean is we loop over volatilities list and at the end of that loop the combinations created should be exactly the same except for the volatility. the length of this group will be the length of volatilities since it's 1 combo per volatility. this group is processed since they share all parameters. once that's processed the code should loop over volatilities again for the next combination of parameters. For example, the first group will have index 0 of all lists as the parameters, but the 2nd group will have the first index of all lists except the last list because they'll have the 2nd index. we loop over the lists just in a different way. The current loops likely need to be redesigned. Ask me more questions if you still aren't clear

Other notes:
1) we're changing the batch size from variable "batch_size" in Process_Volatility_Chunk() to the length of the volatilities list. and then deleteing the valid combinations once we're done calcaulting that batch. so there shouldn't be a memeory issue since volatilities is a short list.

2) Finding matching sublists by parameter tuple is expensive: but we'll deal with this later

------------------------------------------------------------------------------------------------------------------------

it came up with this idea BAD
Instead of changing the combination generation, what about inverting the processing order? Process each unique data row once, and for each row, determine which volatility thresholds it satisfies, then accumulate results across all applicable parameter combinations. This would eliminate redundant data processing entirely.

explaination prompt: your alternative approach sounds interesting. it might be similar time and space complexity to my idea but much more understandable. Please explain your idea in more detail and give me a implementation plan for it. Note that we'll still need to prune local_sublists so it does'nt overload my ram

This is undoable because we can't prune sublists until the end - I'd run out of ram

------------------------------------------------------------------------------------------------------------------------

current state. it's doing 1.4% in 1:17 minutes. issue is probably not using enough vectorization and not using numba enough. It also likely does a lot of redundant calculations and it probably has bugs. It actually got slower when I removed the combination redundancy

------------------------------------------------------------------------------------------------------------------------



redesign
-solve ram issue: prune sublists and use batch computation
-key speed: use numba and vectorization with large batches using every cpu thread

design goal: minimize computations
compute all required combos, and do all required addition at the end

- all combos are required, thus I can brute force it like I used to but change the filtering and make a 2nd addition algo (rsi case)



volatilities = [0.9,0.8,0.7,0.6,0.5,0.4,0.3] # KEEP IN DESCENDING ORDER
ratios = [0.5,0.6,0.7,0.8,0.9,1.0,1.1]
adx28s = [20,30,40,50,60]
adx14s = [20,30,40,50,60]
adx7s = [20,30,40,50,60]
abs_macd_zScores = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]   
extreme_rsis = [True, False, "either"]
normal_targets = [0.2,0.3,0.4,0.5,0.6]
upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
upper_stop_losss = [0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5]
normal_stop_losss = [-0.3,-0.4,-0.5,-0.6]