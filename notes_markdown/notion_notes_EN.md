# Multi-Factor

Created time: January 18, 2026 4:44 AM

## Data Acquisition

Stock universe: US stocks

Fields: open price, close price, high price, low price, volume, traded amount (amt), percentage change, turnover rate

Time range: daily data from 2020 to present

Source: call an API in Python and write the data locally; yahoofinance is free and open source, and Alpaca's free API can trade directly

## Factor Construction

Input and output:

Input: raw volume-price data Excel file, lookback window N (parameter), e.g. from 2 days to 60 days; processing: use the data to construct factor values

Output: one Excel file per factor, where different sheets correspond to the factor under different lookback windows N

Volume-price factors: based on price, percentage change, volume, and turnover rate

~~Fundamental factors: financial reports / earnings flashes / forecasts, e.g. growth ability; profitability: profit; operating ability: expenses, costs~~

~~Analysts: expectations / earnings surprises~~

~~Capital-flow factors~~

~~Market sentiment factors~~

## Data Processing

Input and output:

Input: constructed factor data Excel files

Processing: for each period's factor values, i.e. cross-sectional data at the same time point (day), perform outlier treatment, standardization, and neutralization

i.e. process the values of a single factor across all stock tickers on one day (row), comparing only stocks against other stocks

Output: processed factor data Excel files

Outlier treatment:

- Mean-standard deviation winsorization: replace values outside mean +/- 3 standard deviations with boundary values. This is easily affected by extreme values.

- Median winsorization (most commonly used): calculate the factor median, define the factor absolute median deviation as the median of factor values after subtracting the median, and replace values outside median +/- 3 * 1.4826 * MAD with boundary values.

- Quantile winsorization: take upper and lower quantiles of the factor, such as the 5% and 95% quantiles, and replace values outside the quantile bounds with the quantile values.

Standardization:

1. Subtract the mean and divide by the sample standard deviation, replacing the original factor value with a z-score.

    $zi = (fi - \mu) / \sigma$

2. Standardize all newly obtained factor values.

~~Neutralization~~

~~- Industry neutralization (removing differences between industries): subtract the industry mean, using the current factor value minus the industry mean; or run a regression and take residuals (discrete).~~

~~- Market-cap neutralization: residuals after regressing against the logarithm of market capitalization.~~

## Single-Factor Testing

Input and output:

Input: processed factor Excel files, same-day return Excel file, rebalancing period, number of layers, and within-layer asset allocation method

Processing: use Jupyter to calculate IC, grouping, and other information for each period, then observe statistics across all periods and generate charts

Output: single-factor test PDF, including IC analysis, Rank_IC analysis, long-short backtest, long-only backtest, and short-only backtest, written locally

Preparation:

- Use the previous period's factor values and this period's daily returns for correlation analysis. Shift the factor down by one row.

- For each rebalancing day, calculate factor values, cumulative return, IC, rank_ic, grouping, and group_ic within the period.

e.g. with a rebalancing period T, replace the long-position targets every T days. Use the factor values from the day before the rebalancing date as this period's factor values, and use this month's cumulative return as this period's return.

Grouping: score the factor values within each period. For example, split into 10 layers with 30 stocks per layer; the 30 stocks with the highest factor values are group 10, and the stocks with the lowest factor values are group 1.

group_ic: weight the within-group returns. For example, for a group of 30 stocks, average the returns of the 30 stocks or weight them by factor score to obtain a within-group return.

Perform correlation analysis between groups 1-10 and the within-group returns to see whether within-group return increases as the group number increases.

IC and Rank_IC analysis:

Iterate through different rebalancing periods and take IC and rank_IC separately.

Calculate statistics for IC and rank_IC: mean, IR (IC / standard deviation), skewness, kurtosis, t-value, and p-value.

Charts: annual IC bar chart, monthly IC heatmap, rebalancing-day IC line chart, and rebalancing-day cumulative IC line chart. The cumulative IC line chart is the most important, and ideally it keeps increasing.

Long-short testing:

Iterate through different rebalancing periods and separately test portfolios such as group 1 long / last group short, group 2 long / second-last group short.

i.e. buy group 10 and group 9, and sell group 1 and group 2.

Calculate annualized return, volatility, Sharpe ratio, maximum drawdown, win rate, and PnL.

Charts: NAV line chart, monthly return heatmap, and rebalancing-day return line chart.

Layered long-only:

Iterate through different rebalancing periods and separately take return and excess_return.

Calculate annualized return, volatility, Sharpe ratio, maximum drawdown, win rate, and PnL for each long-only group.

Charts: grouped NAV line chart, grouped cumulative return bar chart, and rebalancing-day return line chart.

Layered short-only:

Iterate through different rebalancing periods and separately take return and excess_return.

Calculate annualized return, volatility, Sharpe ratio, maximum drawdown, win rate, and PnL for each short-only group.

Charts: grouped NAV line chart, grouped cumulative return bar chart, and rebalancing-day return line chart.

## Multi-Factor Batch Testing

Input and output:

Input: Excel files for multiple factor data sets, and return data Excel file

Processing: multi-factor batch testing

Output: multi-factor batch test report Excel file.

sheet1: each row is a different factor name, and each column is a test metric for a rebalancing period: mean, IR, t_value, rank_ic_mean, rank_ic_ir, rank_ic_t_value, group_rank_ic_mean, group_rank_ic_ir, group_rank_ic_t_value, long-short backtest return, long-short backtest Sharpe ratio, long excess return, long Sharpe ratio, short excess return, short Sharpe ratio, ic_p_value, rank_ic_p_value, group_rank_ic_p_value. Draw a heatmap for each column.

sheet2: each row is a date, each column is a different factor name, and the data is the factor's cumulative IC. Generate an additional line chart.

sheet3: long cumulative excess return (return minus market increase), with dates as rows and factor names as columns. Draw a line chart.

sheet4: long cumulative return, with dates as rows and factor names as columns. Draw a line chart.

note: the smaller the p-value, the better (green). Other values are better when redder. t_value = mean(IC) / (std(IC) / sqrt(T)); p_value < 0.05 is significant, and < 0.01 is strongly significant.

Calculating sheet1:

Call the functions built in the single-factor test to obtain IC, rank_IC, factor returns, and group_rank_ic mean, IR, t-value, and p-value.

Calculate the annualized return and Sharpe ratio of the long-short portfolio, plus long excess annualized return and excess Sharpe ratio.

Calculating sheets 2, 3, and 4: cumulative IC, long cumulative excess return, and long cumulative return.

Note: winsorize rank_IC. group_IC avoids factors that are biased toward only a narrow segment, i.e. it can only avoid the worst 10% but cannot distinguish the remaining 90% in detail.

t_value = signal / noise = (sample mean - hypothesized mean) / standard error. The larger the t-value, the stronger the signal relative to noise. When the t-value is close to 0, the signal is drowned out by noise.

p_value significance < 0.01

## Collinearity Analysis

Analyze the correlation between factors, e.g. price slope and price percentage change.

Input and output:

Input: Excel files for multiple factor data sets, and return data Excel file

Processing: collinearity analysis

Output: multi-factor collinearity analysis report Excel file:

sheet1: two matrices. Matrix1 is beta_corr, the correlation between factor return series. Matrix2 is factor_corr, the mean cross-sectional period correlation between factors.

sheet2: corr series. Columns are paired factor names (factor1 vs factor2), rows are each day (rebalancing day), and values are correlation coefficients. A line chart can be drawn for each series.

sheet3: cum_corr cumulative series. Historically rolling cumulative sums are calculated from sheet2, and slope curves for different factor pairs are plotted. If the slope is high, correlation is large; if the slope is low and near 0, they are uncorrelated.

Matrix1 factor return analysis:

On cross-sectional data (each day / parameter-tuning day), fit the current return using the previous period's factor values with univariate regression to obtain the estimated regression parameter beta, i.e. the factor return.

For the factor return series of multiple factors, calculate the correlation matrix beta_corr.

Matrix2 cross-sectional factor correlation analysis:

For five factors, calculate the IC correlation matrix on each rebalancing day, then take the mean of the sequence of factor correlation matrices produced on each rebalancing day to obtain matrix2.

## ~~Factor Transformation~~

~~After obtaining a factor, the factor layers may look strange, e.g. in theory layer 10 should be at the top and layer 1 should be at the bottom.~~

~~Solution: transform based on factor test results, e.g. take the absolute value, compromise, or flip; use machine learning to iterate through factor combination methods and transformation methods.~~

## Factor Combination + Backtesting

Through single-factor testing, multi-factor testing, collinearity analysis, and factor transformation, obtain several factors with relatively low correlation, no degradation, and good performance. For example, use 3 or 5 factors, and combine these final factors into one final factor. Generate multiple composite factors through different combination methods, then backtest the composite factors to find the better-performing ones.

Input and output:

Input: Excel files for the final factors that performed well, return data Excel file, rolling weighted window count N, and multiple-regression rolling weighted window count M

Processing: iterate through M and N, calculate weights, multiply row-wise by the factor data frame, obtain the composite factor, standardize it, and run batch testing

Output 1: composite factor data Excel file. Multiple composite factors are obtained through multiple combination methods. Each sheet is a composite factor data frame, with stocks as columns and dates as rows.

Output 2: composite factor batch backtest report Excel file

sheet1: run batch backtests on all composite factor data. The first column contains different composite factors, and the following columns contain test metrics: ic_mean, ic_IR, ic_t_value, rank_ic_mean, rank_ic_ir, rank_ic_t_value, group_rank_ic_mean, group_rank_ic_ir, group_rank_ic_t_value, long return, long excess return, long Sharpe ratio, ic_p_value, rank_ic_p_value, group_rank_ic_p_value. As in the multi-factor batch test, draw a heatmap for each column.

sheet2: each row is a date, each column is a different factor name, and the data is the factor's cumulative IC. Generate an additional line chart.

sheet3: long cumulative excess return (return minus market increase), with dates as rows and factor names as columns. Draw a line chart.

sheet4: long cumulative return, with dates as rows and factor names as columns. Draw a line chart.

Factor combination methods:

Univariate regression weighting (beta weighted):

Idea: use factor returns (beta) obtained from single-factor regression as weights. Beta is the slope obtained by fitting this rebalancing day's returns using the previous rebalancing day's factor values. Note: the betas do not sum to 1.

Method 1: directly take the mean of the factor return series as the weight, meaning the weight is unchanged each period.

Method 2: take the mean of the factor return series up to the current date. Clarify the difference between Method 1 and Method 2.

Method 3: take the mean of the factor return series over a rolling lookback window of N windows.

IC and Rank_IC weighting:

$w_i \propto IC_i$

Idea: similar to univariate regression weighting. Use the previous period's IC, the correlation coefficient with current-period return, as the weight.

Method 1: directly take the mean of the IC series as the weight, meaning the weight is unchanged each period.

Method 2: take the mean of the IC series up to the current date.

Method 3: take the mean of the IC series over a rolling lookback window of N windows.

Ranking weighting:

Ranking sum: replace the original factor values with the ordinal ranks of factor values in each period, and add different factors with equal weights.

Ranking product: replace the original factor values with the ordinal ranks of factor values in each period divided by the maximum rank, and directly multiply different factors.

Multiple regression weighting (OLS):

Idea: use factor returns obtained from multi-factor regression as weights. The fitting target can be returns, or normalized ranks on the return cross-section.

$$
r_{t+1} = \alpha + \beta_1 F_1 + \beta_2 F_2 + \dots + \epsilon
$$

Use ordinary least squares (OLS) to solve beta as the weight. It automatically handles collinearity, but is prone to overfitting.

Method 1: directly take the mean of the factor return series as the weight, meaning the weight is unchanged each period.

Method 2: take the mean of the factor return series up to the current date.

Method 3: take the mean of the factor return series over a rolling lookback window of M windows.

PCA (principal component analysis):

Idea: use multiple principal components obtained through principal component analysis directly as composite factors.

Factors obtained after PCA decomposition can be used directly as composite factors, or they can be run through multiple regression again and weighted by factor returns.

## Building the Strategy

Input and output:

Input: one final composite factor data frame Excel file, return data Excel file, asset allocation methods to iterate over (better calculation of each stock's weight), number of layers, rebalancing period, and long group selection, e.g. buy the group with the highest factor score (10).

Processing: grid-search long-only strategies under different parameter combinations and run batch backtests.

Output: strategy backtest report Excel file.

sheet1: backtest statistics for strategies under different parameter combinations. Rows are strategy names, and columns include each strategy's specific parameters, including number of layers, target group, rebalancing period, asset allocation method, and statistical metrics. Include a pivot chart where the legend is strategy name, asset allocation, target group, and rebalancing period; the axis is the number of layers; and the value is annualized return.

sheet2: portfolio returns under different parameter combinations, with dates as rows and strategy names as columns.

Specific parameters:

Asset allocation methods:

Equal-weight allocation: within the target layer bought long in each period, allocate equal weights across different stocks.

Markowitz optimal weights (MVO): based on all returns before the rebalancing day for different stocks in the target layer, calculate the portfolio weights that maximize Sharpe ratio. Drawback: high risk and high return, with large drawdowns and volatility.

Minimum variance portfolio (used more often): using the same data as above, calculate portfolio weights that minimize variance. Drawback: small drawdowns and volatility, but low annualized return.

Maximum return portfolio: using the same data as above, calculate portfolio weights that maximize return.

Factor-value scoring: on each rebalancing day, use the previous period's factor values to score the stocks in the target buy layer, and use the scores as weights.

Number of layers: depending on the number of stocks in the universe, choose 5, 10, 15, 20 layers, etc.

Rebalancing period: iterate within 1-60 days, such as 1 day, 3 days, 5 days, 10 days, 30 days, and 60 days.

Target group selection: buy the best-performing group, the second-best group, or the third-best group.

Strategy backtest:

Calculate strategy returns and cumulative return series under different parameter combinations.

Calculate different statistical metrics: latest 1-day, 1-week, 1-month, 3-month, half-year, 1-year, previous full-year return, annualized return, annualized volatility, Sharpe ratio, trade win rate, trade profit-loss ratio, annualized number of opened positions, annualized number of profitable opened positions, maximum drawdown, Calmar ratio, and maximum drawdown start and end dates.

## Walk-Forward Backtest

Using rolling time windows, divide the sample along the timeline into rolling training periods plus testing periods. Perform factor combination in the training period, then run backtest validation in the testing period. This evaluates the strategy's timeliness and parameter stability and avoids overfitting.

Input and output:

Input: price data Excel file, selected factor Excel files

Processing:

Rolling walk: training window + testing window + rolling step size

Each walk: the trainer performs data processing, combines factor weights, and applies them to the testing window. In the testing window, run batch backtests and obtain test results for different strategies.

Output:

Test result report.

sheet1: overall summary and configuration summary

sheet2: parameter_stability: statistical metrics for different backtest results of each strategy

sheet3: sensitivity analysis (number of layers, target group, rebalancing period, asset allocation)

sheet4: analysis of each walk

sheet5: daily returns for different strategies

sheet5: cumulative returns for different strategies

Processing: training window N days, e.g. 400 days; testing window M days, e.g. 60 days; rolling step size, e.g. 30 days; gap between training and testing, e.g. 0 days.

## Follow-Up

1. ~~Build walk-forward validation.~~

~~Please combine this with the process in claude.md. You are a quantitative engineer. Review the already-written multi-factor model. The run flow is: first pull_yhfinance_data, then build_factors to construct factors from factor_library, then data_process to process data, then run_multi_factor_test for factor selection, then run_colinearity_test for collinearity analysis, then run_composite_factor to perform factor combination in different ways and select one, then run_strategy for the final factor backtest under different parameters. For example, if the final composite factor I choose is beta_m3_N10, and the specific parameters are mvo_5G_Top2_P10d, you can find them in different config files. But for a more rigorous backtest, I decided to use walk-forward backtesting. How much historical data do you think should be read, and how far into the future should be backtested?~~

1. ~~Fix future-function bug. Please combine this with the process in claude.md. You are a quantitative engineer. Review the already-written multi-factor model. The run flow is: first pull_yhfinance_data, then build_factors to construct factors from factor_library, then data_process to process data, then run_multi_factor_test for factor selection, then run_colinearity_test for collinearity analysis, then run_composite_factor to perform factor combination in different ways and select one, then run_strategy for the final factor backtest under different parameters. For example, if the final composite factor I choose is rank_mul, and the specific parameters are min_variance_10G_Top2_P10d, you can find them in different config files. Carefully analyze my process and search for future functions (data leakage), logical flaws, configuration inconsistencies, parameter mapping issues, rolling-window bugs, or other bugs that may affect backtest results.~~
2. Current question 1: the rolling backtest is built on five factors and a composite factor method that are confirmed in advance, then iterates through different weights, rebalancing periods, target groups, number of groups, etc. to confirm the optimal strategy. But could this cause other factor combination methods to produce better strategies?

    Question 2: Should I use the full backtest from strategy construction or the walk_forward backtest?

3. ~~Select factors [20, 16, 43, 17, 34], composite factor method "beta_m3", and specific strategy parameters mvo_5G_Top2_P10d. Find the specific rebalancing data for every historical backtest rebalance, including specific stock buy/sell prices and returns. Use one sheet for specific operations, one sheet for the return series, and one for cumulative returns. You can add more sheets yourself with more detailed and useful data.~~
4. ~~Question: close price or open price?~~
5. ~~Build daily / every-rebalancing-day dispatch logic: pull_data, build_factors, data_process, run_multi_factor_test, run_composite_factor, use the fixed parameters previously selected by run_strategy to generate holdings and weights, compare with current holdings to calculate stock buys/sells, and send a Discord notification.~~
6. ~~Build Discord bot.~~
7. ~~Add recent timeliness analysis to factors, multi-factor test, and composite_factor.~~

~~Strategy backtest sorting~~

1. ~~On the rebalancing day, do we want to execute trades before the market close? If yes, when running run_reblance_day on the rebalancing day, should we use the opening price and current price (near close) of that day to assume a closing price, or should we completely avoid using any price data from the rebalancing day and use only prior data?~~

![image.png](image.png)

1. ~~If 30 days is the rebalancing period, should the next trading day be 30 calendar days later or 30 rebalancing/trading days later? Similarly, in backtesting, does N=5 or 10 use calendar days or trading days?~~

    ~~Use rebalancing/trading days.~~

2. ~~Add the specific strategy to Discord replies.~~
3. ~~Fix recent rebalancing day bug. The recent rebalancing day and latest rebalancing day are wrong; see the output of rebalance_day.~~
4. Add a post-run strategy check showing current strategy performance, including specific buy prices, positions, current prices, etc.
5. ~~Fix collinearity analysis bug where factors in config and run results are inconsistent.~~
6. ~~When offset is set to 0 or 7, backtest returns differ greatly.~~

    ~~idea: pull more or fewer days of data when pulling data.~~

    ~~idea2: move the data start date earlier or later by 7 trading days.~~

7. ~~Explanation: e.g. if a rebalancing-day return is 10% and the original NAV is 32, then 32 x 1.1 = 35.2.~~
8. ~~In all run_program files, only keep actions for stocks with weight > 0.0001, ignore operations with tiny weights, and improve backtest speed.~~
9. ~~Change run_rebalance_day factor selection to select from its own config.~~
10. ~~Add rules so each prompt automatically reads them, i.e. automatically read README, answer in Chinese as a quantitative engineer, update Update, etc.~~
11. ~~Add the change from holding date to now in Discord replies.~~
12. ~~yhfinance has data delay. For example, in the early morning of April 9, there is April 8 high, low, and open data but no close. In Pull yhfinance data, use `fast_info.last_price` instead of Close.~~
13. ~~Change maximum drawdown to one rebalancing period.~~
14. Use the multi-factor model to test the 10-plus stocks recommended by the US stock investment site every day. Question: offset in run_rebalance_day does not work.

## Questions / Optimizations

1. ~~Issue where the run_rebalance_day backtest results and rebalancing-day holdings were inconsistent when run on April 1 and April 8. Root cause: BKNG had a 1:25 stock split on April 6, causing BKNG's factor values to be affected. Z-score standardization could not eliminate the deviation, which caused different composite factor values -> different grouping -> different holdings -> different backtest results.~~

| **Factor** | **Specific Formula** | **BKNG 25x Impact** |
| --- | --- | --- |
| **alpha024** | `delta(close, 3)` = current-day close - close 3 days ago | 25x |
| **alpha024** | `delta(sma(close,100), 100)` = 100-day-ago moving average - moving average 100 days before that | 25x |
| **alpha095** | `open_ - ts_min(open_, 12)` = open price - 12-period minimum | 25x |
| **alpha064** | `delta(hl_vwap, 4)` = 4-period difference of mid-price | 25x |
| **alpha065** | `open_ - ts_min(open_, 14)` = open price - 14-period minimum | 25x |
| **alpha062** | `(high + low) / 2`, `open_` absolute levels | 25x |

~~Solution: perform industry standardization on prices, or use price change rates in factors instead of price differences.~~

1. ~~Use vectorization instead of apply or for loops, using vector operations to improve performance.~~
2. Buying US stocks in Canada creates roughly 4% friction from converting to USD and then back to CAD. Use IBKR or moomoo; see Gemini for details.
3. ~~Found issue: future rebalancing days did not account for US stock-market non-federal holidays, such as April 17.~~

~~Specific issue: when running the rebalancing day, future rebalancing days are inferred using only weekends and federal holidays, missing non-federal holidays such as Good Friday on April 3 and early closes on Black Friday and Christmas Eve.~~

~~Specific case: the March 13 rebalancing day incorrectly predicted April 10 as the next rebalancing day, but the actual rebalancing day should have been April 13. The reason is that Good Friday on April 3 was a market holiday.~~

~~Fix: rebalance again on Thursday, April 17.~~

## Live Trading

Two strategies

1:

Stock selection: 100+

Factor selection: [95, 101, 62, 65, 32]  *#3.17*

Composite method: ic_m3_N20

Strategy parameters: max_return_5G_Top1_P10d

First rebalancing date: Friday, March 27

Rebalancing operation on March 27: MU 20%, WDC 40%, SNDK 40%

Rebalancing operation on April 10: STX 20%, WDC 40%, SNDK 40%

Rebalancing operation on April 17 (should have been April 13, but issue found on April 17): ASTS 20%, WDC 40%, SNDK 40%

Rebalancing operation on April 27:

Future rebalancing dates: Monday, April 27; Monday, May 11; Monday, May 26

2:

Stock selection: 100+

Factor selection: *[95, 24, 64, 65, 32] #3.25*

Composite method: ic_m3_N20

Strategy parameters: max_return_10G_Top1_P20d

First rebalancing date: Friday, April 10

Previous rebalancing date: March 13. Operation: TTMI 20%, ONDS 40%, RKLB 40%

Rebalancing operation on April 10: TTMI 20%, WDC 40%, SNDK 40%

Rebalancing operation on April 17 (should have been April 13, but issue found on April 17): ONDS 40%, SNDK 40%, MRK 8.5%, JNJ 7.6%, CSX 3.9%

Future rebalancing dates: Monday, May 11; Tuesday, June 9

~~3:~~

~~Stock selection: US stock investment site 10+~~

~~Factor selection: *[23, 43, 66, 45, 31] #4.15*~~

~~Composite method: rank_add~~

~~Strategy parameters: max_return_10G_Top1_P10d~~

~~First rebalancing date: Friday, April 10~~

~~Previous rebalancing date: March 13. Operation: TTMI 20%, ONDS 40%, RKLB 40%~~

~~Rebalancing operation on April 10: TTMI 20%, WDC 40%, SNDK 40%~~

~~Future rebalancing dates: Friday, May 8; Friday, June 5~~

Solution: perform industry standardization on prices, or use price change rates in factors instead of price differences.
