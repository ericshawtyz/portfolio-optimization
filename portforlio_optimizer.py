import pandas as pd
import numpy as np
import datetime
import cvxpy as cvx
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import time


def get_covariance_returns(returns):
    """
    Calculate covariance matrices.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    returns_covariance  : 2 dimensional Ndarray
        The covariance of the returns
    """

    return np.cov(returns.fillna(0).T)


def get_optimal_weights(covariance_returns, return_vec, expected_return):
    """
    Find the optimal weights.

    Parameters
    ----------
    covariance_returns : 2 dimensional Ndarray
        The covariance of the returns
    return_vec : 2 dimensional Ndarray
        Index weights for all tickers at a period in time
    scale : int
        The penalty factor for weights the deviate from the index
    Returns
    -------
    x : 1 dimensional Ndarray
        The solution for x
    """
    M = 100
    num_assets = covariance_returns.shape[1]
    initial_weights = cvx.Variable(num_assets)
    reduced_weights = cvx.Variable(num_assets)
    etfs_selection = cvx.Variable(num_assets, boolean=True)
    portfolio_variance = cvx.quad_form(reduced_weights, covariance_returns)

    ## Objective function
    objective = cvx.Minimize(portfolio_variance)

    ## Constraints
    mu = return_vec * reduced_weights
    constraints = [initial_weights >= 0.001, initial_weights <= 0.3, sum(reduced_weights)==1,
                   reduced_weights - initial_weights >= -M * (1 - etfs_selection),
                   reduced_weights - initial_weights <= M * (1 - etfs_selection),
                   reduced_weights >= -M * etfs_selection, reduced_weights <= M * etfs_selection,
                   sum(etfs_selection) >= 5, sum(etfs_selection) <= 10, sum(mu) >= expected_return]

    ## Construct and solve the problem
    prob = cvx.Problem(objective, constraints)
    prob.solve()

    return reduced_weights.value, etfs_selection.value


def rebalance_portfolio(returns, expected_return, expense_ratio,
                        shift_size, chunk_size):
    """
    Get weights for each rebalancing of the portfolio.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    expected_return : float
        Expected rate of return for the portfolio
    shift_size : int
        The number of days between each rebalance
    chunk_size : int
        The number of days to look in the past for rebalancing

    Returns
    -------
    all_rebalance_weights  : list of Ndarrays
        The ETF weights for each point they are rebalanced
    """
    all_rebalance_weights = []
    dates_with_rebalance = []
    assets_tickers = list(returns)
    num_assets = len(assets_tickers)
    for end in np.arange(shift_size, len(returns), shift_size):
        rate_of_return = expected_return
        if end - chunk_size < 0:
            continue
        start = end - chunk_size
        partial_returns = returns[start:end]
        dates_with_rebalance.append(returns.index[end])
        #print(partial_returns.shape)
        covariance_returns = get_covariance_returns(partial_returns)
        rebalanced_asset_weights = None
        while rebalanced_asset_weights is None:
            rebalanced_asset_weights, assets_selection = get_optimal_weights(covariance_returns,
                                                                             partial_returns.values,
                                                                             rate_of_return + expense_ratio)
            rate_of_return -= 0.01
        rebalanced_asset_weights = [round(weight, 4) for weight in rebalanced_asset_weights]
        selected_asset_weights = {}
        for i in range(num_assets):
            if abs(assets_selection[i] - 1) < 0.001:
                selected_asset_weights[assets_tickers[i]] = rebalanced_asset_weights[i]
        print("Rebalanced portfolios on %s:" % returns.index[end])
        print(selected_asset_weights)
        print("Actual achievable return: %s" % round(rate_of_return + 0.01, 4))
        print("===========================================")
        print()
        all_rebalance_weights.append(rebalanced_asset_weights)

    return all_rebalance_weights, dates_with_rebalance


if __name__ == "__main__":
	# Acquire ETF price data
    etf_prices = pd.read_csv('data/prices.txt', sep=',')
    etf_prices = etf_prices.rename(columns={etf_prices.columns[0]: "date"}).set_index('date')
    #etf_prices = etf_prices.sample(n=20, random_state=1, axis=1)
    #etf_prices = etf_prices[:400]
    print(etf_prices.shape)

    # Calculate log return
    etf_returns_df = np.log(etf_prices) - np.log(etf_prices.shift(1))
    etf_returns_df = etf_returns_df.dropna()

    chunk_size = 250
    shift_size = 25
    expected_return = 0.1
    annual_expense_ratio = 0.0075
    print("Rebalancing portfolios...")
    start_time = time.time()
    all_rebalance_weights, dates_with_rebalance = rebalance_portfolio(etf_returns_df,
                                                                   expected_return,
                                                                   annual_expense_ratio,
                                                                   shift_size, chunk_size)
    print("Elapsed time: %s seconds" %(round(time.time() - start_time, 4)))
