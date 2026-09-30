#main event loop and T+1 execution logic

import portfolio
for t in range (len (df)):
    curr_bar = df.iloc[t]

    if pending_signal is not None:
        portfolio.execute_order(
            target_weight = pending_signal, 
            fill_price = current_bar['Open'],
            slippage_bps = slippage_bps,
            fee_per_trade = fee_per_trade
        )
        pending_signal = None

    portfolio.update_valuation(curr_bar['Close'])

    pending_signal = strategy.generate_signal(df.iloc[:t+1])