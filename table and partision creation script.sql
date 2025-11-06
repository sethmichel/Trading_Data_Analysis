--the better way to do this is have this as a python file, and each sql block of code is in
--   a function which takes parameters. it would return the sql as a string


--creating the raw market data table and partitions
CREATE TABLE Raw_Market_Data_1 (
    ticker VARCHAR(10) NOT NULL,
    price DECIMAL(7, 3),
    macd_val DECIMAL(6, 4),
    macd_avg DECIMAL(6, 4),
    atr14 DECIMAL(5, 3),
    atr28 DECIMAL(5, 3),
    rsi SMALLINT,
    volume INTEGER,
    adx28 SMALLINT,
    adx14 SMALLINT,
    adx7 SMALLINT, 
    volatility_percent NUMERIC(4, 2),
    volatility_ratio NUMERIC(4, 2), 
    date_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (ticker, date_time)
)
PARTITION BY RANGE (date_time);

CREATE TABLE Raw_Market_Data_Partition_2025_1
PARTITION OF Raw_Market_Data_1
FOR VALUES FROM ('2025-01-01 00:00:00') TO ('2025-07-01 00:00:00');

CREATE TABLE Raw_Market_Data_Partition_2025_2
PARTITION OF Raw_Market_Data_1
FOR VALUES FROM ('2025-07-01 00:00:00') TO ('2026-01-01 00:00:00');

-----------------------------------------------------------------------
CREATE TABLE raw_trades_data_1
(
    exec_time timestamp with time zone NOT NULL,
    Spread text,
    Side text,
    Qty smallint,
    pos_effect text,
    symbol text,
    trade_exp text,
    strike text,
    trade_type text,
    price numeric(7, 3),
    net_price numeric(7, 3),
    price_improvement numeric(5, 2),
    order_type text,
    PRIMARY KEY (exec_time, symbol)
)
PARTITION BY RANGE (exec_time);

CREATE TABLE raw_trades_data_Partition_2025_1
PARTITION OF raw_trades_data_1
FOR VALUES FROM ('2025-01-01 00:00:00') TO ('2025-07-01 00:00:00');

CREATE TABLE raw_trades_data_Partition_2025_2
PARTITION OF raw_trades_data_1
FOR VALUES FROM ('2025-07-01 00:00:00') TO ('2026-01-01 00:00:00');
--------------------------------------------------------------

INSERT INTO Raw_Market_Data_1 (
    ticker, price, macd_val, macd_avg, atr14, atr28, 
    rsi, volume, adx28, adx14, adx7, 
    volatility_percent, volatility_ratio, date_time
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date_time) DO NOTHING

------------------------------------------------------------------

INSERT INTO Raw_Trades_Data_1 (
    ticker, price, macd_val, macd_avg, atr14, atr28, 
    rsi, volume, adx28, adx14, adx7, 
    volatility_percent, volatility_ratio, date_time
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date_time) DO NOTHING

-------------------------------------------------------------------

INSERT INTO raw_trades_data_1 (
    exec_time, spread, side, qty, pos_effect, 
    symbol, trade_exp, strike, trade_type, price, 
    net_price, price_improvement, order_type
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (exec_time, symbol) DO NOTHING

----------------------------------------------------------------------

CREATE TABLE market_data_validation (
    data_date date PRIMARY KEY,
    filename text,
    data_rows integer,
    status text,
    error_info text,
    date_checked text
);

-----------------------------------------------------------------------

CREATE TABLE real_trades_validation (
    data_date date PRIMARY KEY,
    filename text,
    data_rows integer,
    status text,
    error_info text,
    date_checked text
);

-----------------------------------------------------------------------

INSERT INTO market_data_validation (
    data_date, filename, data_rows, status, error_info, date_checked
) VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (data_date) 
DO UPDATE SET
    filename = EXCLUDED.filename,
    data_rows = EXCLUDED.data_rows,
    status = EXCLUDED.status,
    error_info = EXCLUDED.error_info,
    date_checked = EXCLUDED.date_checked

----------------------------------------------------------------------

CREATE TABLE trade_summaries
(
    trade_date date,
    trade_id smallint,
    ticker text,
    entry_time time,
    exit_time time,
    time_in_trade time,
    dollar_change numeric(7,2),
    running_percent_by_ticker numeric(5,2),
    running_percent_all numeric(5,2),
    total_investment numeric(8,2),
    entry_price numeric(7,3),
    exit_price numeric(7,3),
    trade_type text,
    qty integer,
    best_exit_price numeric(7,3),
    best_exit_time_in_trade time,
    worst_exit_price numeric(7,3),
    worst_exit_percent numeric(5,2),
    worst_exit_time_in_trade time,
    entry_atr14 numeric(5,3),
    entry_atr28 numeric(5,3),
    entry_volatility_percent numeric(4,2),
    entry_volatility_ratio numeric(3,2),
    entry_adx28 numeric(2,0),
    entry_adx14 numeric(2,0),
    entry_adx7 numeric(2,0),
    trade_holding_reached boolean,
    trade_best_exit_percent numeric(5,2),
    trade_percent_change numeric(5,2),
    PRIMARY KEY (trade_date, entry_time)
);
