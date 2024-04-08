# %% [markdown]
# # Part 1 - Prerequisites and Environment Setup

# %% [markdown]
# ### Course Server Setup
# 
# #### Git clone repository or extract submitted file in home directory
# 
# If you have the zip file of this project source code, just unzip at your home directory.  
# 
# If you don't have the source code on hand, you can clone the source from git repository by the following command
# 
# ```bash
# git clone https://github.com/reidlai/cp631-final
# ```
# 
# ~/cp631-final will be this project root folder.
# 
# 
# #### Miniconda 3 setup
# 
# Miniconda is a lightweight, open-source package and environment manager developed by Anaconda, Inc. It provides a simple and efficient way to install, manage, and distribute Python packages and their dependencies across multiple platforms, including Windows, macOS, and Linux. Unlike Anaconda, which includes a large collection of pre-installed scientific computing packages, Miniconda only ships the core Conda functionality, allowing users to customize their own package collections according to their specific requirements. With Miniconda, users can easily create isolated environments, switch between them, and share them with others via portable archives or cloud services. Additionally, Miniconda supports fast and parallel package installation through its mamba engine, which significantly improves the overall performance and usability of Conda. Overall, Miniconda offers a flexible and scalable solution for managing Python packages and environments, especially for data scientists, researchers, and developers who work with complex and diverse datasets and applications.
# 
# To install miniconda3, start shell in your server.  For this case, I use course server mcs1.wlu.ca.
# 
# ```bash
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# ```
# 
# Once miniconda3 installed, run the following command to populate conda environment setup script into .bash_profile
# 
# ```bash
# ~/miniconda3/bin/conda init bash
# ```
# 
# #### Create conda environement
# 
# Conda is a versatile tool for managing packages, dependencies, and environments for various programming languages, including Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN, and more. It is particularly popular in the fields of data science and machine learning. The conda create --name command is designed to create a new isolated environment within conda. The --name flag is followed by the name of the environment, in this case, cp631-final.
# 
# 
# ```bash
# conda create --name cp631-final
# ```
# 
# #### SSH Tunneling for remote Jupyter Notebook connection
# 
# To allow local machine connecting to Jupyter Notebook server running in course server, VPN connection must be up and running.  Then you can use SSH Tunnelling to forward all traffic of port 8888 in local macine to course server.
# 
# ```bash
# ssh -L 8888:localhost:8888 wlai11@mcs1.wlu.ca
# ```
# 
# #### Start Jupyter Notebook server 
# 
# Once the shell has been spawn in remote server, run the following command to start jupyter notebook server with new conda environment cp631-final
# 
# ```bash
# conda activate cp631-final
# conda install -c conda-forge -y python=3.10 pip numba=0.55.0 numpy pandas matplotlib seaborn yfinance kaggle jupyter notebook
# pip3 install install mpi4py
# jupyter notebook --no-browser --port=8888
# ```
# 
# All traffice at port 8888 will forward to localhost port

# %% [markdown]
# ### Kaggle Authenticiation
# 
# In this notebook, we will download a dataset from Kaggle. Before beginning the download process, it is necessary to ensure an account on Kaggle available. If you do not wish to sign in and would rather bypass the login prompt by uploading your kaggle.json file directly instead, then obtain it from your account settings page and save it either in the project root directory or content directory of Google Colab before starting this notebook. This way, you can quickly access any datasets without needing to log into Kaggle every time!

# %% [markdown]
# ### Import required packages

# %%
import csv
import numpy as np
import kaggle
import os
import pandas as pd
import random
import time
import yfinance as yf

from datetime import datetime, timedelta
from mpi4py import MPI

# %% [markdown]
# # Part 2 - Core Program

# %% [markdown]
# ## Initialize environment and variables

# %%
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

try:
  from numba import cuda
  cuda_installed = True
except:
  cuda_installed = False


os.environ["PROJECT_ROOT"] = "./"
# params = {}


# %% [markdown]
# ## S&P 500 Constituents Dataset Download
# 
# I will first need to download S&P 500 constituents from my Kaggle repository

# %%
if rank == 0:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('reidlai/s-and-p-500-constituents', path="s-and-p-500-constituents", unzip=True)

# %% [markdown]
# ## Stock Price History Download
# 
# get_stock_price_history_quotes function will download individual stock price history within range between start_date and end_date.

# %%
def get_stock_price_history_quotes(stock_symbol, start_date, end_date) -> pd.DataFrame:
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
    
    quotes_df = pd.DataFrame()
    
    if "." in stock_symbol:
        return None

    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
    except Exception as e:
        pass
    data.reset_index(inplace=True)
    data.rename(columns={
        'Date': 'date', 
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adjclose',
        'Volume': 'volume'
    }, inplace=True)
    data.insert(0, 'symbol', [stock_symbol] * data.shape[0])
    return data

# %% [markdown]
# ## Technical Analysis

# %% [markdown]
# ### EMA
# 
# As per proposal, we understand EMA is based on EMA value of T-1 day.  So there is dependency of daily record. And this is the main reason we cannot use CUDA for calculation.

# %%
def ema(values, days=12) -> np.ndarray:
    alpha = 2 / (days + 1)
    ema_values = np.empty_like(values)  # create an array to store all EMA values
    ema_values[0] = values[0]  # start with the first value
    for i in range(1, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


# %% [markdown]
# ### RSI
# 
# As per proposal, we understand RSI is based on value of EMA 12 and 26. So, same as ema function, there is dependency of daily records. And this is the main reason we cannot use CUDA for calculation.

# %%
def rsi(values, days=14) -> float:
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)
    avg_gain = sum(gains[:days]) / days
    avg_loss = sum(losses[:days]) / days
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi_value = 100 - (100 / (1 + rs))
    return rsi_value

# %% [markdown]
# ### MACD
# 
# In this section, we'll define three functions - macd, macd_cuda, and macd_gpu.
# 
# * macd function is a serial version to use dataframe calculating MACD
# 
# * macd_cuda is CUDA kernel function which has similar logic like macd except using numpy array
# 
# * macd_gpu is a wrapper function to copy data frame values into CUDA device memory and transfer back to host.

# %%

def macd(df, short_period=12, long_period=26, signal_period=9) -> pd.DataFrame:

    df["MACD"] = df["EMA12"] - df["EMA26"]
    return df

if cuda_installed:
    @cuda.jit
    def macd_cuda(ema12, ema26, macd):
        i = cuda.grid(1)
        if i < len(ema12):
            macd[i] = ema12[i] - ema26[i]
            
    def macd_gpu(df, signal_period=9):
        ema12_device = cuda.to_device(df["EMA12"].values)
        ema26_device = cuda.to_device(df["EMA26"].values)
        macd_device = cuda.to_device(np.empty_like(df["EMA12"].values))
        macd_cuda[df["EMA12"].shape[0], 1](ema12_device, ema26_device, macd_device)
        macd = macd_device.copy_to_host()
        
        del ema12_device
        del ema26_device
        del macd_device
        
        cuda.synchronize()
        cuda.current_context().memory_manager.deallocations.clear()
        
        df["MACD"] = macd
        return df

# %% [markdown]
# ### Read stock symbols from CSV file

# %%
# Read symbols from the CSV file
def read_symbols_from_csvfile(csvfile_path):
    symbols = []
    with open(csvfile_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            symbols.append(row[0])  # Assuming the symbol is the first column
    return symbols

# %% [markdown]
# ### Calculating EMA12, EMA26 and RSI for Serial and Parallel programming pattern

# %%
def emarsi(symbols, start_date, end_date, rank, size, params):

    results = pd.DataFrame()
    # Fetch stock price history quotes using the local symbols
    for symbol in symbols:
        # Load the stock price history data into pandas DataFrame
        stock_price_history_df = get_stock_price_history_quotes(symbol, start_date, end_date)
        if stock_price_history_df is not None and stock_price_history_df.shape[0] > 0:
            stock_price_history_df['EMA12'] = stock_price_history_df['close'].ewm(span=12, adjust=False).mean()
            stock_price_history_df['EMA26'] = stock_price_history_df['close'].ewm(span=26, adjust=False).mean()
            stock_price_history_df['RSI'] = stock_price_history_df['close'].rolling(window=14).apply(rsi, raw=True)
            results = pd.concat([results, stock_price_history_df])
    return results

# %% [markdown]
# ## Main Program

# %%

print(f"Rank: {rank}, Size: {size}")
local_symbols = np.array([])
symbol_trunks = np.array([])


df = pd.DataFrame()
df["numberOfProcesses"] = [size] * 5
df["numberOfStocks"] = [10, 50, 100, 200, 400]
df["numberOfDays"] = [30, 90, 180, 365, 730]

df["numberOfRows"] = df["numberOfStocks"] * df["numberOfDays"]

print(f"MainBody: Rank: {rank}, Size: {size}")

for index, row in df.iterrows():
    print(f"Processing {row['numberOfStocks']} stocks for {row['numberOfDays']} days")

    previous_day = datetime.now() - timedelta(days=1)
    end_date = previous_day.strftime('%Y-%m-%dT%H:%M:%S')
    first_day = previous_day - timedelta(days=int(row["numberOfDays"]))
    start_date = first_day.strftime('%Y-%m-%dT%H:%M:%S')
    data_dir = './data'
    
    parallel_start_time = MPI.Wtime()
    
    symbol_trunk = []
    local_symbols = []
    
    # Scatter symbols to all processes
    if rank == 0:
        symbols = read_symbols_from_csvfile(os.environ["PROJECT_ROOT"] + "s-and-p-500-constituents/sandp500-20240310.csv")
        symbols = symbols[:row["numberOfStocks"].astype(int)]
    
        symbols_per_process = len(symbols) // size
        remainder = len(symbols) % size
        if remainder != 0 and rank < remainder:
            symbols_per_process += 1
        
        # In case of processes spawned is more than number of symbols, we need to adjust symbols_per_process
        if symbols_per_process == 0:
            symbols_per_process = 1

        # Scatter symbols to all processes and each process should receive length of symbols / size blocks
        symbol_trunks = [symbols[i:i + symbols_per_process] for i in range(0, len(symbols), symbols_per_process)]  
        if len(symbol_trunks) < size:
            for i in range(len(symbol_trunks), size):
                symbol_trunks.append([])
        
    local_symbols = comm.scatter(symbol_trunks, root=0)
    
    if len(local_symbols) > 0:
        remote_results = emarsi(local_symbols, start_date, end_date, rank, size, params) 
    else:
        remote_results = pd.DataFrame()
    
    results = comm.gather(remote_results, root=0)
        
    parallel_end_time = MPI.Wtime()
    
    if rank == 0:
    
        results = pd.concat(results)
        
        if cuda_installed:
            results = macd_gpu(results)
        else:
            results = macd(results)
        
        parallel_end_time = MPI.Wtime()
        
        numberOfStocks = row["numberOfStocks"].astype(int)
        numberOfDays = row["numberOfDays"].astype(int)
    
        if not os.path.exists(os.environ["PROJECT_ROOT"] + "outputs"):
            os.makedirs(os.environ["PROJECT_ROOT"] + "outputs")
        
        results.to_csv(f"outputs/results-{size}-{numberOfStocks}-{numberOfDays}.csv", index=False)

        df.loc[index, "numberOfProcesses"] = size
        df.loc[index, "elapsedTimes"] = parallel_end_time - parallel_start_time

if rank == 0:
    filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-{size}.csv"
    if not os.path.exists(os.environ["PROJECT_ROOT"] + "outputs"):
        os.makedirs(os.environ["PROJECT_ROOT"] + "outputs")
    df.to_csv(filename, index=False)
    print(f"Saved stats to {filename}")
        


# %% [markdown]
# ## Export notebook into Python Script and Run with mpirun

# %% [markdown]
# ```bash
# mpirun -np 1 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# mpirun -np 2 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# mpirun -np 4 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# mpirun -np 8 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# mpirun -np 16 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# mpirun -np 32 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# ```

# %% [markdown]
# ```bash
# (cp631-final) [wlai11@mcs1 cp631-final]$ mpirun -np 64 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final_v2.py
# Traceback (most recent call last):
#   File "/home/wlai11/cp631-final/cp631_final_v2.py", line 83, in <module>
# Traceback (most recent call last):
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 215, in __init__
#     self._repopulate_pool()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 306, in _repopulate_pool
# Traceback (most recent call last):
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 215, in __init__
#     self._repopulate_pool()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 306, in _repopulate_pool
# Traceback (most recent call last):
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 215, in __init__
#     self._repopulate_pool()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 306, in _repopulate_pool
# Traceback (most recent call last):
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 215, in __init__
#     self._repopulate_pool()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 306, in _repopulate_pool
#     import kaggle
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/site-packages/kaggle/__init__.py", line 22, in <module>
#     api = KaggleApi(ApiClient())
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/site-packages/kaggle/api_client.py", line 85, in __init__
#     self.pool = ThreadPool()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 930, in __init__
#     return self._repopulate_pool_static(self._ctx, self.Process,
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 329, in _repopulate_pool_static
#     return self._repopulate_pool_static(self._ctx, self.Process,
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 329, in _repopulate_pool_static
#     w.start()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/dummy/__init__.py", line 51, in start
#     return self._repopulate_pool_static(self._ctx, self.Process,
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 329, in _repopulate_pool_static
#     w.start()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/dummy/__init__.py", line 51, in start
#     return self._repopulate_pool_static(self._ctx, self.Process,
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 329, in _repopulate_pool_static
#     w.start()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/dummy/__init__.py", line 51, in start
#     threading.Thread.start(self)
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/threading.py", line 935, in start
#     w.start()
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/dummy/__init__.py", line 51, in start
#     threading.Thread.start(self)
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/threading.py", line 935, in start
#     threading.Thread.start(self)
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/threading.py", line 935, in start
#     threading.Thread.start(self)
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/threading.py", line 935, in start
#     Pool.__init__(self, processes, initializer, initargs)
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 245, in __init__
#     _start_new_thread(self._bootstrap, ())
# RuntimeError: can't start new thread
# ```

# %% [markdown]
# # Part 3 - Data Visualization and Performance Analysis

# %% [markdown]
# ## Data Visualization

# %%
import pandas as pd
import os

os.environ["PROJECT_ROOT"] = "./"
filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-1.csv"

if os.path.exists(filename):
    df_stat_1 = pd.read_csv(filename)
else:
    df_stat_1 = None

filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-2.csv"
if os.path.exists(filename):
    df_stat_2 = pd.read_csv(filename)
else:
    df_stat_2 = None
    
filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-4.csv"
if os.path.exists(filename):
    df_stat_4 = pd.read_csv(filename)
else:
    df_stat_4 = None
    
filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-8.csv"
if os.path.exists(filename):
    df_stat_8 = pd.read_csv(filename)
else:
    df_stat_8 = None

filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-16.csv"
if os.path.exists(filename):
    df_stat_16 = pd.read_csv(filename)
else:
    df_stat_16 = None

filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-32.csv"
if os.path.exists(filename):
    df_stat_32 = pd.read_csv(filename)
else:
    df_stat_32 = None
    
df_stat = pd.concat([df_stat_1, df_stat_2, df_stat_4, df_stat_8, df_stat_16, df_stat_32,])
df_stat.reset_index(drop=True, inplace=True)

# %%
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
if is_notebook():
    display(df_stat)
else:
    print(df_stat)

# %%
# import matplotlib.pyplot as plt
# import seaborn as sns

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if is_notebook():
    fig = px.line(df_stat, x="numberOfRows", y="elapsedTimes", color="numberOfProcesses", markers=True, title="Elapsed Times vs Number of Rows")
    fig.show()
    

# %% [markdown]
# ## Performance Analysis

# %%
df_serial = df_stat.loc[df_stat["numberOfProcesses"] == 1, ["numberOfRows","elapsedTimes"]]
display(df_serial)


# %%
df_perf = pd.concat([
  df_stat.loc[df_stat["numberOfProcesses"] == 2, ["numberOfProcesses", "numberOfRows","elapsedTimes"]],
  df_stat.loc[df_stat["numberOfProcesses"] == 4, ["numberOfProcesses", "numberOfRows","elapsedTimes"]],
  df_stat.loc[df_stat["numberOfProcesses"] == 8, ["numberOfProcesses", "numberOfRows","elapsedTimes"]],
  df_stat.loc[df_stat["numberOfProcesses"] == 16, ["numberOfProcesses", "numberOfRows","elapsedTimes"]],
  df_stat.loc[df_stat["numberOfProcesses"] == 32, ["numberOfProcesses", "numberOfRows","elapsedTimes"]],
])
df_perf = df_perf.merge(df_serial, on="numberOfRows", suffixes=('_parallel', '_serial'))
df_perf["speedup"] = df_perf["elapsedTimes_serial"] / df_perf["elapsedTimes_parallel"]
df_perf["efficiency"] = df_perf["speedup"] / df_perf["numberOfProcesses"]
df_perf["overhead"] = (1 / df_perf["efficiency"] - 1) * 100
display(df_perf.sort_values(by=["numberOfRows", "numberOfProcesses" ])[["numberOfRows", "numberOfProcesses", "elapsedTimes_serial", "elapsedTimes_parallel", "speedup", "efficiency", "overhead"]])  

# %%
if is_notebook():
    fig = px.line(df_perf, x="numberOfRows", y="speedup", color="numberOfProcesses", markers=True, title="Speedup vs Number of Rows")
    fig.show()

# %%
if is_notebook():
    fig = px.line(df_perf, x='numberOfRows', y='efficiency', color='numberOfProcesses', markers=True, title='Efficiency vs Number of Rows')
    fig.show()

# %%
if is_notebook():
    fig = px.line(df_perf, x='numberOfRows', y='overhead', color='numberOfProcesses', markers=True, title='Overhead vs Number of Rows')
    fig.show()


