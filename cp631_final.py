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
# conda install -c conda-forge -y python=3.10 pip numba=0.55.0 numpy pandas plotly yfinance kaggle jupyter notebook
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
# Initialize MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Check if GPU device is available through Numba CUDA interface
cuda_installed = False
try:
  from numba import cuda
  device = cuda.get_current_device()
  if device is not None:
    cuda_installed = True
except Exception as e:
  cuda_installed = False

# Set project root directory
os.environ["PROJECT_ROOT"] = "./"


# %% [markdown]
# ## S&P 500 Constituents Dataset Download
# 
# I will first need to download S&P 500 constituents from my Kaggle repository in root process.

# %%
# Download S&P 500 constituents from my Kaggle repository in root process
if rank == 0:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('reidlai/s-and-p-500-constituents', path="s-and-p-500-constituents", unzip=True)

# %% [markdown]
# ## Stock Price History Download
# 
# get_stock_price_history_quotes function will download individual stock price history within range between start_date and end_date.

# %%
# get_stock_price_history_quotes function will download individual stock price history within range between start_date and end_date.
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
def emarsi(symbols, start_date, end_date, rank, size):

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

# Iterate over each sampling size
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
    
    # Scatter symbols to all processes from root process
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
    
    # Calculate EMA 12, EMA 26, and RSI using MPI
    if len(local_symbols) > 0:
        remote_results = emarsi(local_symbols, start_date, end_date, rank, size) 
    else:
        remote_results = pd.DataFrame()
    
    # Gather EMA 12, EMA26 and RSI results from all processes to root process
    results = comm.gather(remote_results, root=0)
    parallel_end_time = MPI.Wtime()
    
    # Concatenate all results from all processes and calculate MACD using CUDA
    if rank == 0:
        results = pd.concat(results)
        
        # Determine if we should use GPU or CPU for MACD calculation
        if cuda_installed == True:
            results = macd_gpu(results)
        else:
            results = macd(results)
        
        parallel_end_time = MPI.Wtime()
        
        # Output results to CSV file
        numberOfStocks = row["numberOfStocks"].astype(int)
        numberOfDays = row["numberOfDays"].astype(int)
        if not os.path.exists(os.environ["PROJECT_ROOT"] + "outputs"):
            os.makedirs(os.environ["PROJECT_ROOT"] + "outputs")
        results.to_csv(f"outputs/results-{size}-{numberOfStocks}-{numberOfDays}.csv", index=False)
        print(f"Saved results to outputs/results-{size}-{numberOfStocks}-{numberOfDays}.csv")

        # Update the elapsed time in the statistics dataframe
        df.loc[index, "elapsedTimes"] = parallel_end_time - parallel_start_time

# Save the statistics to a CSV file
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
# While carrying out program execution, it became evident that the application failed to create threads when attempting to utilize sixty-four processes simultaneously. This issue might stem from constraints imposed by the course server settings aimed at apportioning system resources fairly amongst multiple users enrolled in the same course. Consequently, the scope of our sampling size will now be confined to a maximum of thirty-two processes.

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
# While executing the program concurrently with both MPI and CUDA, an exception appeared, yet it did not affect the final outcome. Interestingly, when utilizing CUDA independently, there were no issues encountered. Following investigation within the open-source community, it is advised to update the CUDA toolkit version from 10.0 to 10.2. Nonetheless, it should be noted that such an upgrade could potentially influence various aspects of other project environments, thus I plan to address this matter during future maintenance procedures.
# 
# ```bash
# df_serial:    numberOfRows  elapsedTimes
# 0           300      0.947467
# 1          4500      3.273152
# 2         18000      6.807278
# 3         73000     14.379980
# 4        292000     33.064787
#     numberOfRows  ...     overhead
# 0            300  ...    65.171222
# 1            300  ...   210.205771
# 2            300  ...   353.955106
# 3            300  ...   902.776276
# 4            300  ...  1762.798583
# 5           4500  ...    11.708379
# 6           4500  ...    62.970777
# 7           4500  ...    68.678253
# 8           4500  ...   104.777233
# 9           4500  ...   285.126068
# 10         18000  ...    18.561165
# 11         18000  ...    37.822484
# 12         18000  ...    22.535292
# 13         18000  ...    79.396803
# 14         18000  ...   143.671642
# 15         73000  ...    21.229651
# 16         73000  ...    42.511562
# 17         73000  ...    18.240306
# 18         73000  ...    36.845391
# 19         73000  ...   121.695044
# 20        292000  ...    20.778152
# 21        292000  ...    36.712170
# 22        292000  ...    15.362431
# 23        292000  ...    27.213532
# 24        292000  ...    42.274391
# 
# [25 rows x 7 columns]
# Exception ignored in: <function Pool.__del__ at 0x2b866dbd53f0>
# Traceback (most recent call last):
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/pool.py", line 271, in __del__
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/queues.py", line 377, in put
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/connection.py", line 200, in send_bytes
#   File "/home/wlai11/miniconda3/envs/cp631-final/lib/python3.10/multiprocessing/connection.py", line 400, in _send_bytes
# TypeError: 'NoneType' object is not callable
# ```

# %% [markdown]
# # Part 3 - Data Visualization and Performance Analysis

# %% [markdown]
# ## Data Visualization
# 
# In Part 2, statistics files stats-{size}.csv have been generated in outputs folde.  These files are now imported into data frame to allow plotly rendering graphics in later sections for further analysis.

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

# %% [markdown]
# If running in Jupyter notebook, you can use the following code to display the data frame.

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

# %% [markdown]
# ### Elapsed Time vs Number of Rows
# 
# According to the line chart labeled "Elapsed Times versus Number of Rows," the program undergoes a remarkable boost in performance once eight processes are deployed, as demonstrated by the graph. Furthermore, the chart implies that while spawning 16 or 32 processes, there appears to be little discernible decline in processing time.

# %%

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if is_notebook():
    fig = px.line(df_stat, x="numberOfRows", y="elapsedTimes", color="numberOfProcesses", markers=True, title="Elapsed Times vs Number of Rows")
    fig['layout']['xaxis'].update(title_text='Number of Rows')
    fig['layout']['yaxis'].update(title_text='Elapsed Time (s)')
    fig.show()
    

# %% [markdown]
# ## Performance Analysis

# %%
df_serial = df_stat.loc[df_stat["numberOfProcesses"] == 1, ["numberOfRows","elapsedTimes"]]
if is_notebook():
    display(df_serial)
else:
    print(f"df_serial: {df_serial}")


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
if is_notebook():
    display(df_perf.sort_values(by=["numberOfRows", "numberOfProcesses" ])[["numberOfRows", "numberOfProcesses", "elapsedTimes_serial", "elapsedTimes_parallel", "speedup", "efficiency", "overhead"]]) 
else:
    print(df_perf.sort_values(by=["numberOfRows", "numberOfProcesses" ])[["numberOfRows", "numberOfProcesses", "elapsedTimes_serial", "elapsedTimes_parallel", "speedup", "efficiency", "overhead"]]) 

# %% [markdown]
# ### Speedup
# 
# The graph displayed below exhibits an intriguing trend. With the exception of the instance involving 32 processes, the line representing speedup remains horizontally flat when dealing with 73,000 rows. This phenomenon occurs more promptly when the number of processes does not exceed 8. Upon analyzing the quantity of rows, it becomes apparent that 73,000 rows divided by 16 processes yields approximately 4,563 rows per process, whereas 292,000 rows split by 32 processes equates to roughly 9,125 rows per process. These figures indicate that beyond a certain threshold known as the parallel overhead breakpoint, which lies somewhere between 4,563 rows per process and 9,125 rows per process, reducing parallel overhead results in a considerable reduction in processing requirements. Essentially, if we possess sufficient rows per process, we can significantly lessen parallel overhead.

# %%
if is_notebook():
    fig = px.line(df_perf, x="numberOfRows", y="speedup", color="numberOfProcesses", markers=True, title="Speedup vs Number of Rows")
    fig['layout']['xaxis'].update(title_text='Number of Rows')
    fig['layout']['yaxis'].update(title_text='Speedup')
    fig.show()

# %% [markdown]
# ### Efficiency
# 
# This graph corroborates the assertion put forth in the graph "Elapsed Times vs Number of Rows". The analysis reveals that when the number of processes stands at eight, the program attains optimal efficiency compared to all other tests carried out. Moreover, the parallel efficiency demonstrates a significant improvement when handling more than 73,000 rows in circumstances where thirty-two processes are active.

# %%
if is_notebook():
    fig = px.line(df_perf, x='numberOfRows', y='efficiency', color='numberOfProcesses', markers=True, title='Efficiency vs Number of Rows')
    fig['layout']['xaxis'].update(title_text='Number of Rows')
    fig['layout']['yaxis'].update(title_text='Efficiency')
    fig.show()

# %% [markdown]
# ### Overhead
# 
# As illustrated in the overhead graph, the program has the ability to substantially diminish parallel overhead regardless of the testing environment whenever the number of rows surpasses 18,000.

# %%
if is_notebook():
    fig = px.line(df_perf, x='numberOfRows', y='overhead', color='numberOfProcesses', markers=True, title='Overhead vs Number of Rows')
    fig['layout']['xaxis'].update(title_text='Number of Rows')
    fig['layout']['yaxis'].update(title_text='Overhead')
    fig.show()


