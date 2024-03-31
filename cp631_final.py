# %% [markdown]
# ## Prerequisites

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
# #### Required packages installation by conda
# 
# The conda install command is utilized to install packages in a specific conda environment, with the --name flag specifying the name of the environment.
# 
# ```bash
# cd ~/cp631-final # change directory to project root
# conda install --name cp631-final conda_requirements.txt
# rm ~/miniconda3/envs/cp631-final/compiler_compat/ld
# pip3 install -f pip3_requirements.txt
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
# jupyter notebook --no-browser --port=8888
# ```
# 
# All traffice at port 8888 will forward to localhost port

# %% [markdown]
# ### Delare params dict

# %%
params = {}

# %% [markdown]
# ### Google Colab Environment Checking

# %%
import os
import sys

params["in_colab"] = 'google.colab' in sys.modules
print("In colab: ", params["in_colab"])
os.environ["PROJECT_ROOT"] = "./"
print("Project root: ", os.environ["PROJECT_ROOT"])



# %% [markdown]
# ### Jupyter Notebook Environment Checking

# %%
from IPython import get_ipython
def in_notebook():
    try:
        ipython_instance = get_ipython()
        if ipython_instance is None:
            return False
        elif ipython_instance and 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True

params["in_notebook"] = in_notebook()
print(f"in_notebook: {params['in_notebook']}")

# %% [markdown]
# ### MacOS Environment Checking
# 
# In this code, platform.system() returns the name of the operating system dependent module imported. The returned value is 'Darwin' for MacOS, 'Linux' for Linux, 'Windows' for Windows and so on. If the returned value is 'Darwin', it means you are using MacOS.

# %%
import platform
import distro

if platform.system() == 'Darwin':
    params["is_macos"] = True
else:
    params["is_macos"] = False

print(f'is_macos: {params["is_macos"]}')

if platform.system() == 'Linux':
    distro_name = distro.id()
    if 'debian' in distro_name.lower() or 'ubuntu' in distro_name.lower():
        params["is_debian"] = True
        params["is_redhat"] = False
    elif 'centos' in distro_name.lower() or 'rhel' in distro_name.lower():
        params["is_debian"] = False
        params["is_redhat"] = True
    else:
        params["is_debian"] = False
        params["is_redhat"] = False
else:
    params["is_debian"] = False
    params["is_redhat"] = False
    
print(f'is_debian: {params["is_debian"]}')
print(f'is_redhat: {params["is_redhat"]}')

if platform.system() == 'Windows':
    params["is_windows"] = True
else:
    params["is_windows"] = False
print(f'is_windows: {params["is_windows"]}')

# %% [markdown]
# ### Check if using Ubuntu WSL 2.0 or not

# %%
def is_wsl():
    try:
        with open('/proc/version', 'r') as fh:
            return 'microsoft' in fh.read().lower()
    except FileNotFoundError:
        return False

params["is_wsl"] = is_wsl()
print(f"is_wsl: {params['is_wsl']}")


# %% [markdown]
# ### Check if MPI installed in OS
# 
# Use the mpirun command to see if MPI is up and running.
# 
# **Remarks**: We cannot use subprocess module to spawn os command "mpirun" to check because mprun will call this script in later testing stage and crash with recursive call error.

# %%
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# if size > 0:
#     params["mpi_installed"] = True
#     print(f'MPI installed: {params["mpi_installed"]}')
# else:
#     params["mpi_installed"] = False
#     print(f'MPI installed: {params["mpi_installed"]}')
    # print("MPI is not installed or only has one process")


# %% [markdown]
# ### Check if NVIDIA CUDA toolkit installed
# 
# Use the numba command to see if CUDA toolkit works properly.

# %%
from numba import cuda

def is_cuda_installed():
    try:
        cuda.detect()
        return True
    except cuda.CudaSupportError:
        return False

if rank == 0:
    params["cuda_installed"] = is_cuda_installed()    
    print(f'CUDA installed: {params["cuda_installed"]}')

    if not params["cuda_installed"]:
        print("[FATAL] CUDA is not installed")

# %% [markdown]
# ## Environment Setup

# %% [markdown]
# ### Kaggle Authenticiation
# 
# In this notebook, we will download a dataset from Kaggle. Before beginning the download process, it is necessary to ensure an account on Kaggle available. If you do not wish to sign in and would rather bypass the login prompt by uploading your kaggle.json file directly instead, then obtain it from your account settings page and save it either in the project root directory or content directory of Google Colab before starting this notebook. This way, you can quickly access any datasets without needing to log into Kaggle every time!

# %% [markdown]
# ### Install PyPi packages
# 
# Installing PyPi packages is an essential step in this notebook. Among the mandatory packages, mpi4py and opendatasets provide crucial functionalities for data manipulation, distributed computing, and accessing large datasets. While Google Colab offers the convenience of bundled packages such as numpy, matplotlib, pandas, and seaborn, these packages still need to be installed separately in a local environment.
# 
# Run the following code in shell to install all required library and packages
# 
# ```bash
# conda install --file requirements.txt
# ```

# %% [markdown]
# ### Check CUDA Toolkit and Numba info

# %%
# import subprocess

# if rank == 0 and params["cuda_installed"]:
#     subprocess.run(["nvcc", "--version"])

# %%
# if rank == 0 and params["in_notebook"] and params["cuda_installed"]:
#     subprocess.run(["numba", "-s"])

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

if params["cuda_installed"]:
    from numba import cuda, float32


# %% [markdown]
# ## S&P 500 Constituents Dataset Download
# 
# I will first need to download S&P 500 constituents from my Kaggle repository

# %%
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
    
    quotes_df = pd.DataFrame(columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'adjclose', 'volume'])
    
    if "." in stock_symbol:
        return None

    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        quotes_df['timestamp'] = data.index
        quotes_df['open'] = data['Open']
        quotes_df['high'] = data['High']
        quotes_df['low'] = data['Low']
        quotes_df['close'] = data['Close']
        quotes_df['adjclose'] = data['Adj Close']
        quotes_df['volume'] = data['Volume']
    except Exception as e:
        pass

    quotes_df.sort_values(by='timestamp', inplace=True)

    if quotes_df.shape[0] > 0:
        quotes_df['symbol'] = stock_symbol
    return quotes_df

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

if params["cuda_installed"]:
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
def emarsi(mode, symbols, start_date, end_date, rank, size, params):

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

# %% [markdown]
# ### Main Logic with Serial Programming

# %%
def main_serial(params):
    # When using logging module, it crash iKernel so fallback to print method
    print("*" * 80)
    print("* Serial execution")
    print("*" * 80)
    
    previous_day = datetime.now() - timedelta(days=1)
    first_day = previous_day - timedelta(days=int(params["numberOfDays"]))

    start_date = first_day.strftime('%Y-%m-%dT%H:%M:%S')
    end_date = previous_day.strftime('%Y-%m-%dT%H:%M:%S')
    
    data_dir = './data'

    gpu_cores = 0
    rank = 0
    size = 1
    serial_fetching_stock_start_time = time.time()

    print(f"Rank: {rank}, Size: {size}")

    # Read symbols from the CSV file
    symbols = read_symbols_from_csvfile(os.environ["PROJECT_ROOT"] + "s-and-p-500-constituents/sandp500-20240310.csv")
    symbols = symbols[:params["numberOfStocks"]]


    # ************** #
    # * Core logic * #
    # ************** #
    results = emarsi("serial", symbols, start_date, end_date, rank, size, params)
    results = macd(results)

    serial_fetching_stock_end_time = time.time()
    print(f"Serial fetching stock price history quotes completed in {serial_fetching_stock_end_time - serial_fetching_stock_start_time} seconds")
    serial_elapsedtime = serial_fetching_stock_end_time - serial_fetching_stock_start_time
    return results, serial_elapsedtime

# %% [markdown]
# ### Main Logic with Hybrid Programming

# %%
def main_hybrid(params):
    # When using logging module, it crash iKernel so fallback to print method
    print("*" * 80)
    print("* Parallel execution")
    print("*" * 80)
  
    previous_day = datetime.now() - timedelta(days=1)
    first_day = previous_day - timedelta(days=int(params["numberOfDays"]))

    start_date = first_day.strftime('%Y-%m-%dT%H:%M:%S')
    end_date = previous_day.strftime('%Y-%m-%dT%H:%M:%S')
    
    data_dir = './data'

    # Create a lock for each GPU
    if params["cuda_installed"]:

        device = cuda.get_current_device()
        print(f"GPU name: {device.name.decode('utf-8')}")

        gpu_cores = len(cuda.gpus)
        print(f"GPU cores: {gpu_cores}")

    else:
        print("CUDA is not available")
        gpu_cores = 0
        

    # MPI WTime
    parallel_fetching_stock_start_time = MPI.Wtime()


    print(f"Rank: {rank}, Size: {size}")

    # Root process should scatter the symbols to all processes
    if rank == 0:

        # Read symbols from the CSV file
        symbols = read_symbols_from_csvfile(os.environ["PROJECT_ROOT"] + "s-and-p-500-constituents/sandp500-20240310.csv")
        symbols = symbols[:params["numberOfStocks"]]

        # Calculate how many symbols each process should receive
        symbols_per_process = len(symbols) // size
        if size > 1:
            remainder = len(symbols) % size
            if remainder != 0 and rank < remainder:
                symbols_per_process += 1

            # Scatter symbols to all processes and each process should receive length of symbols / size blocks
            local_symbols = [symbols[i:i + symbols_per_process] for i in range(0, len(symbols), symbols_per_process)]
        else:
          local_symbols = [symbols]

    else:
        local_symbols = None

    if comm:
        local_symbols = comm.scatter(local_symbols, root=0)

    # ************** #
    # * Core logic * #
    # ************** #
    print(f"params: {params}")

    results = emarsi("parallel", local_symbols, start_date, end_date, rank, size, params)
    if comm and rank > 0:
        # remote_result = comm.gather(results, root=0)
        comm.send(results, dest=0)
        return None, None

    elif rank == 0:
        
        for i in range(1, size):
            remote_results = comm.recv(source=i)
            results = pd.concat([results, remote_results])
            
        if params["cuda_installed"]:
            results = macd_gpu(results)
        else:
            results = macd(results)
        elapsed_time = 0.0;
        
        # MPI WTime
        parallel_fetching_stock_end_time = MPI.Wtime()
        print(f"Parallel fetching stock price history quotes completed in {parallel_fetching_stock_end_time - parallel_fetching_stock_start_time} seconds")
        elapsed_time = parallel_fetching_stock_end_time - parallel_fetching_stock_start_time
        
        return results, elapsed_time



# %% [markdown]
# ### Core Logic

# %%
def core_logic(row, params):
    # Remove data directory recursively if exists
    if os.path.exists("data"):
        os.system("rm -rf data")
    results, serial_elapsedtime = main_serial(params)
    results.rename(columns={
        "EMA12": "EMA12_S",
        "EMA26": "EMA26_S", 
        "RSI": "RSI_S", 
        "MACD": "MACD_S", 
    }, inplace=True)
    
    temp_result, temp_elapsedtime = main_hybrid(params)
    if temp_result is not None:
        results["EMA12_P"] = temp_result["EMA12"]
        results["EMA26_P"] = temp_result["EMA26"]
        results["RSI_P"] = temp_result["RSI"]
        results["MACD_P"] = temp_result["MACD"]
        

    if not os.path.exists(os.environ["PROJECT_ROOT"] + "outputs"):
        os.makedirs(os.environ["PROJECT_ROOT"] + "outputs")
    
    filename = os.environ["PROJECT_ROOT"] + f"outputs/results-{size}-{params['numberOfStocks']}-{params['numberOfDays']}.csv"
    print(f"filename: {filename}")
    
    results.to_csv(filename, index=False)
    print("Results saved to CSV file {filename}")
    
    return results.shape[0], serial_elapsedtime, temp_elapsedtime
        
    

# %% [markdown]
# ## Main Body

# %%

df = pd.DataFrame()
df["numberOfStocks"] = [10, 50, 100, 200, 400]
df["numberOfDays"] = [30, 90, 180, 365, 730]

df["numberOfRows"] = df["numberOfStocks"] * df["numberOfDays"]

# Fill zeros
df["serialElapsedTimes"] = [0.0] * len(df)
df["parallelElapsedTimes"] = [0.0] * len(df)
df["numberOfProcesses"] = [0] * len(df)

print(f"MainBody: Rank: {rank}, Size: {size}")

for index, row in df.iterrows():
    print(f"Processing {row['numberOfStocks']} stocks for {row['numberOfDays']} days")
    params["numberOfStocks"] = row["numberOfStocks"].astype(int)
    params["numberOfDays"] = row["numberOfDays"].astype(int)
    
    print(f"Params: {params}")
    numberOfRows, serialElapsedTime, parrallelElapsedTime = core_logic(row, params)
    df.loc[index, "numberOfProcesses"] = size
    df.loc[index, "numberOfRows"] = numberOfRows
    df.loc[index, "serialElapsedTimes"] = serialElapsedTime
    df.loc[index, "parallelElapsedTimes"] = parrallelElapsedTime
    
    
filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-{size}.csv"
df.to_csv(filename, index=False)
print(f"Saved stats to {filename}")
        


# %% [markdown]
# ## Export notebook into Python Script and Run with mpirun

# %% [markdown]
# ```bash
# mpirun -np 1 -mca opal_cuda_support 1 ~/miniconda3/envs/cp631-final/bin/python ~/cp631-final/cp631_final.py
# ```

# %% [markdown]
# ## Data Visualization

# %% [markdown]
# ## Performance Analysis

# %% [markdown]
# # Exit

# %%
if not params["in_notebook"]:
    exit(0)


