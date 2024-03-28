# %% [markdown]
# ## Prerequisites

# %% [markdown]
# ### Course Server Setup
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
# conda install --name cp631-final requirement.txt
# ```
# 
# #### SSH Tunneling
# 
# To allow local machine connecting to Jupyter Notebook server running in course server, VPN connection must be up and running.  Then you can use SSH Tunnelling to forward all traffic of port 8888 in local macine to course server.
# 
# ```bash
# ssh -L 8888:localhost:8888 wlai11@mcs1.wlu.ca
# ```
# 
# #### Start Jupyter Notebook
# 
# Once the shell has been spawn in remote server, run the following command to start jupyter notebook server with new conda environment cp631-final
# 
# ```bash
# conda activate cp631-final
# jupyter notebook --no-browser --port=8888
# ```

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
import subprocess
import importlib.util

if params["in_notebook"] and importlib.util.find_spec("distro") is None:
    # !conda install distro -y
    subprocess.run(["conda", "install", "distro", "-y"])

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

# %%
import subprocess

def is_mpi_installed():
    try:
        if params["is_macos"]:
          subprocess.check_output(["/usr/local/bin/mpirun", "--version"])
        else:
          subprocess.check_output(["mpirun", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

params["mpi_installed"] = is_mpi_installed()
print(f'MPI installed: {params["mpi_installed"]}')

if not params["mpi_installed"]:
    print("[FATAL] MPI is not installed")

# %% [markdown]
# ### Check if NVIDIA CUDA toolkit installed
# 
# Use the numba command to see if MPI is up and running.

# %%
if params["in_notebook"] and importlib.util.find_spec("numba") is None:
    subprocess.run(["conda", "install", "numba=0.55.0", "-y"])

# %%
from numba import cuda

def is_cuda_installed():
    try:
        cuda.detect()
        return True
    except cuda.CudaSupportError:
        return False

params["cuda_installed"] = is_cuda_installed()    
print(f'CUDA installed: {params["cuda_installed"]}')

if not params["cuda_installed"]:
    print("[FATAL] CUDA is not installed")

# %% [markdown]
# ### Install MPI and CUDA if not installed

# %% [markdown]
# **Reminder**: Because latest Macbook does not bundle with NVIDIA CUDA compatible GPU and CUDA toolkits since at least CUDA 4.0 have not supported an ability to run cuda code without a GPU, this program cannot support MacOS environment.

# %% [markdown]
# If mpi_installed of the above result show False, please install openmpi binary and library based on your platform.
# 
# In Ubuntu you can install Open MPI as follow
# 
# ```bash
# sudo apt update
# sudo apt install openmpi-bin
# sudo apt install libopenmpi-dev
# ```
# 
# The following code will install Open MPI in Google Colab

# %%
import os

if params["in_notebook"]:
    if params["in_colab"] and not params["mpi_installed"]:
        print("Installing MPI")
        subprocess.run(["conda", "install", "openmpi", "-y" ])
        print("MPI installed")
    elif params["mpi_installed"]:
        print("MPI is installed")

# %% [markdown]
# if cuda_installed show False, please install NVIDIA CUDA toolkit in your platform
# 
# In Ubuntu (except Ubuntu WSL 2.0 under Windows 10/11) you can install CUDA as follow
# 
# ```bash
# sudo apt update
# sudo apt install -y gpupg2
# wget https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/cuda-repo-debian10_10.2.89-1_amd64.deb
# sudo dpkg -i cuda-repo-debian10_10.2.89-1_amd64.deb
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
# sudo apt update
# sudo apt-get install cuda
# ```
# 
# Under Google Colab, cuda is bundled.

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

# %%
if params["in_notebook"]:
    subprocess.run(["conda", "install", "pip", "-y"])
    if importlib.util.find_spec("mpi4py") is None:
        subprocess.run(["conda", "install", "-c", "conda-forge", "mpi4py=3.1.4", "-y"])
    if importlib.util.find_spec("kaggle") is None:
        subprocess.run(["conda", "install", "-c", "conda-forge", "kaggle", "-y"])
    # if importlib.util.find_spec("opendatasets") is None:
    #     subprocess.run(["conda", "install", "-c", "conda-forge", "opendatasets", "-y"])
    if importlib.util.find_spec("yfinance") is None:
        subprocess.run(["conda", "install", "-c", "conda-forge", "yfinance", "-y"])

    if not params["in_colab"]:
        print("Installing required packages for local environment")
        if importlib.util.find_spec("numpy") is None:
            subprocess.run(["conda", "install", "numpy", "-y"])
        if importlib.util.find_spec("matplotlib") is None:
            subprocess.run(["conda", "install", "matplotlib", "-y"])
        if importlib.util.find_spec("seaborn") is None:
            subprocess.run(["conda", "install", "seaborn", "-y"])
        if importlib.util.find_spec("pandas") is None:
            subprocess.run(["conda", "install", "pandas", "-y"])
        
        if params["cuda_installed"]:
            if importlib.util.find_spec("cudatoolit") is None:
                subprocess.run(["conda", "install", "cudatoolkit", "-y"])

        print("Common required packages installed")


# %% [markdown]
# Check numba info

# %%
if params["cuda_installed"]:
    subprocess.run(["nvcc", "--version"])

# %%
if params["in_notebook"] and params["cuda_installed"]:
    subprocess.run(["numba", "-s"])

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

if params["mpi_installed"]:
    from mpi4py import MPI

if params["cuda_installed"]:
    from numba import cuda, float32


# %% [markdown]
# ## S&P 500 Constituents Dataset Download
# 
# I will first need to download S&P 500 constituents from my Kaggle repository

# %%
# od.download("https://www.kaggle.com/datasets/reidlai/s-and-p-500-constituents")
kaggle.api.authenticate()
kaggle.api.dataset_download_files('reidlai/s-and-p-500-constituents', path="s-and-p-500-constituents", unzip=True)

# %% [markdown]
# ## Stock Price History Download

# %%
def get_stock_price_history_quotes(stock_symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
    
    quotes_df = pd.DataFrame(columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'adjclose', 'volume'])
    
    if "." in stock_symbol:
        return None

    try:
        # TODO: fixing kernel crash when passing progress=False to yf.download function
        # data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        data = yf.download(stock_symbol, start=start_date, end=end_date)
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

# %%
def ema(values, days=12):
    alpha = 2 / (days + 1)
    ema_values = np.empty_like(values)  # create an array to store all EMA values
    ema_values[0] = values[0]  # start with the first value
    for i in range(1, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


# %% [markdown]
# ### RSI

# %%
def rsi(values, days=14):
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

# %%

def macd(df, short_period=12, long_period=26, signal_period=9):

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
        cuda.synchronize()
        macd = macd_device.copy_to_host()
        cuda.current_context().memory_manager.deallocations.clear()
        df["MACD"] = macd
        return df

# %% [markdown]
# ## Core Main Program

# %% [markdown]
# ### Read CSV files

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
# ### Calculating EMA12, EMA26 and RSI

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
        
    if params["mpi_installed"]:
        # Initialize MPI
        comm = MPI.COMM_WORLD

        # check if mpi is initialized
        if comm:
            rank = comm.Get_rank()
            size = comm.Get_size()

            # MPI WTime
            parallel_fetching_stock_start_time = MPI.Wtime()
        else:
            rank = 0
            size = 1
            serial_fetching_stock_start_time = time.time()
    else:
        rank = 0
        size = 1
        serial_fetching_stock_start_time = time.time()

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

    ## Gather the results from all processes
    remote_results = pd.DataFrame()
    if comm:
        remote_result = comm.gather(results, root=0)

    if rank == 0:
        results = pd.concat([results, remote_results])
        if params["cuda_installed"]:
            results = macd_gpu(results)
        else:
            results = macd(results)
        elapsed_time = 0.0;
        if params["mpi_installed"] and comm:
            # MPI WTime
            parallel_fetching_stock_end_time = MPI.Wtime()
            print(f"Parallel fetching stock price history quotes completed in {parallel_fetching_stock_end_time - parallel_fetching_stock_start_time} seconds")
            elapsed_time = parallel_fetching_stock_end_time - parallel_fetching_stock_start_time
        else:
            serial_fetching_stock_end_time = time.time()
            print(f"Serial fetching stock price history quotes completed in {serial_fetching_stock_end_time - serial_fetching_stock_start_time} seconds")
            elapsed_time = serial_fetching_stock_end_time - serial_fetching_stock_start_time
        return results, elapsed_time, size
    else:
        return None, None, None
    


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
    
    temp_result, temp_elapsedtime, temp_size = main_hybrid(params)
    if temp_result is not None:
        results["EMA12_P"] = temp_result["EMA12"]
        results["EMA26_P"] = temp_result["EMA26"]
        results["RSI_P"] = temp_result["RSI"]
        results["MACD_P"] = temp_result["MACD"]
        

    if not os.path.exists(os.environ["PROJECT_ROOT"] + "outputs"):
        os.makedirs(os.environ["PROJECT_ROOT"] + "outputs")
        
    
    
    filename = os.environ["PROJECT_ROOT"] + f"outputs/results-{temp_size}-{params['numberOfStocks']}-{params['numberOfDays']}.csv"
    print(f"filename: {filename}")
    
    results.to_csv(filename, index=False)
    
    print("Returning df from core_logic")
    return temp_size, results.shape[0], serial_elapsedtime, temp_elapsedtime
        
    

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

for index, row in df.iterrows():
    print(f"Processing {row['numberOfStocks']} stocks for {row['numberOfDays']} days")
    params["numberOfStocks"] = row["numberOfStocks"].astype(int)
    params["numberOfDays"] = row["numberOfDays"].astype(int)
    
    print(f"Params: {params}")
    numberOfProcesses, numberOfRows, serialElapsedTime, parrallelElapsedTime = core_logic(row, params)
    df.loc[index, "numberOfProcesses"] = numberOfProcesses
    df.loc[index, "numberOfRows"] = numberOfRows
    df.loc[index, "serialElapsedTimes"] = serialElapsedTime
    df.loc[index, "parallelElapsedTimes"] = parrallelElapsedTime
    
    print("Received df from core_logic")
    
filename = os.environ["PROJECT_ROOT"] + f"outputs/stats-{numberOfProcesses}.csv"
df.to_csv(filename, index=False)

# Clean up MPI, CUDA and data
if params["mpi_installed"]:
    MPI.Finalize()
    print("MPI Finalized")

if params["cuda_installed"]:
    cuda.close()
    print("CUDA closed")
    

print(df)
        


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


