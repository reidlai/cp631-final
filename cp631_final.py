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
    if importlib.util.find_spec("opendatasets") is None:
        subprocess.run(["conda", "install", "-c", "conda-forge", "opendatasets", "-y"])
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
subprocess.run(["nvcc", "--version"])

# %%
if params["in_notebook"]:
    subprocess.run(["numba", "-s"])

# %% [markdown]
# ### Import required packages

# %%
# import datetime
import csv
import logging
import numpy as np
import opendatasets as od
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
od.download("https://www.kaggle.com/datasets/reidlai/s-and-p-500-constituents")

# %% [markdown]
# ## Stock Price History Download

# %%
class Row:
    def __init__(self, timestamp, open, high, low, close, adjclose, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume

def get_stock_price_history_quotes(stock_symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")

    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
    except Exception as e:
        logging.error(f"Symbol not found: {stock_symbol}")
        return []

    quotes = []
    for index, row in data.iterrows():
        quote = Row(index, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'])
        quotes.append(quote)

    quotes.sort(key=lambda x: x.timestamp)

    # convert quotes into dataframe
    quotes_df = pd.DataFrame([vars(quote) for quote in quotes])
    # add symbol column
    quotes_df['symbol'] = stock_symbol
    return quotes_df

# %% [markdown]
# ## Technical Analysis

# %% [markdown]
# ### CPU based technical indicator funtions

# %%
def ema(days, values):
    alpha = 2 / (days + 1)
    ema_values = [values[0]]  # start with the first value
    for value in values[1:]:
        ema_values.append(alpha * value + (1 - alpha) * ema_values[-1])
    return ema_values[-1]

def rsi(days, values):
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

def macd(values, short_period=12, long_period=26, signal_period=9):
    ema_short = ema(short_period, values)
    ema_long = ema(long_period, values)
    macd_line = np.array(ema_short) - np.array(ema_long)
    signal_line = ema(signal_period, macd_line.tolist())
    return macd_line, signal_line

# %% [markdown]
# ### GPU based technical indicator funtions

# %%
if params["cuda_installed"]:

    @cuda.jit
    def ema_cuda(values, ema_values, days, n, m):
        idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        if idx < n:
            alpha = 2.0 / (days + 1)
            for j in range(m):
                if j == 0:
                    ema_values[idx * m] = values[idx * m]  # start with the first value
                else:
                    ema_values[idx * m + j] = alpha * values[idx * m + j] + (1 - alpha) * ema_values[idx * m + j - 1]

    @cuda.jit
    def compute_gains_losses_cuda(values, gains, losses, n):
        idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        if idx < n:
            change = values[idx] - values[idx - 1]
            gains[idx] = max(change, 0)
            losses[idx] = max(-change, 0)

    @cuda.jit
    def macd_cuda(values, macd_values, signal_values, short_period, long_period, signal_period, n, m):
        idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        if idx < n:
            alpha_short = 2.0 / (short_period + 1)
            alpha_long = 2.0 / (long_period + 1)
            alpha_signal = 2.0 / (signal_period + 1)
            ema_short = 0
            ema_long = 0
            ema_signal = 0
            for j in range(m):
                if j < short_period:
                    ema_short = alpha_short * values[idx * m + j] + (1 - alpha_short) * ema_short
                if j < long_period:
                    ema_long = alpha_long * values[idx * m + j] + (1 - alpha_long) * ema_long
                macd_val = ema_short - ema_long
                if j < signal_period:
                    ema_signal = alpha_signal * macd_val + (1 - alpha_signal) * ema_signal
                macd_values[idx * m + j] = macd_val
                signal_values[idx * m + j] = ema_signal

    def ema_gpu(values, days):
        if values.ndim == 1:
            n = values.shape[0]
            m = 1
        else:
            n, m = values.shape

        ema_values = np.empty_like(values)

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        values_device = cuda.to_device(values)
        ema_values_device = cuda.to_device(ema_values)

        ema_cuda[grid_size, block_size](values_device, ema_values_device, days, n, m)
        ema_values = ema_values_device.copy_to_host()

        return ema_values

    def rsi_gpu(days, values):
        n = len(values)
        gains = np.empty_like(values)
        losses = np.empty_like(values)
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        values_device = cuda.to_device(values)
        gains_device = cuda.to_device(gains)
        losses_device = cuda.to_device(losses)

        compute_gains_losses_cuda[grid_size, block_size](values_device, gains_device, losses_device, n)
        gains = gains_device.copy_to_host()
        losses = losses_device.copy_to_host()

        avg_gain = np.sum(gains[:days]) / days
        avg_loss = np.sum(losses[:days]) / days
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi_value = 100 - (100 / (1 + rs))
        return rsi_value

    def macd_gpu(values, short_period=12, long_period=26, signal_period=9):
        # n, m = values.shape
        n = len(values)
        m = 1
        macd_values = np.empty_like(values)
        signal_values = np.empty_like(values)

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        values_device = cuda.to_device(values)
        macd_values_device = cuda.to_device(macd_values)
        signal_values_device = cuda.to_device(signal_values)


        macd_cuda[grid_size, block_size](values_device, macd_values_device, signal_values_device, short_period, long_period, signal_period, n, m)
        macd_values = macd_values_device.copy_to_host()
        signal_values = signal_values_device.copy_to_host()

        return macd_values, signal_values

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
# ### Obtain GPU lock

# %%
# Obtain available GPU and lock it; otherwise wait for available GPU
def obtain_available_gpu_lock(locks, params, rank, win):
    if params["mpi_installed"]:

        # Wait for available GPU
        while True:
            for i in range(len(locks)):
                # lock_status = locks[i].Get_attr(MPI.WIN_LOCK_STATUS)
                # if lock_status == 0:
                #     locks[i].lock(i)
                #     return i
                win.Lock(i)
                if locks[i] == -1:
                    locks[i] = rank
                    return i
                win.Unlock(i)
            # Sleep for random time to avoid busy waiting
            time.sleep(random.choice([0.1, 0.2, 0.3, 0.5, 0.7]))
    return None

# %% [markdown]
# ### Release GPU Lock

# %%
def release_gpu_lock(locks, gpu_index, params, rank, win):
    if params["mpi_installed"]:
        if locks[gpu_index] == rank:
            locks[gpu_index] = -1
            win.Unlock(gpu_index)
            return True
        return False
    else:
        return True

# %% [markdown]
# ### Core Logic

# %%
def core_logic(symbols, start_date, end_date, rank, size, params, locks, win):

    results = pd.DataFrame()
    # Fetch stock price history quotes using the local symbols
    for symbol in symbols:

        # Load the stock price history data into pandas DataFrame
        stock_price_history_df = get_stock_price_history_quotes(symbol, start_date, end_date)
        if stock_price_history_df.shape[0] > 0:

            # Calculate technical indicators using CUDA
            if params["cuda_installed"]:
                # gpu_id = obtain_available_gpu_lock(locks, params, rank, win)
                stock_price_history_df['EMA'] = ema_gpu(stock_price_history_df['close'].values, 12)
                # release_gpu_lock(locks, gpu_id, params, win)

                # gpu_id = obtain_available_gpu_lock(locks, params, rank, win)
                stock_price_history_df['RSI'] = rsi_gpu(14, stock_price_history_df['close'].values)
                # release_gpu_lock(locks, gpu_id, params, win)

                # gpu_id = obtain_available_gpu_lock(locks, params, rank, win)
                macd_values, signal_values = macd_gpu(stock_price_history_df['close'].values)
                # release_gpu_lock(locks, gpu_id, params, win)

                stock_price_history_df['MACD'] = macd_values
                stock_price_history_df['Signal'] = signal_values
            else:
                stock_price_history_df['EMA'] = stock_price_history_df['close'].rolling(window=12).mean()
                stock_price_history_df['RSI'] = stock_price_history_df['close'].rolling(window=14).apply(rsi, raw=True)
                # macd_values, signal_values = macd(stock_price_history_df['close'].values)
                stock_price_history_df['MACD'] = macd_values
                stock_price_history_df['Signal'] = signal_values

            results = pd.concat([results, stock_price_history_df])


    return results

# %% [markdown]
# ### Main Logic with Serial Programming

# %%
def main_serial(params):

    current_year = datetime.now().year
    previous_day = datetime.now() - timedelta(days=1)
    first_day_of_year = f"{current_year}-01-01"
    previous_day_str = previous_day.strftime("%Y-%m-%d")

    start_date = first_day_of_year + "T00:00:00"
    end_date = previous_day_str + "T23:59:59"
    data_dir = './data'

    gpu_cores = 0
    rank = 0
    size = 1
    serial_fetching_stock_start_time = time.time()

    print(f"Rank: {rank}, Size: {size}")

    # Read symbols from the CSV file
    symbols = read_symbols_from_csvfile(os.environ["PROJECT_ROOT"] + "s-and-p-500-constituents/sandp500-20240310.csv")


    # ************** #
    # * Core logic * #
    # ************** #
    results = core_logic(symbols, start_date, end_date, rank, size, params, None, None)


    serial_fetching_stock_end_time = time.time()
    print(f"Serial fetching stock price history quotes completed in {serial_fetching_stock_end_time - serial_fetching_stock_start_time} seconds")

# %% [markdown]
# ### Main Logic with Hybrid Programming

# %%
def main_hybrid(params):

    current_year = datetime.now().year
    previous_day = datetime.now() - timedelta(days=1)
    first_day_of_year = f"{current_year}-01-01"
    previous_day_str = previous_day.strftime("%Y-%m-%d")

    start_date = first_day_of_year + "T00:00:00"
    end_date = previous_day_str + "T23:59:59"
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

    if params["mpi_installed"] and params["cuda_installed"]:
        print(f"Initializing GPU locks")
        # locks = [MPI.Win.Create(None, 1, MPI.INFO_NULL, MPI.COMM_WORLD) for _ in range(gpu_cores)]
        # locks = [MPI.Win.Allocate(1, 1, MPI.INFO_NULL, MPI.COMM_WORLD) for _ in range(gpu_cores)]
        # for i in range(gpu_cores):
        #     locks[i].Fence(0)
        locks = np.zeros(gpu_cores, dtype='i')
        # locks = np.ascontiguousarray(np.zeros(gpu_cores, dtype='i'))
        # win = MPI.Win.Create(locks, comm=comm)
        win = MPI.Win.Allocate(gpu_cores, 1, MPI.INFO_NULL, comm)
        print(f"{len(locks)} GPU(s) are allocated")
    else:
      locks = []

    print(f"Rank: {rank}, Size: {size}")

    # Root process should scatter the symbols to all processes
    if rank == 0:

        # Read symbols from the CSV file
        symbols = read_symbols_from_csvfile(os.environ["PROJECT_ROOT"] + "s-and-p-500-constituents/sandp500-20240310.csv")

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

    results = core_logic(local_symbols, start_date, end_date, rank, size, params, locks, win)

    ## Gather the results from all processes
    remote_results = pd.DataFrame()
    if comm:
        remote_result = comm.gather(results, root=0)

    if rank == 0:
        results = pd.concat([results, remote_results])
        if params["mpi_installed"] and comm:
            display(results)

            # MPI WTime
            parallel_fetching_stock_end_time = MPI.Wtime()
            print(f"Parallel fetching stock price history quotes completed in {parallel_fetching_stock_end_time - parallel_fetching_stock_start_time} seconds")
        else:
            serial_fetching_stock_end_time = time.time()
            print(f"Serial fetching stock price history quotes completed in {serial_fetching_stock_end_time - serial_fetching_stock_start_time} seconds")


# %% [markdown]
# ### Main Body

# %%
if __name__ == "__main__":
    # Remove data directory recursively if exists
    if os.path.exists("data"):
        os.system("rm -rf data")
    main_serial(params)
    main_hybrid(params)

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


