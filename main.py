from mpi4py import MPI
import csv
import stock_price_history
import technical_analysis 
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def main():
    current_year = datetime.now().year
    previous_day = datetime.now() - timedelta(days=1)
    first_day_of_year = f"{current_year}-01-01"
    previous_day_str = previous_day.strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description='Fetch stock price history quotes.')
    parser.add_argument('--start-date', type=str, required=False, default=first_day_of_year, help='Start date in the format "YYYY-MM-DD"')
    parser.add_argument('--end-date', type=str, required=False, default=previous_day_str, help='End date in the format "YYYY-MM-DD"')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory to store the fetched data')

    args = parser.parse_args()
    
    # Create a lock for each GPU
    num_gpus = cuda.Device.count()
    locks = [MPI.WIN.Create(None, 1, MPI.INFO_NULL, MPI.COMM_WORLD) for _ in range(num_gpus)]
    for i in range(num_gpus):
        locks[i].Fence(0)
    

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
    
    # Root process should scatter the symbols to all processes
    if rank == 0:
        
        # Read symbols from the CSV file
        symbols = read_symbols_from_csvfile("./sandp500-20240310.csv")
        
        # Calculate how many symbols each process should receive
        symbols_per_process = len(symbols) // size
        remainder = len(symbols) % size
        if remainder != 0 and rank < remainder:
            symbols_per_process += 1
            
        # Scatter symbols to all processes and each process should receive length of symbols / size blocks
        local_symbols = [symbols[i:i + symbols_per_process] for i in range(0, len(symbols), symbols_per_process)]
        
    else:
        local_symbols = None
    
    if comm:
        local_symbols = comm.scatter(local_symbols, root=0)
    
    # ************** #
    # * Core logic * #
    # ************** #
    
    # Fetch stock price history quotes using the local symbols
    for symbol in local_symbols:

        # Load the stock price history data into pandas DataFrame
        stock_price_history_df = stock_price_history.get_stock_price_history_quotes(symbol, args.start_date, args.end_date, args.data_dir)
        
        # Lock the GPU by GPU index if multiple GPUs are available
        if num_gpus > 1:
            gpu_index = rank % num_gpus
            locks[gpu_index].Lock(gpu_index)
        else if num_gpus == 1:
            gpu_index = 0
            locks[gpu_index].Lock(gpu_index)
        
        # Call EMA kernel function
        
        # Call RSI kernel function
        
        # MPI Gather the results from all processes
        
    
    if rank == 0:
        
        if comm:
            # Gather the results from all processes
            # Merge the results from all processes
            
            # MPI WTime
            parallel_fetching_stock_end_time = MPI.Wtime()
            print(f"Parallel fetching stock price history quotes completed in {parallel_fetching_stock_end_time - parallel_fetching_stock_start_time} seconds")
        else:
            serial_fetching_stock_end_time = time.time()
            print(f"Serial fetching stock price history quotes completed in {serial_fetching_stock_end_time - serial_fetching_stock_start_time} seconds")
            
def read_symbols_from_csvfile(csvfile_path):
    symbols = []
    with open(csvfile_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            symbols.append(row[0])  # Assuming the symbol is the first column
    return symbols

if __name__ == "__main__":
    main()