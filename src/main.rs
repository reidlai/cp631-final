use cp631_final::stock_price_history;
use csv;
use log;
use env_logger;
use mpi::traits::Communicator;

fn main() {
    // Initialize logger
    env_logger::init();

    // Read symbols from CSV file
    log::info!("Reading symbols from CSV file");
    let symbols = read_symbols_from_csvfile("./sandp500-20240310.csv");
    // log available symbols
    log::debug!("Symbols: {:?}", symbols.clone());

    // Call serial programming
    log::info!("Running serial programming to fetch stock price history quotes for each symbol");
    serial_programming(symbols.clone());
    log::info!("Serial programming completed");

    parallel_programming(symbols.clone());

}

fn serial_programming(symbols: Vec<String>) {
    // Serial programming
    let serial_fetching_stock_start_time = std::time::Instant::now();
    for symbol in symbols.iter() {
        stock_price_history::fetch_stock_price_history_quotes(symbol, "2021-01-01T00:00:00+00:00", "2021-01-31T23:59:59+00:00", "./data");
    }
    let serial_fetching_stock_end_time = serial_fetching_stock_start_time.elapsed();
    log::info!("Serial fetching stock price history quotes completed in {} seconds", serial_fetching_stock_end_time.as_secs());
}

fn parallel_programming(symbols: Vec<String>) {

    // MPI WTime
    let parallel_fetching_stock_start_time = mpi::time();

    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;
    let symbols_per_process = symbols.len() / (size as usize);
    let remainer = symbols.len() % (size as usize);
    if remainer != 0 && rank < remainder {
        symbols_per_process += 1; 
    }
    // Split the symbols into local chunks based on symbols_per_process and rank id
    let local_chunk = symbols[rank * symbols_per_process..(rank + 1) * symbols_per_process].to_vec();
    for symbol in local_chunk.iter() {
        stock_price_history::fetch_stock_price_history_quotes(symbol, "2021-01-01T00:00:00+00:00", "2021-01-31T23:59:59+00:00", "./data");
    }

    // MPI WTime
    let parallel_fetching_stock_end_time = mpi::time();
    log::info!("Parallel fetching stock price history quotes completed in {} seconds", parallel_fetching_stock_end_time - parallel_fetching_stock_start_time);
}

fn read_symbols_from_csvfile(csvfile_path: &str) -> Vec<String> {
    let mut symbols = Vec::new();
    // use CSV reader to read symbols from file and skip the first header row
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csvfile_path)
        .unwrap();
    for record in reader.records() {
        let record = record.unwrap();
        symbols.push(record[0].to_string());
    }
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_symbols_from_csvfile() {
        let symbols = read_symbols_from_csvfile("./sandp500-20240310.csv");
        assert_eq!(symbols.len(), 505 as usize);
    }
}


