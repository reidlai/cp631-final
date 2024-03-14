use yahoo_finance_api as yahoo;
use chrono::{DateTime, Utc};
use time::OffsetDateTime;
use csv;
use serde;
use log;

#[derive(serde::Serialize)]
struct Row<'a> {
    timestamp: &'a str,
    open: &'a str,
    high: &'a str,
    low: &'a str,
    close: &'a str,
    adjclose: &'a str,
    volume: &'a str,
}

pub fn get_stock_price_history_quotes(stock_symbol: &str, start_date: &str, end_date: &str) -> Vec<yahoo::Quote> {
    let provider = yahoo::YahooConnector::new();
    let start_date = DateTime::parse_from_rfc3339(start_date).unwrap().with_timezone(&Utc);
    let end_date = DateTime::parse_from_rfc3339(end_date).unwrap().with_timezone(&Utc);
    let start_date = match OffsetDateTime::from_unix_timestamp(start_date.timestamp()) {
        Ok(start_date) => start_date,
        Err(_) => panic!("Invalid start date"),
    };
    let end_date = match OffsetDateTime::from_unix_timestamp(end_date.timestamp()) {
        Ok(end_date) => end_date,
        Err(_) => panic!("Invalid end date"),
    };
    let resp = match provider.get_quote_history(stock_symbol, start_date, end_date) {
        Ok(resp) => resp,
        Err(_) => {
            // skip symbol if it is not found
            // println!("Symbol not found: {}", stock_symbol);
            log::error!("Symbol not found: {}", stock_symbol);
            return Vec::new();
        }
    };
    let mut quotes = resp.quotes().unwrap();
    quotes.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    return quotes;
}

pub fn fetch_stock_price_history_quotes(stock_symbol: &str, start_date: &str, end_date: &str, dir_path: &str) {
    // Call get_stock_price_history_quotes and store the result in CSV file
    let stock_price_history_quotes = get_stock_price_history_quotes(stock_symbol, start_date, end_date);

    // if directory does not exist, create it
    if !std::path::Path::new(dir_path).exists() {
        std::fs::create_dir(dir_path).unwrap();
    }

    let mut wtr = csv::WriterBuilder::new()
        .has_headers(true)
        .from_path(format!("{}/{}.csv", dir_path, stock_symbol))
        .unwrap();
    for quote in stock_price_history_quotes.iter() {
        wtr.serialize(Row {
            timestamp: &quote.timestamp.to_string(),
            open: &quote.open.to_string(),
            high: &quote.high.to_string(),
            low: &quote.low.to_string(),
            close: &quote.close.to_string(),
            adjclose: &quote.adjclose.to_string(),
            volume: &quote.volume.to_string(),
        }).unwrap();
    }
    wtr.flush().unwrap();
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_get_stock_price_history_quotes() {
        let stock_price_history_quotes = get_stock_price_history_quotes("AAPL", "2021-01-01T00:00:00+00:00", "2021-01-31T23:59:59+00:00");

        // print all quotes
        for quote in stock_price_history_quotes.iter() {
            println!("Quote: {:?}", quote);
        }
        assert_eq!(stock_price_history_quotes.len(), 19 as usize);

        // Check the first quote
        let first_quote = stock_price_history_quotes.get(0).unwrap();
        assert_eq!(first_quote.timestamp, 1609770600 as u64);
    
        // Check the last quote
        let last_quote = stock_price_history_quotes.get(18).unwrap();
        assert_eq!(last_quote.timestamp, 1611930600 as u64);

    }

    #[test]
    fn test_fetch_stock_price_history_quotes() {
        // Check directory exists
        let dir_path = "./data";

        // if directory does not exist, create it
        if !std::path::Path::new(dir_path).exists() {
            std::fs::create_dir(dir_path).unwrap();
        }

        fetch_stock_price_history_quotes("AAPL", "2021-01-01T00:00:00+00:00", "2021-01-31T23:59:59+00:00", "./data");

        // Check the number of record in file is same as the result of get_stock_price_history_quotes
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(format!("{}/{}.csv", dir_path, "AAPL"))
            .unwrap();
        let mut count = 0;
        for _ in rdr.records() {
            count += 1;
        }
        // Call get_stock_price_history and count the number of records
        let stock_price_history_quotes = get_stock_price_history_quotes("AAPL", "2021-01-01T00:00:00+00:00", "2021-01-31T23:59:59+00:00");
        assert_eq!(count, stock_price_history_quotes.len());

    }
}
