use cp631_final::technical_analysis;

fn main() {
    println!("Hello, world!");
    let ema_values = technical_analysis::exponentail_moving_average(2, vec![1.0, 2.0, 3.0]);
    println!("EMA: {}", ema_values);
    let rsi_value = technical_analysis::relative_strength_index(2, vec![1.0, 2.0, 3.0]);
    println!("RSI: {}", rsi_value);
}

