use ta::{indicators::{ExponentialMovingAverage, RelativeStrengthIndex}, Next};

pub fn exponentail_moving_average(days: usize, prices: Vec<f64>) -> f64 {
  let mut ema = ExponentialMovingAverage::new(days).unwrap();
  let mut ema_value = 0.0;
  for price in prices {
    ema_value = ema.next(price); // Call the next method on ema
  }
  return ema_value;
}

pub fn relative_strength_index(days: usize, ema_values: Vec<f64>) -> f64 {
  let mut rsi = RelativeStrengthIndex::new(days).unwrap();
  let mut rsi_value = 0.0;
  for ema_value in ema_values {
      rsi_value = rsi.next(ema_value)
  }
  return rsi_value;
}

// Unit Tests
#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_exponentail_moving_average() {
    let ema_values = exponentail_moving_average(2, vec![1.0, 2.0, 3.0]);
    assert_ne!(ema_values, 2.0);
    assert_eq!(ema_values, 2.5555555555555554);

    let prices = vec![22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
                      22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
                      23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17];
    let ema = exponentail_moving_average(20, prices);
    print!("EMA: {}\n", ema);
    let rounded_ema = (ema * 100.0).round() / 100.0; // Round to 2 decimal places
    assert!((rounded_ema - 22.98).abs() < 0.01, "EMA should be approximately 22.22");
  }

  #[test]
  fn test_relative_strength_index() {
    let rsi_value = relative_strength_index(2, vec![1.0, 2.0, 3.0]);
    assert_ne!(rsi_value, 98.0);
    assert_eq!(rsi_value, 98.78048780487805);

    let ema_values = vec![22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
                          22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
                          23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17];
    let rsi = relative_strength_index(14, ema_values);
    print!("RSI: {}\n", rsi);
    let expected_rsi = 31.62; // Replace with the expected RSI value
    assert!((rsi - expected_rsi).abs() < 0.01, "RSI should be approximately equal to the expected value");
  }
}

