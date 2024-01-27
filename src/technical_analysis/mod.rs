use ta::indicators::{ExponentialMovingAverage, RelativeStrengthIndex};
use ta::{Next};

pub fn exponentail_moving_average(days: usize, prices: Vec<f64>) -> f64 {
    let mut ema = ExponentialMovingAverage::new(days).unwrap();
    let mut ema_value = 0.0;
    for price in prices {
        ema_value = ema.next(price);
    }
    return ema_value;
}

pub fn relative_strength_index(days: usize, ema_values: Vec<f64>) -> f64 {
  let mut rsi = RelativeStrengthIndex::new(days).unwrap();
  let mut rsi_value = 0.0;
  for ema_value in ema_values {
      rsi_value = rsi.next(ema_value);
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
  }

  #[test]
  fn test_relative_strength_index() {
    let rsi_value = relative_strength_index(2, vec![1.0, 2.0, 3.0]);
    assert_ne!(rsi_value, 98.0);
    assert_eq!(rsi_value, 98.78048780487805);
  }
}

// pub fn moving_average_convergence_divergence() {
//     println!("Hello, moving_average_convergence_divergence!");
// }