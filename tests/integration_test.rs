#[cfg(test)]
extern crate rspec;

#[cfg(test)]
#[macro_use]
extern crate hamcrest;

mod cpu;
mod mpi;

use std::{io};
use std::sync::{Arc};
use cp631_final::technical_analysis;

pub fn main() {
  let logger = Arc::new(rspec::Logger::new(io::stdout()));
  let configuration = rspec::ConfigurationBuilder::default().build().unwrap();
  let runner = rspec::Runner::new(configuration, vec![logger]);

  #[derive(Clone, Debug)]
  struct Environment;

  let environment = Environment{};
  runner.run(&rspec::describe("Integration Tests", environment, |ctx| {

    // CPU Core Checking
    ctx.specify("CPU Core Checking", |ctx| {

      ctx.it("logical cpu cores must be greater than or equal to physical cpu cores", |_| {
        cpu::test_cpu_core_checking();
      });
      
    });

    // MPI Testing
    ctx.specify("MPI Testing", |ctx| {

      ctx.it("MPI must be able to initialize", |_| {
        mpi::test_mpi_initialization();
      });

    });

    // EMA function testing
    ctx.specify("EMA function testing", |ctx| {

      ctx.it("EMA function must return correct value", |_| {
        let ema_values = technical_analysis::exponentail_moving_average(2, vec![1.0, 2.0, 3.0]);
        assert_ne!(ema_values, 2.0);
        assert_eq!(ema_values, 2.5555555555555554);
      });

    });

    // RSI function testing
    ctx.specify("RSI function testing", |ctx| {

      ctx.it("RSI function must return correct value", |_| {
        let rsi_value = technical_analysis::relative_strength_index(2, vec![1.0, 2.0, 3.0]);
        assert_ne!(rsi_value, 98.0);
        assert_eq!(rsi_value, 98.78048780487805);
      });

    });
    

  }));

}
