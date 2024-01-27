#[cfg(test)]
extern crate rspec;

#[cfg(test)]
#[macro_use]
extern crate hamcrest;

use std::{io};
use std::sync::{Arc};

mod cpu;
mod mpi;

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
    

  }));

}
