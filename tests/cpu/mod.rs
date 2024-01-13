use std::io;
use std::sync::Arc;
use num_cpus;
use hamcrest::prelude::*;
use mpi;

#[test]
pub fn tests() {

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
        let logical_cores = num_cpus::get(); 
        let physical_cores = num_cpus::get_physical();
        assert_that!(logical_cores, greater_than_or_equal_to(physical_cores ));
      });
      
    });

    // MPI Checking
    ctx.specify("MPI Checking", |ctx| {

      ctx.it("mpi must be initialized", |_| {
        mpi::initialize().unwrap();
        assert_that!(mpi::is_initialized(), is(equal_to(true)));
        mpi::finalize().unwrap();
      });

    });

  }));

}