use hamcrest::prelude::*;
use mpi::traits::*;

#[test]
pub fn test_mpi_initialization() {

  let universe = mpi::initialize().unwrap();
  let world = universe.world();

  assert_that!(world.size(), greater_than(0));
  drop(world);
  drop(universe);
}