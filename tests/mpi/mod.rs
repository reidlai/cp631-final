use mpi::traits::*;
use hamcrest::prelude::*;

#[test]
pub fn test_mpi_initialization() {
  let universe = mpi::initialize().unwrap();
  let world = universe.world();
  assert_that!(world.size(), greater_than_or_equal_to(1));
}