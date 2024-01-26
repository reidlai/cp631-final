
use num_cpus;
use hamcrest::prelude::*;

#[test]
pub fn test_cpu_core_checking() {
  let logical_cores = num_cpus::get(); 
  let physical_cores = num_cpus::get_physical();
  assert_that!(logical_cores, greater_than_or_equal_to(physical_cores ));
}