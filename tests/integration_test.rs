#[cfg(test)]
extern crate rspec;

#[cfg(test)]
#[macro_use]
extern crate hamcrest;

mod cpu;

pub fn main() {
  cpu::tests();
}
