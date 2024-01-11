package main

import (
	"fmt"
	"runtime"
	"sync"
)

func main() {
	// Set the maximum number of CPUs to utilize all available cores
	fmt.Printf("Number of CPUs: %d\n", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Define a wait group to ensure all goroutines finish before exiting
	var wg sync.WaitGroup

	// Get the total number of CPUs
	totalCPUs := runtime.NumCPU()

	// Launch goroutines for each CPU
	for i := 0; i < totalCPUs; i++ {
		wg.Add(1)
		go func(cpuID int) {
			defer wg.Done()

			// Perform some computation or task here
			// For example, you can simulate a heavy computation
			// by running a loop for a large number of iterations
			for j := 0; j < 1000000000; j++ {
				// Perform computation or task
			}

			fmt.Printf("Completed computation on CPU %d\n", cpuID)
		}(i)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	fmt.Println("Finished executing on all CPUs")
}
