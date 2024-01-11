package mpp

// import (
// 	"fmt"
// 	"github.com/stretchr/testify"
// 	"runtime"
// 	"sync"
// 	"testing"
// )

import (
	"runtime"
	"testing"

	// "fmt"
	"github.com/shirou/gopsutil/cpu"
	"github.com/stretchr/testify/suite"
	// "github.com/shirou/gopsutil/cpu"
	// "github.com/shirou/gopsutil/disk"
	// "github.com/shirou/gopsutil/host"
	// "github.com/shirou/gopsutil/mem"
	// "github.com/shirou/gopsutil/net"
	// "net/http"
	// "runtime"
	// "strconv"
)

// func main() {
// 	// Set the maximum number of CPUs to utilize all available cores
// 	fmt.Printf("Number of CPUs: %d\n", runtime.NumCPU())
// 	runtime.GOMAXPROCS(runtime.NumCPU())

// 	// Define a wait group to ensure all goroutines finish before exiting
// 	var wg sync.WaitGroup

// 	// Get the total number of CPUs
// 	totalCPUs := runtime.NumCPU()

// 	// Launch goroutines for each CPU
// 	for i := 0; i < totalCPUs; i++ {
// 		wg.Add(1)
// 		go func(cpuID int) {
// 			defer wg.Done()

// 			// Perform some computation or task here
// 			// For example, you can simulate a heavy computation
// 			// by running a loop for a large number of iterations
// 			for j := 0; j < 1000000000; j++ {
// 				// Perform computation or task
// 			}

// 			fmt.Printf("Completed computation on CPU %d\n", cpuID)
// 		}(i)
// 	}

// 	// Wait for all goroutines to finish
// 	wg.Wait()

// 	fmt.Println("Finished executing on all CPUs")
// }

type MppTestSuite struct {
	suite.Suite
}

func (s *MppTestSuite) TestNumberOfLogicalCPU() {
	runtimeOS := runtime.GOOS
	s.T().Logf(`OS: %s`, runtimeOS)

	cpuInfo, err := cpu.Info()
	s.NoError(err)
	s.T().Logf(`Physical CPU cores: %v`, cpuInfo[0].Cores)

	s.T().Logf(`Number of Logical CPU: %d`, runtime.NumCPU())

	maxProcs := runtime.GOMAXPROCS(0)
	s.T().Logf(`MaxProcs: %d`, maxProcs)

	s.Assert().GreaterOrEqual(int32(maxProcs), int32(cpuInfo[0].Cores), `MaxProcs should be greater or equal to physical CPU cores`)
}

func TestMppTestSuite(t *testing.T) {
	suite.Run(t, new(MppTestSuite))
}
