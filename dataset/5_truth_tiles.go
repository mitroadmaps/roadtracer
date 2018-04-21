package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"

	"image"
	"image/color"
	"image/png"
	"fmt"
	"math"
	"os"
	"runtime"
)

const ZOOM = 18

func main() {
	graphDir := os.Args[1]
	outDir := os.Args[2]

	regions := lib.GetRegions()
	rtrees := make(map[string]common.Rtree)
	for _, region := range regions {
		fmt.Printf("reading graph for region %s\n", region.Name)
		graph, err := common.ReadGraph(fmt.Sprintf("%s/%s.graph", graphDir, region.Name))
		if err != nil {
			panic(err)
		}
		rtrees[region.Name] = graph.Rtree()
	}

	fmt.Println("initializing tasks")
	type Task struct {
		Label string
		Region string
		Rect common.Rectangle
	}

	var tasks []Task
	for _, region := range regions {
		for x := -region.RadiusX; x < region.RadiusX; x++ {
			for y := -region.RadiusY; y < region.RadiusY; y++ {
				rect := common.Rectangle{
					common.Point{float64(x), float64(y)}.Scale(4096),
					common.Point{float64(x) + 1, float64(y) + 1}.Scale(4096),
				}
				tasks = append(tasks, Task{
					Label: fmt.Sprintf("%s_%d_%d", region.Name, x, y),
					Region: region.Name,
					Rect: rect,
				})
			}
		}
	}

	processTask := func(task Task, threadID int) {
		lengths := task.Rect.Lengths()
		sizex := int(math.Round(lengths.X))
		sizey := int(math.Round(lengths.Y))
		values := make([][]uint8, sizex)
		for i := range values {
			values[i] = make([]uint8, sizey)
		}
		edges := rtrees[task.Region].Search(task.Rect.AddTol(20))
		for _, edge := range edges {
			segment := edge.Segment()
			start := segment.Start.Sub(task.Rect.Min)
			end := segment.End.Sub(task.Rect.Min)
			for _, pos := range common.DrawLineOnCells(int(start.X), int(start.Y), int(end.X), int(end.Y), sizex, sizey) {
				for i := -4; i <= 4; i++ {
					for j := -4; j <= 4; j++ {
						d := math.Sqrt(float64(i * i + j * j))
						if d > 4 {
							continue
						}
						x := pos[0] + i
						y := pos[1] + j
						if x >= 0 && x < sizex && y >= 0 && y < sizey {
							values[x][y] = 255
						}
					}
				}
			}
		}

		img := image.NewGray(image.Rect(0, 0, sizex, sizey))
		for i := 0; i < sizex; i++ {
			for j := 0; j < sizey; j++ {
				img.SetGray(i, j, color.Gray{values[i][j]})
			}
		}

		f, err := os.Create(fmt.Sprintf("%s/%s_osm.png", outDir, task.Label))
		if err != nil {
			panic(err)
		}
		if err := png.Encode(f, img); err != nil {
			panic(err)
		}
		f.Close()
	}

	n := runtime.NumCPU()
	fmt.Printf("launching %d workers\n", n)
	taskCh := make(chan Task)
	doneCh := make(chan bool)
	for threadID := 0; threadID < n; threadID++ {
		go func(threadID int) {
			for task := range taskCh {
				processTask(task, threadID)
			}
			doneCh <- true
		}(threadID)
	}
	fmt.Println("running tasks")
	for i, task := range tasks {
		if i % 10 == 0 {
			fmt.Printf("... task progress: %d/%d\n", i, len(tasks))
		}
		taskCh <- task
	}
	close(taskCh)
	for threadID := 0; threadID < n; threadID++ {
		<- doneCh
	}
}
