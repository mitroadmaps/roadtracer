package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"fmt"
	"os"
)

func main() {
	graph, err := common.ReadGraph(os.Args[2])
	if err != nil {
		panic(err)
	}
	var rect common.Rectangle
	if os.Args[1] == "chicago" {
		rect = common.Rectangle{
			common.Point{-4096, -8192},
			common.Point{4096, 0},
		}
	} else if os.Args[1] == "la" || os.Args[1] == "ny" || os.Args[1] == "toronto" || os.Args[1] == "amsterdam" || os.Args[1] == "denver" || os.Args[1] == "kansascity" || os.Args[1] == "montreal" || os.Args[1] == "paris" || os.Args[1] == "pittsburgh" || os.Args[1] == "saltlakecity" || os.Args[1] == "tokyo" || os.Args[1] == "vancouver" || os.Args[1] == "doha" || os.Args[1] == "sandiego" || os.Args[1] == "denver" || os.Args[1] == "atlanta" {
		rect = common.Rectangle{
			common.Point{-4096, -4096},
			common.Point{4096, 4096},
		}
	} else if os.Args[1] == "boston" {
		rect = common.Rectangle{
			common.Point{4096, -4096},
			common.Point{12288, 4096},
		}
	} else {
		fmt.Printf("unknown type %s\n", os.Args[1])
	}
	boundables := []common.Boundable{common.EmbeddedImage{
		Src: rect.Min,
		Dst: rect.Max,
		Image: fmt.Sprintf("./%s.png", os.Args[1]),
	}}
	boundables = append(boundables, common.ColoredBoundable{graph, "yellow"})

	outname := "out.svg"
	if len(os.Args) >= 4 {
		outname = os.Args[3]
	}

	if err := common.CreateSVG(outname, [][]common.Boundable{boundables}, common.SVGOptions{StrokeWidth: 2.0, Zoom: 2, Bounds: rect, Unflip: true}); err != nil {
		panic(err)
	}
}
