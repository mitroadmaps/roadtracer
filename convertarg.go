package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"

	"fmt"
	"math"
	"os"
)

const ZOOM = 18

func main() {
	widthWorld := 2 * math.Pi * 6378137 / math.Exp2(ZOOM) / 256 // meters per pixel

	var region lib.Region
	for _, r := range lib.GetRegions(widthWorld) {
		if r.Name == os.Args[1] {
			region = r
			break
		}
	}
	mode := os.Args[2]
	graph, err := common.ReadGraph(os.Args[3])
	if err != nil {
		panic(err)
	}

	for _, node := range graph.Nodes {
		if mode == "frompix" {
			node.Point = lib.PixelToLonLat(node.Point, region.CenterWorld, widthWorld)
		} else if mode == "topix" {
			node.Point = lib.LonLatToPixel(node.Point, region.CenterWorld, widthWorld)
		} else {
			fmt.Errorf("bad mode %s, must be frompix or topix\n", mode)
		}
	}

	if err := graph.Write(os.Args[4]); err != nil {
		panic(err)
	}
}
