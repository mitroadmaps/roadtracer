package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/googlemaps"

	"fmt"
	"os"
)

const ZOOM = 18

func main() {
	var region lib.Region
	for _, r := range lib.GetRegions() {
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
			node.Point = googlemaps.PixelToLonLat(node.Point, region.CenterGPS, ZOOM)
		} else if mode == "topix" {
			node.Point = googlemaps.LonLatToPixel(node.Point, region.CenterGPS, ZOOM)
		} else {
			fmt.Errorf("bad mode %s, must be frompix or topix\n", mode)
		}
	}

	if err := graph.Write(os.Args[4]); err != nil {
		panic(err)
	}
}
