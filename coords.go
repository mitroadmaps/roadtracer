package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"

	"fmt"
	"math"
	"os"
	"strconv"
)

const ZOOM = 18

func main() {
	widthWorld := 2 * math.Pi * 6378137 / math.Exp2(ZOOM) / 256 // meters per pixel
	rname := os.Args[1]
	mode := os.Args[2]
	x, _ := strconv.ParseFloat(os.Args[3], 64)
	y, _ := strconv.ParseFloat(os.Args[4], 64)
	var region lib.Region
	for _, r := range lib.GetRegions(widthWorld) {
		if r.Name == rname {
			region = r
		}
	}
	p := common.Point{x, y}
	if mode == "frompix" {
		fmt.Println(lib.PixelToLonLat(p, region.CenterWorld, widthWorld))
	} else if mode == "topix" {
		fmt.Println(lib.LonLatToPixel(p, region.CenterWorld, widthWorld))
	}
}
