package main

import (
	"./lib"

	"fmt"
	"math"
)

const ZOOM = 18

func main() {
	widthWorld := 2 * math.Pi * 6378137 / math.Exp2(ZOOM) / 256 // meters per pixel
	for _, region := range lib.GetRegions(widthWorld) {
		fmt.Println(region)
	}
}
