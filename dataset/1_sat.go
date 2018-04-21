package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/googlemaps"

	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
)

const ZOOM = 18

func main() {
	apiKey := os.Args[1]
	outDir := os.Args[2]

	widthWorld := 2 * math.Pi * 6378137 / math.Exp2(ZOOM) / 256 // meters per pixel
	regions := lib.GetRegions()

	type Tile struct {
		Region lib.Region
		X int
		Y int
		Filename string
	}

	var requiredTiles []Tile
	for _, region := range regions {
		for x := -region.RadiusX; x < region.RadiusX; x++ {
			for y := -region.RadiusY; y < region.RadiusY; y++ {
				fname := fmt.Sprintf("%s/%s_%d_%d_sat.png", outDir, region.Name, x, y)
				if _, err := os.Stat(fname); err == nil {
					continue
				}
				requiredTiles = append(requiredTiles, Tile{
					Region: region,
					X: x,
					Y: y,
					Filename: fname,
				})
			}
		}
	}

	fmt.Printf("found %d required tiles\n", len(requiredTiles))

	for _, tile := range requiredTiles {
		fmt.Printf("creating %s\n", filepath.Base(tile.Filename))
		im := image.NewNRGBA(image.Rect(0, 0, 4096, 4096))
		for xOffset := 0; xOffset < 4096; xOffset += 512 {
			for yOffset := 0; yOffset < 4096; yOffset += 512 {
				centerWorld := tile.Region.CenterWorld.Add(common.Point{float64(tile.X * 4096 + xOffset), float64(-(tile.Y * 4096 + yOffset))}.Scale(widthWorld))
				centerGPS := googlemaps.MetersToLonLat(centerWorld)
				satelliteImage := googlemaps.GetSatelliteImage(centerGPS, ZOOM, apiKey)
				for i := 0; i < 512; i++ {
					for j := 0; j < 512; j++ {
						im.Set(xOffset + i, yOffset + j, satelliteImage.At(i, j))
					}
				}
			}
		}
		f, err := os.Create(tile.Filename)
		if err != nil {
			panic(err)
		}
		if err := png.Encode(f, im); err != nil {
			panic(err)
		}
		f.Close()
	}
}
