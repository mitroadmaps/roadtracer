package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/googlemaps"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

const ZOOM = 18

func main() {
	pbfFname := os.Args[1]
	outDir := os.Args[2]

	regions := lib.GetRegions()

	boundList := make([]common.Rectangle, len(regions))
	for i, region := range regions {
		radius := common.Point{float64(region.RadiusX+2), float64(region.RadiusY+2)}.Scale(4096)
		extreme1 := googlemaps.PixelToLonLat(common.Point{-256, -256}.Sub(radius), region.CenterGPS, ZOOM)
		extreme2 := googlemaps.PixelToLonLat(common.Point{256, 256}.Add(radius), region.CenterGPS, ZOOM)
		r := extreme1.Bounds().Extend(extreme2)
		boundList[i] = r
		fmt.Printf("%s: %v\n", region.Name, r)
	}
	motorwayEdges := make([]map[int]bool, len(boundList))
	tunnelEdges := make([]map[int]bool, len(boundList))
	for i := range motorwayEdges {
		motorwayEdges[i] = make(map[int]bool)
		tunnelEdges[i] = make(map[int]bool)
	}
	graphs, err := common.LoadOSMMultiple(pbfFname, boundList, common.OSMOptions{
		Verbose: true,
		NoParking: true,
		MotorwayEdges: motorwayEdges,
		TunnelEdges: tunnelEdges,
	})
	if err != nil {
		panic(err)
	}
	for i, graph := range graphs {
		if err := graph.Write(fmt.Sprintf("%s/%s.graph", outDir, regions[i].Name)); err != nil {
			panic(err)
		}
		func() {
			bytes, err := json.Marshal(motorwayEdges[i])
			if err != nil {
				panic(err)
			}
			if err := ioutil.WriteFile(fmt.Sprintf("%s/%s.motorway.json", outDir, regions[i].Name), bytes, 0644); err != nil {
				panic(err)
			}
		}()
		func() {
			bytes, err := json.Marshal(tunnelEdges[i])
			if err != nil {
				panic(err)
			}
			if err := ioutil.WriteFile(fmt.Sprintf("%s/%s.tunnel.json", outDir, regions[i].Name), bytes, 0644); err != nil {
				panic(err)
			}
		}()
	}
}
