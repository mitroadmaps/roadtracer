package main

import (
	"./lib"
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/googlemaps"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
)

const ZOOM = 18

func main() {
	graphDir := os.Args[1]

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

	type Location struct {
		X int `json:"x"`
		Y int `json:"y"`
		EdgeID int `json:"edge_id"`
	}

	selectRandomLocations := func(edges []*common.Edge, count int) []Location {
		var totalLength float64
		for _, edge := range edges {
			totalLength += edge.Segment().Length()
		}
		if totalLength == 0 {
			return nil
		}
		locations := make([]Location, count)
		for i := range locations {
			randLength := rand.Float64() * totalLength
			var selectedEdge *common.Edge
			for _, edge := range edges {
				randLength -= edge.Segment().Length()
				if randLength <= 0 {
					selectedEdge = edge
					break
				}
			}
			edgePos := common.EdgePos{
				Edge: selectedEdge,
				Position: rand.Float64() * selectedEdge.Segment().Length(),
			}
			point := edgePos.Point()
			locations[i] = Location{
				X: int(point.X),
				Y: int(point.Y),
				EdgeID: selectedEdge.ID,
			}
		}
		return locations
	}

	type PyTile struct {
		Region string `json:"region"`
		X int `json:"x"`
		Y int `json:"y"`
	}

	allStartingLocations := make(map[string][]Location)
	var pytiles []PyTile

	processTile := func(region lib.Region, x int, y int, rect common.Rectangle) {
		pytile := PyTile{
			Region: region.Name,
			X: x,
			Y: y,
		}

		edges := rtrees[region.Name].Search(rect.AddTol(-256))
		startingLocations := selectRandomLocations(edges, 512)
		if len(startingLocations) == 0 {
			fmt.Printf("skipping tile %s_%d_%d since no edges (%v)\n", region.Name, x, y, googlemaps.PixelToLonLat(rect.Min, region.CenterGPS, ZOOM))
			return
		}

		k := fmt.Sprintf("%s_%d_%d", region.Name, x, y)
		allStartingLocations[k] = startingLocations
		pytiles = append(pytiles, pytile)
	}

	fmt.Println("computing starting locations")
	for _, region := range regions {
		for x := -region.RadiusX; x < region.RadiusX; x++ {
			for y := -region.RadiusY; y < region.RadiusY; y++ {
				rect := common.Rectangle{
					common.Point{float64(x), float64(y)}.Scale(4096),
					common.Point{float64(x) + 1, float64(y) + 1}.Scale(4096),
				}
				processTile(region, x, y, rect)
			}
		}
	}

	fmt.Println("writing locations as json")
	func() {
		bytes, err := json.Marshal(allStartingLocations)
		if err != nil {
			panic(err)
		}
		if err := ioutil.WriteFile("starting_locations.json", bytes, 0644); err != nil {
			panic(err)
		}
	}()
	func() {
		bytes, err := json.Marshal(pytiles)
		if err != nil {
			panic(err)
		}
		if err := ioutil.WriteFile("pytiles.json", bytes, 0644); err != nil {
			panic(err)
		}
	}()
}
