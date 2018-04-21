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
	inDir := os.Args[1]
	outDir := os.Args[2]

	regions := lib.GetRegions()

	for _, region := range regions {
		graph, err := common.ReadGraph(fmt.Sprintf("%s/%s.graph", inDir, region.Name))
		if err != nil {
			panic(err)
		}
		for _, node := range graph.Nodes {
			node.Point = googlemaps.LonLatToPixel(node.Point, region.CenterGPS, ZOOM)
		}

		bytes, err := ioutil.ReadFile(fmt.Sprintf("%s/%s.tunnel.json", inDir, region.Name))
		if err != nil {
			panic(err)
		}
		tunnelEdges := make(map[int]bool)
		if err := json.Unmarshal(bytes, &tunnelEdges); err != nil {
			panic(err)
		}
		graph = filter(graph, tunnelEdges)

		if err := graph.Write(fmt.Sprintf("%s/%s.graph", outDir, region.Name)); err != nil {
			panic(err)
		}
	}
}

func filter(graph *common.Graph, tunnelEdges map[int]bool) *common.Graph {
	seenEdges := make(map[int]bool)
	badEdges := make(map[int]bool)
	for edgeID := range tunnelEdges {
		if seenEdges[edgeID] {
			continue
		}
		edge := graph.Edges[edgeID]
		seenEdges[edge.ID] = true
		componentEdges := []*common.Edge{edge}
		edgeQueue := []*common.Edge{edge}
		var componentLength float64 = 0
		for len(edgeQueue) > 0 {
			edge := edgeQueue[len(edgeQueue) - 1]
			edgeQueue = edgeQueue[:len(edgeQueue) - 1]
			var neighborEdges []*common.Edge
			neighborEdges = append(neighborEdges, edge.Src.In...)
			neighborEdges = append(neighborEdges, edge.Src.Out...)
			neighborEdges = append(neighborEdges, edge.Dst.In...)
			neighborEdges = append(neighborEdges, edge.Dst.Out...)
			for _, other := range neighborEdges {
				if seenEdges[other.ID] || !tunnelEdges[other.ID] {
					continue
				}
				seenEdges[other.ID] = true
				componentEdges = append(componentEdges, other)
				edgeQueue = append(edgeQueue, other)
				componentLength += other.Segment().Length()
			}
		}
		if componentLength > 200 {
			for _, edge := range componentEdges {
				badEdges[edge.ID] = true
			}
		}
	}
	return graph.FilterEdges(badEdges)
}
