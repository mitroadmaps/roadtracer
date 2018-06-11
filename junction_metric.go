package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"fmt"
	"io/ioutil"
	"math"
	"os"
)

const DEFAULT_INTERSECTION_CLUSTER_RADIUS = 50
const DEFAULT_INTERSECTION_ANGLE_DISTANCE = 65
const DEFAULT_INTERSECTION_MATCH_RADIUS = 65
const DEFAULT_INTERSECTION_SCORE_THRESHOLD = 20

type CompareIntersectionsParams struct {
	Scale float64
	ClusterRadius float64
	AngleDistance float64
	MatchRadius float64
	ScoreThreshold float64
}

func (params CompareIntersectionsParams) GetScale() float64 {
	if params.Scale == 0 {
		return 1
	} else {
		return params.Scale
	}
}

func (params CompareIntersectionsParams) GetClusterRadius() float64 {
	if params.ClusterRadius == 0 {
		return DEFAULT_INTERSECTION_CLUSTER_RADIUS * params.GetScale()
	} else {
		return params.ClusterRadius * params.GetScale()
	}
}

func (params CompareIntersectionsParams) GetAngleDistance() float64 {
	if params.AngleDistance == 0 {
		return DEFAULT_INTERSECTION_ANGLE_DISTANCE * params.GetScale()
	} else {
		return params.AngleDistance * params.GetScale()
	}
}

func (params CompareIntersectionsParams) GetMatchRadius() float64 {
	if params.MatchRadius == 0 {
		return DEFAULT_INTERSECTION_MATCH_RADIUS * params.GetScale()
	} else {
		return params.MatchRadius * params.GetScale()
	}
}

func (params CompareIntersectionsParams) GetScoreThreshold() float64 {
	if params.ScoreThreshold == 0 {
		return DEFAULT_INTERSECTION_SCORE_THRESHOLD * params.GetScale()
	} else {
		return params.ScoreThreshold * params.GetScale()
	}
}

type IntersectionCluster struct {
	Point common.Point
	Nodes []*common.Node
	InAngles []float64
	OutAngles []float64
}

func (cluster IntersectionCluster) Bounds() common.Rectangle {
	return cluster.Point.Bounds()
}

func (cluster IntersectionCluster) Unwrap() common.Boundable {
	return cluster.Point
}

func isIntersection(node *common.Node) bool {
	neighborSet := make(map[int]bool)
	for _, edge := range node.In {
		neighborSet[edge.Src.ID] = true
	}
	for _, edge := range node.Out {
		neighborSet[edge.Dst.ID] = true
	}
	return len(neighborSet) >= 3
}

func GetIntersectionClusters(graph *common.Graph, params CompareIntersectionsParams) []IntersectionCluster {
	// create another version of the graph where all edges are bidirectional
	// the node IDs will match between these two versions
	undirectedGraph := graph.Clone()
	undirectedGraph.MakeBidirectional()

	seenNodes := make(map[int]bool)
	var clusters []IntersectionCluster

	for _, node := range graph.Nodes {
		if seenNodes[node.ID] {
			continue
		} else if !isIntersection(node) {
			continue
		}

		cluster := IntersectionCluster{Point: node.Point}
		intersectionNodes := make(map[int]bool)

		// find all nodes that should go in this cluster
		followNodes := make(map[int]bool)
		undirectedGraph.Follow(common.FollowParams{
			SourceNodes: []*common.Node{undirectedGraph.Nodes[node.ID]},
			Distance: params.GetClusterRadius() * 2,
			SeenNodes: followNodes,
		})
		for candidateID := range followNodes {
			candidate := graph.Nodes[candidateID]
			if node.Point.Distance(candidate.Point) < params.GetClusterRadius() && isIntersection(candidate) {
				intersectionNodes[candidate.ID] = true
				seenNodes[candidate.ID] = true
				cluster.Nodes = append(cluster.Nodes, candidate)
			}
		}

		// follow the graph again, and then extract angles from positions
		// we first set up stop nodes, which should be any intersections that are not in this cluster
		// (this way, positions do not branch in case there is another intersection close by)
		stopNodes := make(map[int]bool)
		for _, other := range graph.Nodes {
			if !intersectionNodes[other.ID] && isIntersection(other) {
				stopNodes[other.ID] = true
			}
		}
		forwardPositions := graph.Follow(common.FollowParams{
			SourceNodes: cluster.Nodes,
			Distance: params.GetAngleDistance(),
			StopNodes: stopNodes,
		})
		backwardPositions := graph.Follow(common.FollowParams{
			SourceNodes: cluster.Nodes,
			Distance: params.GetAngleDistance(),
			StopNodes: stopNodes,
			NoForwards: true,
			Backwards: true,
		})
		angleFromPosition := func(pos common.EdgePos) float64 {
			point := pos.Point()
			vector := point.Sub(cluster.Point)
			return common.Point{1, 0}.SignedAngle(vector)
		}
		for _, pos := range forwardPositions {
			cluster.OutAngles = append(cluster.OutAngles, angleFromPosition(pos))
		}
		for _, pos := range backwardPositions {
			cluster.InAngles = append(cluster.InAngles, angleFromPosition(pos))
		}

		clusters = append(clusters, cluster)
	}

	return clusters
}

func GetIntersectionClustersWithContext(graph *common.Graph, params CompareIntersectionsParams, context []IntersectionCluster) []IntersectionCluster {
	// create another version of the graph where all edges are bidirectional
	// the node IDs will match between these two versions
	undirectedGraph := graph.Clone()
	undirectedGraph.MakeBidirectional()

	// for each intersection in graph, match it to the closest intersection in context
	matchedIntersections := make(map[int][]*common.Node)
	for _, node := range graph.Nodes {
		if !isIntersection(node) {
			continue
		}

		var bestIdx int = -1
		var bestDistance float64
		for idx, cluster := range context {
			distance := cluster.Point.Distance(node.Point)
			if bestIdx == -1 || distance < bestDistance {
				bestIdx = idx
				bestDistance = distance
			}
		}

		matchedIntersections[bestIdx] = append(matchedIntersections[bestIdx], node)
	}

	// for each context intersection with at least one match, pick the closest match and
	//  create a cluster from it
	// other graph intersections that matched to the same context intersection are included
	//  in the cluster only if they are within a certain distance to it on the network;
	//  otherwise, we need to put them in a separate cluster
	var clusters []IntersectionCluster
	for contextIdx, nodes := range matchedIntersections {
		var bestNode *common.Node
		var bestDistance float64
		for _, node := range nodes {
			distance := node.Point.Distance(context[contextIdx].Point)
			if bestNode == nil || distance < bestDistance {
				bestNode = node
				bestDistance = distance
			}
		}

		cluster := IntersectionCluster{Point: bestNode.Point}

		potentialNodes := make(map[int]bool)
		for _, node := range nodes {
			potentialNodes[node.ID] = true
		}

		followNodes := make(map[int]bool)
		undirectedGraph.Follow(common.FollowParams{
			SourceNodes: []*common.Node{undirectedGraph.Nodes[bestNode.ID]},
			Distance: params.GetClusterRadius() * 3,
			SeenNodes: followNodes,
		})
		for candidateID := range followNodes {
			candidate := graph.Nodes[candidateID]
			if potentialNodes[candidate.ID] && isIntersection(candidate) {
				cluster.Nodes = append(cluster.Nodes, candidate)
			}
		}
		clusters = append(clusters, cluster)

		for nodeID := range potentialNodes {
			if !followNodes[nodeID] {
				node := graph.Nodes[nodeID]
				clusters = append(clusters, IntersectionCluster{
					Point: node.Point,
					Nodes: []*common.Node{node},
				})
			}
		}
	}

	// determine angles for each cluster
	for i, cluster := range clusters {
		// follow the graph and extract angles from positions
		// we first set up stop nodes, which should be any intersections that are not in this cluster
		// (this way, positions do not branch in case there is another intersection close by)
		stopNodes := make(map[int]bool)
		for _, other := range graph.Nodes {
			if isIntersection(other) {
				stopNodes[other.ID] = true
			}
		}
		for _, clusterNode := range cluster.Nodes {
			stopNodes[clusterNode.ID] = false
		}

		forwardPositions := graph.Follow(common.FollowParams{
			SourceNodes: cluster.Nodes,
			Distance: params.GetAngleDistance(),
			StopNodes: stopNodes,
		})
		backwardPositions := graph.Follow(common.FollowParams{
			SourceNodes: cluster.Nodes,
			Distance: params.GetAngleDistance(),
			StopNodes: stopNodes,
			NoForwards: true,
			Backwards: true,
		})
		angleFromPosition := func(pos common.EdgePos) float64 {
			point := pos.Point()
			vector := point.Sub(cluster.Point)
			return common.Point{1, 0}.SignedAngle(vector)
		}
		for _, pos := range forwardPositions {
			clusters[i].OutAngles = append(clusters[i].OutAngles, angleFromPosition(pos))
		}
		for _, pos := range backwardPositions {
			clusters[i].InAngles = append(clusters[i].InAngles, angleFromPosition(pos))
		}
	}

	return clusters
}

func angleDifference(a float64, b float64) float64 {
	d := math.Abs(a - b)
	if d > math.Pi {
		d = 2 * math.Pi - d
	}
	return d
}

func isMatch(a IntersectionCluster, b IntersectionCluster) (float64, float64) {
	if len(a.OutAngles) == 0 || len(b.OutAngles) == 0 {
		return 0, 1
	}
	var aBad, bBad int
	for _, angle := range a.OutAngles {
		isMatched := false
		for _, other := range b.OutAngles {
			if angleDifference(angle, other) < math.Pi / 2 {
				isMatched = true
				break
			}
		}
		if !isMatched {
			aBad++
		}
	}
	for _, angle := range b.OutAngles {
		isMatched := false
		for _, other := range a.OutAngles {
			if angleDifference(angle, other) < math.Pi / 2 {
				isMatched = true
				break
			}
		}
		if !isMatched {
			bBad++
		}
	}
	return 1.0 - float64(aBad) / float64(len(a.OutAngles)), float64(bBad) / float64(len(b.OutAngles))
}

func CompareIntersectionClusters(truth []IntersectionCluster, inferred []IntersectionCluster, params CompareIntersectionsParams, bounds common.Rectangle) (total int, correct [][2]IntersectionCluster, wrong [][2]IntersectionCluster, extra []IntersectionCluster, missed []IntersectionCluster, correctScore float64, extraScore float64) {
	usedClusters := make(map[int]bool)

	// loop over truth clusters
	// for each cluster, identify candidate inferred clusters
	// if there are multiple candidates, pick the closest one
	for _, cluster := range truth {
		if !bounds.Contains(cluster.Point) {
			continue
		}
		var bestClusterIdx int = -1
		var bestCorrectScore float64
		var bestExtraScore float64
		var bestFscore float64
		for idx, other := range inferred {
			if other.Point.Distance(cluster.Point) > params.GetMatchRadius() {
				continue
			} else if usedClusters[idx] {
				continue
			}
			correctScore, extraScore := isMatch(cluster, other)
			score := correctScore - extraScore
			if score <= 0 {
				continue
			}
			fscore := score - other.Point.Distance(cluster.Point) / params.GetMatchRadius()
			if bestClusterIdx == -1 || fscore > bestFscore {
				bestClusterIdx = idx
				bestCorrectScore = correctScore
				bestExtraScore = extraScore
				bestFscore = fscore
			}
		}

		if bestClusterIdx == -1 {
			missed = append(missed, cluster)
			total++
			continue
		}

		usedClusters[bestClusterIdx] = true
		correct = append(correct, [2]IntersectionCluster{cluster, inferred[bestClusterIdx]})
		correctScore += bestCorrectScore
		extraScore += bestExtraScore
		total++
	}

	// mark unmatched inferred clusters as extra
	for idx, cluster := range inferred {
		if !usedClusters[idx] && bounds.Contains(cluster.Point) {
			extra = append(extra, cluster)
			extraScore += 1
		}
	}

	return
}

func CompareAndVisualize(aClusters []IntersectionCluster, bClusters []IntersectionCluster, params CompareIntersectionsParams, bounds common.Rectangle) ([]common.Boundable, []common.Boundable) {
	fmt.Printf("%d clusters in a\n", len(aClusters))
	fmt.Printf("%d clusters in b\n", len(bClusters))
	total, correct, wrong, extra, missed, correctScore, extraScore := CompareIntersectionClusters(aClusters, bClusters, params, bounds)
	fmt.Printf("%d total, %d correct, %d wrong, %d extra, %d missed ;;; %.0f correct, %.0f extra\n", total, len(correct), len(wrong), len(extra), len(missed), correctScore, extraScore)

	var aBoundables []common.Boundable
	for _, c := range correct {
		aBoundables = append(aBoundables, common.ColoredBoundable{common.WidthBoundable{c[0], 8}, "green"})
	}
	for _, c := range missed {
		aBoundables = append(aBoundables, common.ColoredBoundable{common.WidthBoundable{c, 8}, "red"})
	}

	var bBoundables []common.Boundable
	for _, c := range correct {
		bBoundables = append(bBoundables, common.ColoredBoundable{common.WidthBoundable{c[1], 8}, "green"})
	}
	for _, c := range wrong {
		bBoundables = append(bBoundables, common.ColoredBoundable{common.WidthBoundable{c[1], 8}, "yellow"})
	}
	for _, c := range extra {
		bBoundables = append(bBoundables, common.ColoredBoundable{common.WidthBoundable{c, 8}, "red"})
	}

	return aBoundables, bBoundables
}

func WriteClusters(clusters []IntersectionCluster, fname string) {
	s := ""
	for _, cluster := range clusters {
		s += fmt.Sprintf("%f %f [", cluster.Point.X, cluster.Point.Y)
		for _, angle := range cluster.InAngles {
			s += fmt.Sprintf("%f,", angle)
		}
		s += "] ["
		for _, angle := range cluster.OutAngles {
			s += fmt.Sprintf("%f,", angle)
		}
		s += "\n"
	}
	if err := ioutil.WriteFile(fname, []byte(s), 0755); err != nil {
		panic(err)
	}
}

func main() {
	truth, err := common.ReadGraph(os.Args[1])
	if err != nil {
		panic(err)
	}
	inferred, err := common.ReadGraph(os.Args[2])
	if err != nil {
		panic(err)
	}
	r4096 := common.Rectangle{
		common.Point{-4096, -4096},
		common.Point{4096, 4096},
	}
	rects := map[string]common.Rectangle{
		"toronto": r4096,
		"la": r4096,
		"ny": r4096,
		"boston": common.Rectangle{
			common.Point{4096, -4096},
			common.Point{12288, 4096},
		},
		"chicago": common.Rectangle{
			common.Point{-4096, -8192},
			common.Point{4096, 0},
		},
		"amsterdam": r4096,
		"denver": r4096,
		"kansascity": r4096,
		"montreal": r4096,
		"paris": r4096,
		"pittsburgh": r4096,
		"saltlakecity": r4096,
		"sandiego": r4096,
		"tokyo": r4096,
		"vancouver": r4096,
	}
	regionRect := rects[os.Args[3]]
	rect := regionRect.AddTol(-300)

	truth = truth.GetSubgraphInRect(regionRect)
	inferred = inferred.GetSubgraphInRect(regionRect)
	params := CompareIntersectionsParams{Scale: 1.7}
	aClusters := GetIntersectionClusters(truth, params)
	bClusters := GetIntersectionClustersWithContext(inferred, params, aClusters)
	aBoundables, bBoundables := CompareAndVisualize(aClusters, bClusters, params, rect)
	im := common.EmbeddedImage{
		Src: regionRect.Min,
		Dst: regionRect.Max,
		Image: fmt.Sprintf("./14-%s.png", os.Args[3]),
	}
	if err := common.CreateSVG("truth.svg", [][]common.Boundable{[]common.Boundable{im, common.ColoredBoundable{truth, "blue"}}, aBoundables}, common.SVGOptions{StrokeWidth: 2.0, Zoom: 2.0, Bounds: rect, Unflip: true}); err != nil {
		panic(err)
	}
	if err := common.CreateSVG("inferred.svg", [][]common.Boundable{[]common.Boundable{im, common.ColoredBoundable{inferred, "blue"}}, bBoundables}, common.SVGOptions{StrokeWidth: 2.0, Zoom: 2.0, Bounds: rect, Unflip: true}); err != nil {
		panic(err)
	}
	WriteClusters(aClusters, "truth_clusters.txt")
	WriteClusters(bClusters, "inferred_clusters.txt")
}
