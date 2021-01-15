// #include /graph.kojo

cleari()
val cb = canvasBounds
val n = 10
val stepx = cb.width / (n + 1)
val stepy = cb.height / (n + 1)

case class GridPos(x: Int, y: Int)
type GraphNode = Node[GridPos]
case class GridCell(pos: GridPos, pic: Picture, var node: Option[GraphNode])

val gridPoints = Array.ofDim[GridCell](n, n)

val emptyCellColor = lightGray
val nodeCellColor = cm.blueViolet
val edgeFirstCellColor = red
val visitedColor = green

var edgeFirstCell: Option[GridCell] = None

val nodes = HashSet.empty[GraphNode]
val edges = HashMap.empty[GraphNode, MSet[EdgeTo[GridPos]]]
val edgeCells = HashSet.empty[(GridCell, GridCell)]
val edgePics = HashSet.empty[Picture]
val pathPics = HashSet.empty[Picture]
val visitedPics = HashSet.empty[Picture]

def eraseEdges() {
    edgePics.foreach { pic =>
        pic.erase()
    }
}

//def drawEdges() {
//    eraseEdges()
//    edgeCells.foreach {
//        case (cell1, cell2) =>
//            drawEdge(cell1, cell2)
//    }
//}

def distance(n1: GraphNode, n2: GraphNode) = {
    val p1 = n1.data; val p2 = n2.data
    math.sqrt(math.pow(p2.x - p1.x, 2) + math.pow(p2.y - p1.y, 2))
}

def addEdge(c1: GridCell, c2: GridCell) {
    val p1 = c1.pos; val p2 = c2.pos
    val dist = distance(c1.node.get, c2.node.get)
    edgeCells += ((c1, c2))
    val c1Adj = edges.getOrElseUpdate(c1.node.get, HashSet())
    c1Adj += EdgeTo(Node(c2.pos), dist)
    val c2Adj = edges.getOrElseUpdate(c2.node.get, HashSet())
    c2Adj += EdgeTo(Node(c1.pos), dist)
    val edge = makeAndDrawEdge(c1, c2)
    edgePics += edge
}

def highlightNode(n: GraphNode, c: Color, r: Int): Picture = {
    val pos = n.data
    val cell = gridPoints(pos.x)(pos.y)
    val pic = cell.pic
    val vpic = Picture.circle(r)
    vpic.setPosition(pic.position)
    vpic.setPenColor(c)
    vpic.setFillColor(c)
    vpic.moveToBack()
    draw(vpic)
    vpic
}

def visitCallback(n: GraphNode) {
    val vpic = highlightNode(n, visitedColor, 20)
    visitedPics += vpic
    pause(1)
}

def eraseVisits() {
    visitedPics.foreach { pic =>
        pic.erase()
    }
}

def erasePath() {
    pathPics.foreach { pic =>
        pic.erase()
    }
}

def drawPath(start: Node[GridPos], path: Option[PathEdges[GridPos]]) {
    erasePath()
    if (path.isEmpty) return

    def drawEdgeBetweenPos(gridPos1: GridPos, gridPos2: GridPos) {
        val cell1 = gridPoints(gridPos1.x)(gridPos1.y)
        val cell2 = gridPoints(gridPos2.x)(gridPos2.y)
        val edge = makeAndDrawEdge(cell1, cell2, yellow, 8, true)
        pathPics += edge
    }

    val firstSegment = path.get.apply(0)
    val gridPos1 = start.data
    val gridPos2 = firstSegment.node.data
    drawEdgeBetweenPos(gridPos1, gridPos2)

    path.get.sliding(2).foreach { posPair =>
        val gridPos1 = posPair(0).node.data
        val gridPos2 = posPair(1).node.data
        drawEdgeBetweenPos(gridPos1, gridPos2)
    }
}

def makeAndDrawEdge(cell1: GridCell, cell2: GridCell, color: Color = blue,
                    width: Int = 2, back: Boolean = true): Picture = {
    val pos1 = cell1.pic.position; val pos2 = cell2.pic.position
    val edge = trans(pos1.x, pos1.y) * penColor(color) * penWidth(width) ->
        Picture.line(pos2.x - pos1.x, pos2.y - pos1.y)
    if (back) {
        edge.moveToBack()
    }
    draw(edge)
    edge
}

repeatFor(0 to n - 1) { nx =>
    val x = cb.x + (nx + 1) * stepx
    repeatFor(0 to n - 1) { ny =>
        val y = cb.y + (ny + 1) * stepy
        val pos = GridPos(nx, ny)
        val pic = Picture.circle(10)
        pic.setPosition(x, y)
        pic.setPenColor(emptyCellColor)
        pic.setFillColor(emptyCellColor)
        draw(pic)
        val cell = GridCell(pos, pic, None)
        gridPoints(nx)(ny) = cell
        pic.onMouseClick { (x, y) =>
            if (isKeyPressed(Kc.VK_CONTROL)) {
                cell.node match {
                    case Some(edgeCell2Node) =>
                        edgeFirstCell match {
                            case Some(edgeCell1) =>
                                if (cell == edgeCell1) {
                                    cell.pic.setPenColor(nodeCellColor)
                                    cell.pic.setFillColor(nodeCellColor)
                                    edgeFirstCell = None
                                }
                                else {
                                    val node1 = edgeCell1.node.get
                                    val node2 = edgeCell2Node
                                    edgeCell1.pic.setPenColor(nodeCellColor)
                                    edgeCell1.pic.setFillColor(nodeCellColor)
                                    addEdge(edgeCell1, cell)
                                    edgeFirstCell = None
                                }
                            case None =>
                                edgeFirstCell = Some(cell)
                                pic.setPenColor(edgeFirstCellColor)
                                pic.setFillColor(edgeFirstCellColor)
                        }
                    case None =>
                }
            }
            else {
                cell.node match {
                    case Some(n) =>
                        if (!edges.contains(n)) {
                            cell.node = None
                            nodes -= n
                            pic.setPenColor(emptyCellColor)
                            pic.setFillColor(emptyCellColor)
                        }
                    case None =>
                        val newNode = new GraphNode(pos)
                        cell.node = Some(newNode)
                        nodes += newNode
                        pic.setPenColor(nodeCellColor)
                        pic.setFillColor(nodeCellColor)
                }
            }
        }
    }
}

def loadNodesAndEdges(gnodes: Nodes[GridPos], gedges: GraphEdges[GridPos]) {
    gnodes.foreach { node =>
        val pos = node.data
        val cell = gridPoints(pos.x)(pos.y)
        val pic = cell.pic
        cell.node = Some(node)
        nodes += node
        pic.setPenColor(nodeCellColor)
        pic.setFillColor(nodeCellColor)
    }
    gedges.foreach {
        case (node, adjSet) =>
            val pos = node.data
            val cell1 = gridPoints(pos.x)(pos.y)
            adjSet.foreach { edgeTo =>
                val pos2 = edgeTo.node.data
                val cell2 = gridPoints(pos2.x)(pos2.y)
                addEdge(cell1, cell2)
            }
    }
}

def eraseSearch() {
    eraseVisits()
    erasePath()
}

val gEdges: MMap[GraphNode, MSet[EdgeTo[GridPos]]] = HashMap(
    Node(GridPos(7, 1)) -> HashSet(EdgeTo(Node(GridPos(3, 2)), 4.123105625617661), EdgeTo(Node(GridPos(8, 6)), 5.0990195135927845)),
    Node(GridPos(0, 2)) -> HashSet(EdgeTo(Node(GridPos(0, 0)), 2.0), EdgeTo(Node(GridPos(1, 4)), 2.23606797749979)),
    Node(GridPos(3, 2)) -> HashSet(EdgeTo(Node(GridPos(5, 2)), 2.0), EdgeTo(Node(GridPos(2, 0)), 2.23606797749979), EdgeTo(Node(GridPos(5, 6)), 4.47213595499958), EdgeTo(Node(GridPos(7, 1)), 4.123105625617661)),
    Node(GridPos(8, 6)) -> HashSet(EdgeTo(Node(GridPos(5, 2)), 5.0), EdgeTo(Node(GridPos(5, 6)), 3.0), EdgeTo(Node(GridPos(7, 1)), 5.0990195135927845), EdgeTo(Node(GridPos(9, 9)), 3.1622776601683795)),
    Node(GridPos(6, 8)) -> HashSet(EdgeTo(Node(GridPos(5, 6)), 2.23606797749979), EdgeTo(Node(GridPos(9, 9)), 3.1622776601683795)),
    Node(GridPos(2, 0)) -> HashSet(EdgeTo(Node(GridPos(0, 0)), 2.0), EdgeTo(Node(GridPos(3, 2)), 2.23606797749979)),
    Node(GridPos(5, 6)) -> HashSet(EdgeTo(Node(GridPos(3, 2)), 4.47213595499958), EdgeTo(Node(GridPos(8, 6)), 3.0), EdgeTo(Node(GridPos(6, 8)), 2.23606797749979), EdgeTo(Node(GridPos(1, 4)), 4.47213595499958)),
    Node(GridPos(0, 0)) -> HashSet(EdgeTo(Node(GridPos(0, 2)), 2.0), EdgeTo(Node(GridPos(1, 4)), 4.123105625617661), EdgeTo(Node(GridPos(2, 0)), 2.0)),
    Node(GridPos(5, 2)) -> HashSet(EdgeTo(Node(GridPos(8, 6)), 5.0), EdgeTo(Node(GridPos(3, 2)), 2.0)),
    Node(GridPos(1, 4)) -> HashSet(EdgeTo(Node(GridPos(0, 0)), 4.123105625617661), EdgeTo(Node(GridPos(0, 2)), 2.23606797749979), EdgeTo(Node(GridPos(5, 6)), 4.47213595499958)),
    Node(GridPos(9, 9)) -> HashSet(EdgeTo(Node(GridPos(6, 8)), 3.1622776601683795), EdgeTo(Node(GridPos(8, 6)), 3.1622776601683795))
)

val gNodes = HashSet(
    Node(GridPos(5, 2)), Node(GridPos(5, 6)), Node(GridPos(6, 8)), Node(GridPos(0, 2)),
    Node(GridPos(7, 1)), Node(GridPos(8, 6)), Node(GridPos(9, 9)), Node(GridPos(1, 4)),
    Node(GridPos(0, 0)), Node(GridPos(2, 0)), Node(GridPos(3, 2))
)

loadNodesAndEdges(gNodes, gEdges)
val g = ExplicitGraph(nodes, edges)
val start = Node(GridPos(0, 0))
val end = Node(GridPos(9, 9))

highlightNode(start, red, 15)
highlightNode(end, red, 15)

def search(algo: String) {
    val path = algo match {
        case "dfs" => GraphSearch.dfs(g, start, end, visitCallback)
        case "bfs" => GraphSearch.bfs(g, start, end, visitCallback)
        case "ucs" => GraphSearch.ucs(g, start, end, visitCallback)
        case "astar" =>
            GraphSearch.astarSearch(g, start, end, visitCallback, distance)
    }
    drawPath(start, path)
}

def redoSearch(algo: String) {
    eraseSearch()
    search(algo)
}

search("astar")
