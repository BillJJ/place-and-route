# Problem
We want to place and route an input neural network to run on a compute tile
architecture. There are restrictions on how routes may be assigned so first
order of solution is coming up with a legal functional solution. Once we
have a legal solution the optimization objective is maximizing the result
for throughput.

Input:
 - A directed acyclic graph with at least 1 input node (0 inputs) and 1 output node (0 outputs)
   - each node in the graph represents 1 program
 - A fixed size rectangular device of shape, width x height
   - each grid cell represents 1 processor
Compute Model:
 - Each node represents a fixed computation with $n_i$ inputs and 1 distinct output
   - $n_i$ is between [0, 4]
 - Each node executes it's computation only once it has received all n_i inputs
 - Each node has a fixed processing time of $t_n$ cycles for the n'th node
 - Each input and output represents a single atomic packet or chunk that may be
   communicated in negligible time compared to the processing time.
 - Each output though singular may be sent to multiple downstream receivers (fan-out)
   - though represented as edges in the graph, this is better conceptualized as a "net"
     or 1 to many connection.
   - the fan-out number is between [0, 40]


Routing Restrictions:
 - All routes must fit on the device
 - Communication is only neighbor to neighbor
   - for purposes of this problem, assume each receiver has 1 async buffer slot they
     may use. So there is no back-pressure for the first packet but there is for the
     second one.
 - Routes are expressed as a sequence of neighbor to neighbor communications
 - If a route enters-exits a processor, this is accomplished by a "route node"
   - This is a new program that will run on the same processor as the node
     already on the processor.
 - A processor can support at most 1 node + 1 route node or 2 route nodes
   - The total processing time for each both nodes becomes the sum of the 2
     - e.g. A node with t_n = 5 is on a processor with a route node with t_r = 2; then
       the node will have t_n = 7 and the route node will have t_r = 7
     - This is because the total period of the processor to send all the packets for
       one iteration through the graph requires it to time-slice both processes
 - A route node can send to both nodes: if a node is sending to multiple nodes, then it can send to both the route node and compute node in the processor (store & forward)
 - Each neighbor to neighbor boundary can support only 1 route in each direction
   - So if another route wants to go in the same direction as another, it can't
     it must follow a different path.
   - This means when doing placement, you must account for the routes and what processors
     they will need.
 - Inputs and outputs must be on the edge of the device
   - communication with the outside is world is through the device boundary

Metrics:
 - Can we find legal routes for all the connections?
   - count the number of routes in each direction between neighbors and the number
     of route nodes and nodes on the different processors.
 - Throughput of the solution, this will correspond to the slowest t_n in the network
   after routes have been applied.
   - If you can avoid putting routes on the slowest nodes in a network then you may
     could in theory arrive at a perfect solution.
 - wire length (count of route nodes, half-perimeter wire-length ?)
 

Assumptions for Initial Problem:
 - Assume t_n is within a bounded range, [10, 20]
   - can use uniform random distribution to begin with.
 - Assume t_r for all route nodes is a fixed constant, 10
 - Assume a small device, 10x10

Guidance:
 - Research heuristic optimization approaches and place and route methods to get ideas
   - For this problem, you likely can't separate placement and routing as is typically done
     for other problems. Instead you want to look for a way to solve placement and routing
     together.
 - Use visualization tools to guide your work and figure out what is going on

Extensions:
 - Larger device and networks
   - go up to 25x25 device
 - Different shapes of device, thinner/wider rectangles
   - take a notch out of the device
-  Different weightings of routing time and node compute time, can make easier or harder to route.
- Different network topologys
   - try a scale-free network
     - This means that the number of nodes with a given degree k is proportional to $k^{-α}$, where α is a constant.
   - graphs with cycles
 - Calculate the latency for a given input, how long does it take to traverse the device


 TODO:
 - ~~generate an appropriate input~~
 - generate placement and routing:
    - 