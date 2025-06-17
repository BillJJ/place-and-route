//********************************************************************
//  dag_placer_router.cpp  (rewrite)
//
//  – Reads a DAG in the “# Nodes / # Edges / # Dimensions” format
//  – Places every node in an R×C 4‑neighbour grid under these rules
//      • inputs  (in‑degree  == 0) are fixed to the top rows
//      • outputs (out‑degree == 0) are fixed to the bottom rows
//      • each tile:   1 compute  + 0–1 route  OR 0 compute + 0–2 routes
//  – Routes every edge with a simple BFS that inserts route‑nodes as
//    needed and enforces:
//      • each boundary is used by at most one directed flow
//      • flows never merge; each edge owns every boundary it crosses
//  – For visualisation we emit, per edge, the exact sequence of
//    segments, marking which logical node inside every tile emitted
//    the data (C=compute, R=route) and what was at the receiving side.
//
//  Build:
//      g++ -std=c++17 -O2 dag_placer_router.cpp -o place_route
//********************************************************************
#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;

/*------------------------------------------------- directions ------*/
enum class Dir
{
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3
};
static const int DR[4] = {-1, 0, 1, 0};
static const int DC[4] = {0, 1, 0, -1};
static inline char dirChr(Dir d)
{
    static const char t[4] = {'U', 'R', 'D', 'L'};
    return t[(int)d];
}

/*------------------------------------------------- node‑kind -------*/
enum class NodeKind
{
    Comp,
    Route
};
static inline char nkChr(NodeKind k) { return k == NodeKind::Comp ? 'C' : 'R'; }

/*------------------------------------------------- grid + tiles ----*/
struct Tile
{
    int comp = -1; // id of compute node (‑1 ⇒ none)
    int rCnt = 0;  // number of route nodes in this tile
    int r1_data = -1;
    int r2_data = -1;
    bool has_data(int compute_node_data) {
        return compute_node_data == r1_data || compute_node_data == r2_data;
    }
    void add_data(int compute_node_data) {
        if (r1_data == -1) r1_data = compute_node_data;
        else r2_data = compute_node_data;
    }
};
struct Grid
{
    int R = 0, C = 0;
    vector<Tile> cell; // row‑major
    Grid() = default;
    Grid(int r, int c) : R(r), C(c), cell(r * c) {}
    inline bool inside(int r, int c) const { return r >= 0 && r < R && c >= 0 && c < C; }
    inline Tile &at(int r, int c) { return cell[r * C + c]; }
    inline const Tile &at(int r, int c) const { return cell[r * C + c]; }
};

/* one‑way usage flags on every grid boundary*/
// Distinguish between up/down and left/right with negative / positive
// Going up is -1, down is 1, left = -1, right = 1,
// If 0, then it's free use
struct EdgeUse {
    vector<vector<bool>> used[4]; // used[direction][row][col] true / false
    EdgeUse(int R, int C) {
        for (int i = 0; i < 4; i++) used[i].assign(R, vector<bool> (C, 0));
    }
};

/*------------------------------------------------- input problem ---*/
struct Problem
{
    vector<int> pt;          // processing time per node id
    vector<vector<int>> adj; // out‑edges
    int R = 0, C = 0;
};

Problem readDag(const string &fn) {
    string path = "graphs/" + fn + "/" + fn + "_input.txt";
    ifstream fin(path);
    if (!fin.is_open())
        throw runtime_error("cannot open " + fn);
    enum Sect
    {
        NONE,
        NODES,
        EDGES,
        DIM
    };
    Sect s = NONE;
    string line;
    vector<pii> eb;
    Problem P;
    while (getline(fin, line))
    {
        if (line.empty() || line[0] == '#')
        {
            if (line.rfind("# Nodes", 0) == 0)
                s = NODES;
            else if (line.rfind("# Edges", 0) == 0)
                s = EDGES;
            else if (line.rfind("# Dimensions", 0) == 0)
                s = DIM;
            continue;
        }
        stringstream ss(line);
        if (s == NODES)
        {
            int id, pt;
            ss >> id >> pt;
            if ((int)P.pt.size() <= id)
                P.pt.resize(id + 1, 0);
            P.pt[id] = pt;
        }
        else if (s == EDGES)
        {
            int u, v;
            ss >> u >> v;
            eb.emplace_back(u, v);
            int mx = max(u, v);
            if ((int)P.pt.size() <= mx)
                P.pt.resize(mx + 1, 0);
        }
        else if (s == DIM)
        {
            ss >> P.R >> P.C;
        }
    }
    P.adj.assign(P.pt.size(), {});
    for (auto &e : eb)
        P.adj[e.first].push_back(e.second);
    return P;
}

/*------------------------------------------------- util helpers ----*/

static inline bool at_capacity(Grid &grid, int nr, int nc, bool dst) {
    // check if there are already two route nodes present
    // if there are, then we are at capacity and can't add more route nodes
    if (dst) return false;
    if (grid.at(nr, nc).comp != -1) { // if there is a compute node present
        return grid.at(nr, nc).rCnt == 1; // at capacity if non-zero route node count
    } else {
        return grid.at(nr, nc).rCnt == 2; // at capacity if 2 route nodes present
    }
}

static inline bool same_data_type(Grid &grid, const pii compute_src, const pii route_other) {
    // returns if compute node at compute_src is outputting the same type that flows through
    // a route node in other

    int data_type = grid.at(compute_src.first, compute_src.second).comp;
    return (grid.at(route_other.first, route_other.second)).has_data(data_type);
}

/*------------------------------------------------- edge path record -*/
struct Segment {                // one boundary‑crossing
    int r, c;    // origin tile coord
    Dir dir;     // direction data left origin tile
    NodeKind from, to; // logical node kinds (C/R) on origin & dest tiles
};

/*------------------------------------------------- BFS router ------*/
bool routeEdge(Grid &grid, EdgeUse &edge_use, const pii &src, const pii &dst, vector<Segment> &path) {
    const int R = grid.R, C = grid.C;
    struct Parent {
        int pr, pc;
        Dir dir;
    };
    vector<vector<char>> vis(R, vector<char>(C, 0));
    vector<vector<Parent>> par(R, vector<Parent>(C, {-1, -1, Dir::Up}));
    queue<pii> q;
    q.push(src);
    vis[src.first][src.second] = 1;
    auto edgeFree = [&](int r, int c, Dir d) { // check if an edge is free in that direction
        // note that new tile will always be inside the graph
        return edge_use.used[(int)d][r][c] == 0;
    };

    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        if (r == dst.first && c == dst.second)
            break;
        for (int k = 0; k < 4; ++k)
        {
            Dir d = (Dir)k;
            int nr = r + DR[k], nc = c + DC[k];
            if (!grid.inside(nr, nc) || vis[nr][nc])
                continue;
            if (!edgeFree(r, c, d))
                continue;
            bool atDst = (nr == dst.first && nc == dst.second);
            if (at_capacity(grid, nr, nc, atDst) && !same_data_type(grid, src, {nr, nc}))
                // all that needs checking here is that the tile isn't at capacity, cuz then we can squeeze in a route node
                // well that, and check that we don't already have the data flowing through there
                continue;
            vis[nr][nc] = 1;
            par[nr][nc] = {r, c, d};
            q.emplace(nr, nc);
        }
    }
    if (!vis[dst.first][dst.second])
        return false; // no route
    // reconstruct + reserve
    vector<Segment> tmp;
    int r = dst.first, c = dst.second;
    NodeKind dstKind = NodeKind::Comp; // data lands in compute node

    while (!(r == src.first && c == src.second)) {
        auto parent = par[r][c];
        int pr = parent.pr, pc = parent.pc;
        Dir d = parent.dir;

        // reserve edge boundary
        edge_use.used[(int)d][min(pr, r)][min(pc, c)] = 1;

        NodeKind fromKind;
        const Tile &Tsrc = grid.at(pr, pc);
        bool firstStep = (pr == src.first && pc == src.second);
        if (firstStep)
            fromKind = NodeKind::Comp;
        else
            fromKind = NodeKind::Route;
        tmp.push_back({pr, pc, d, fromKind, dstKind});

        // add in route nodes where appropriate
        // no need to add in route nodes to places where that data type is already present
        if (!firstStep && !same_data_type(grid, src, {pr, pc})) {
            grid.at(pr, pc).rCnt++; // only add in route nodes to intermediate places
            grid.at(pr, pc).add_data(grid.at(src.first, src.second).comp); // add in data type
        }


        // next iteration
        r = pr;
        c = pc;
        dstKind = NodeKind::Route; // after first hop, every arrival is via route node
    }
    reverse(tmp.begin(), tmp.end());
    path.insert(path.end(), tmp.begin(), tmp.end());
    return true;
}

/*------------------------------------------------- placement & full solve */
struct SolveResult
{
    Grid grid;
    vector<pii> pos; // node id → (r,c)
    vector<vector<Segment>> paths;  // one vector per edge (same order as Problem.adj iteration)
};

bool solve(const Problem &problem, SolveResult &result, int trials = 100000) {

    // collect which nodes are inputs / outputs / mids
    const int NUM_NODES = problem.pt.size();
    vector<int> inDeg(NUM_NODES, 0), outDeg(NUM_NODES, 0);
    for (int u = 0; u < NUM_NODES; ++u)
        for (int v : problem.adj[u])
        {
            ++outDeg[u];
            ++inDeg[v];
        }
    vector<int> ins, outs, mids;
    for (int i = 0; i < NUM_NODES; ++i)
        if (inDeg[i] == 0)
            ins.push_back(i);
        else if (outDeg[i] == 0)
            outs.push_back(i);
        else
            mids.push_back(i);

    // collect edge positions
    vector<pii> edge_positions;
    for (int i = 0; i < problem.R; i++) {
        edge_positions.push_back({i, 0});
        edge_positions.push_back({i, problem.C - 1});
    }
    for (int j = 0; j < problem.C; j++) {
        edge_positions.push_back({0, j});
        edge_positions.push_back({problem.R - 1, j});
    }


    // mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
    mt19937 rng(0);
    Grid grid;
    vector<pii> node_position(NUM_NODES);

    for (int t = 0; t < trials; ++t)
    {
        grid = Grid(problem.R, problem.C);
        fill(node_position.begin(), node_position.end(), make_pair(-1, -1));
        // place inputs / outputs at random places on the edges
        assert(ins.size() + outs.size() < grid.R * 2 + grid.C * 2 - 4); // enough space for all ins and outs
        shuffle(edge_positions.begin(), edge_positions.end(), rng);
        for (auto v : {ins, outs}) {
            for (int i = 0; i < v.size(); i++) {
                int x = edge_positions[i].first, y = edge_positions[i].second;
                grid.at(x, y).comp = ins[i];
                node_position[v[i]] = {x, y};
            }
        }

        // free list for mids
        vector<pii> free;
        for (int r = 0; r < grid.R; ++r)
            for (int c = 0; c < grid.C; ++c)
                if (grid.at(r, c).comp == -1)
                    free.emplace_back(r, c);
        if (free.size() < mids.size())
            return false;
        shuffle(free.begin(), free.end(), rng);
        for (size_t i = 0; i < mids.size(); ++i)
        {
            auto [r, c] = free[i];
            grid.at(r, c).comp = mids[i];
            node_position[mids[i]] = {r, c};
        }
        
        
        // route all edges
        EdgeUse edge_use(grid.R, grid.C);
        bool ok = true;
        vector<vector<Segment>> paths;
        for (int u = 0; u < NUM_NODES && ok; ++u)
            for (int v : problem.adj[u]) {
                vector<Segment> path;
                ok = routeEdge(grid, edge_use, node_position[u], node_position[v], path);
                paths.push_back(move(path));
            }
        if (ok) {
            result.grid = move(grid);
            result.pos = move(node_position);
            result.paths = move(paths);    
            return true;
        }
    }
    return false;
}

/*------------------------------------------------- output ----------*/
void writeOut(const Problem &problem, const SolveResult &result, const string &fn) {   
    string parent_dir = "graphs/" + fn + "/"; 
    ofstream f(parent_dir + fn + "_placement.txt");
    f << "# Grid\n"
      << result.grid.R << " " << result.grid.C << "\n";
    f << "# Tiles (row col comp routeCnt)\n";

    for (int r = 0; r < result.grid.R; ++r)
        for (int c = 0; c < result.grid.C; ++c) {
            const Tile &t = result.grid.at(r,c);
            /* ---- ASSERT LEGAL CONFIG ---- */
            if (t.comp != -1)               // has a compute
                assert(t.rCnt <= 1);
            else                            // empty of compute
                assert(t.rCnt <= 2);
            /* ---- --------------------------------- ---- */

            if (t.comp != -1 || t.rCnt)   // emit non-empty tiles only
                f << r << ' ' << c << ' ' << t.comp << ' ' << t.rCnt << '\n';
        }

    f << "# Paths (each edge listed in order given in input)\n";
    f << result.paths.size() << endl;
    size_t idx = 0;
    for (const auto &vec : result.paths)
    {
        f << "EDGE " << idx++ << " " << vec.size() << "\n";
        for (const auto &s : vec)
        {
            f << s.r << " " << s.c << " " << dirChr(s.dir) << " " << nkChr(s.from) << " " << nkChr(s.to) << "\n";
        }
    }
    f << "# Node Positions and Processing times\n";
    f << result.pos.size() << endl;
    for (int i = 0; i < result.pos.size(); i++) {
        f << i << " " << result.pos[i].first << " " << result.pos[i].second << " " << problem.pt[i] << endl;
    }
    
}

/*------------------------------------------------- main ------------*/
int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " <input_name> [trials]\n";
        return 1;
    }
    string in = argv[1];
    string out = in;
    int trials = (argc > 2 ? stoi(argv[2]) : 2000);
    try
    {
        Problem P = readDag(in);
        SolveResult S;
        if (!solve(P, S, trials))
        {
            cout << "no legal embedding found\n";
            return 0;
        }
        writeOut(P, S, out);
        cout << "✓ placement & routes written to " << out << "\n";
    }
    catch (const exception &e)
    {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
