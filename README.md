# place-and-route



### Place-and-Route Pipeline

To create a DAG input for a specific category:

```bash
python3 generate_input.py <num_inputs> <num_outputs> <avg_degree> <total_nodes> <min_proc_time> <max_proc_time> [file_name]
```

This will generate a file at:

```
graphs/<file_name>/<file_name>_input.txt
```

Compile the C++ file and run it over the generated input:

```bash
g++ dag_place_and_route.cpp -o dag_place_and_route
./dag_place_and_route <file_name>
```

Replace `<file_name>` with the one you used when generating the input.

To visualize the mapped DAG on a 2D grid:

```bash
python3 grid_visualizer.py <file_name>
```

---

#### Example Workflow

```bash
# 1. Generate input DAG
python3 generate_input.py 3 2 1.8 10 1 5 test_case

# 2. Run place-and-route
g++ dag_place_and_route.cpp -o dag_place_and_route
./dag_place_and_route test_case

# 3. Visualize the placement
python3 grid_visualizer.py test_case
```