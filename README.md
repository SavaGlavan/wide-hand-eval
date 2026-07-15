# wide-hand-eval: A CLI Poker Evaluator
This is a CLI poker hand evaluator. It calculates estimated win equity using Monte Carlo simulations powered by an AVX-512 SIMD hand scoring algorithm.

## Performance

When benchmarked on AVX-512 supporting hardware, the hand evaluation component is capable of scoring **> 1 billion hands per second**.

## Requirements

### Software
Because this project relies on the experimental `portable_simd` feature, it strictly requires the nightly Rust toolchain.

### Hardware
This evaluator is heavily optimized using AVX-512 SIMD vectorization. For optimal performance a modern CPU with AVX-512 support is highly recommended.

## Building the Project

Compile the project in release mode via:
```bash
cargo build --release
```

## Usage

You can run the generated binary with specific arguments to set up the poker hand scenario.

**Example Command:**
```bash
./target/release/cards --player AS AC -b AD TC 9S -o 8C 6C -n 5
```

### Command Line Arguments

| Argument | Flag | Description |
| :--- | :--- | :--- |
| `--player` | `-p` | The player's exactly 2 hole cards (e.g., `AS KS`). This argument is required. |
| `--board` | `-b` | The board cards, accepting up to 5 total cards (e.g., `2H 3D 4C`). |
| `--num-opponents` | `-n` | The number of random opponents in the hand (e.g., `3`). |
| `--opponent-hand` | `-o` | Specific opponent hands, requiring exactly 2 cards each. You can specify this multiple times to add multiple known opponents. |

## Game Rules & Validation

The evaluator enforces strict validation before running the equity calculation to ensure a valid poker state :

* The board cannot have more than 5 cards.
* The total number of opponents (both specific and random combined) must fall between 1 and 21.
* All inputted cards must be entirely unique; a single card cannot appear twice in the setup.

## Output

Once the setup is validated, the application calculates the hand equity. It will print the estimated equity as a percentage alongside the total time elapsed for the computation.

## Testing and Benchmarking

This project includes a test suite to ensure the accuracy of the poker hand evaluations, as well as a benchmarking suite to measure the performance of the SIMD hand evaluator.

### Running Tests
To run the standard test suite and verify the core evaluation logic :
```bash
cargo test
```

### Running Benchmarks
To evaluate the hand scoring performance on your specific hardware, run the benchmark suite. The benchmarks utilize Criterion and will automatically apply native CPU optimizations to test your machine's vectorization capabilities:
```bash
cargo bench
```
Note that given the SIMD design, each scoring iteration scores 16 hands in parallel. Interpret the benchmark results accordingly.
