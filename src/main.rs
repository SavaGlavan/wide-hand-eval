#![feature(portable_simd)]
mod core;
use crate::core::card::*;
use crate::core::equity::calculate_equity_unified_mc_avx512;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    author = "Sava Glavan",
    version = "0.1.0",
    about = "A CLI Poker Evaluator"
)]
struct Args {
    /// The player's exactly 2 hole cards. (e.g., --player AS KS)
    #[arg(short = 'p', long = "player", num_args = 2, required = true)]
    player: Vec<Card>,

    /// The board cards, up to 5. (e.g., --board 2H 3D 4C)
    #[arg(
        short = 'b',
        long = "board",
        num_args = 0..=5
    )]
    board: Option<Vec<Card>>,

    /// Number of random opponents (e.g., --num-opponents 3)
    #[arg(short = 'n', long = "num-opponents")]
    num_opponents: Option<usize>,

    /// Specific opponent hands. Specify multiple times, 2 cards each.
    /// (e.g., -o 2H 3D -o AC JC)
    #[arg(
        short = 'o',
        long = "opponent-hand",
        num_args = 2,
        action = clap::ArgAction::Append,
    )]
    opponent_hands: Option<Vec<Card>>,
}

fn main() {
    // handling input args begins
    let args = Args::parse();

    let hole: [Card; 2] = [args.player[0], args.player[1]];

    let board: Vec<Card> = args.board.unwrap_or_default();
    assert!(board.len() <= 5, "Board cannot have more than 5 cards.");

    let mut specific_opponents = Vec::new();
    if let Some(hands) = args.opponent_hands {
        for chunk in hands.chunks(2) {
            specific_opponents.push([chunk[0], chunk[1]]);
        }
    }

    let num_random_opponents = args.num_opponents.unwrap_or(0);
    let num_opponents = specific_opponents.len() + num_random_opponents;
    assert!(
        (1..=21).contains(&num_opponents),
        "Invalid number of opponents."
    );

    // ensure that no cards are reused in input
    let expected_cards: u32 = (2 + board.len() + specific_opponents.len() * 2) as u32;
    let actual_cards = board.iter().fold(0, |acc, &card| acc | 1u64 << card.0)
        ^ hole.iter().fold(0, |acc, &card| acc | 1u64 << card.0)
        ^ specific_opponents.iter().fold(0, |acc, &hand| {
            acc | hand.iter().fold(0, |acc, &card| acc | 1u64 << card.0)
        });
    assert_eq!(
        expected_cards,
        actual_cards.count_ones(),
        "Cards are unique, cannot appear twice."
    );
    // completed all argument processing and validating

    println!("Calculating equity...");

    let start_time = Instant::now(); // begin timer to benchmark equity calcs

    let equity = calculate_equity_unified_mc_avx512(
        hole,
        &board,
        &specific_opponents,
        num_random_opponents,
        actual_cards,
    );

    let duration = start_time.elapsed();

    println!(
        "Estimated Equity: {:.2}%, Time elapsed: {:?}",
        equity * 100.0,
        duration
    );
}
