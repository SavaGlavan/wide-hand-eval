#![feature(portable_simd)]
mod core;
use crate::core::card::*;
use crate::core::equity::calculate_equity_unified_mc_avx512;
use clap::Parser;
use std::time::Instant;

use rand::Rng;
use rand::rand_core::{RngCore, SeedableRng};
use simd_rand::portable::*;
use std::mem::transmute;
use std::simd::prelude::*;
use std::simd::{Simd, mask8x64, u8x64, u32x16};

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

fn extract_non_zeros(vector: u8x64) -> Vec<u8> {
    // 1. Create a SIMD mask where lanes != 0 are true
    let mask = vector.simd_ne(u8x64::splat(0));

    // 2. Convert to a single u64 bitmask (compiles to `vptestmb` on AVX-512)
    let mut bitmask = mask.to_bitmask();

    // 3. Pre-allocate exact capacity to avoid reallocations
    let mut extracted = Vec::with_capacity(bitmask.count_ones() as usize);
    let array = vector.to_array();

    // 4. Rapidly jump from set-bit to set-bit
    while bitmask != 0 {
        let index = bitmask.trailing_zeros(); // Hardware tzcnt

        // Extract the non-zero value
        extracted.push(array[index as usize]);

        // Clear the lowest set bit (compiles to `blsr` instruction)
        bitmask &= bitmask - 1;
    }

    extracted
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

    // let fixed_forbidden = [
    //     0b00001111,
    //     0b00000000,
    //     0b00000001,
    //     hole[0].rank() as u8,
    //     hole[1].rank() as u8,
    // ];
    //
    // let forbidden_cards_splats: Vec<u8x64> = fixed_forbidden
    //     .into_iter()
    //     .chain(board.iter().map(|card| card.rank() as u8))
    //     .chain(
    //         specific_opponents
    //             .iter()
    //             .flat_map(|hand| [hand[0].rank() as u8, hand[1].rank() as u8]),
    //     )
    //     .map(u8x64::splat)
    //     .collect();
    //
    // // for card in forbidden_cards_splats {
    // //     println!("{:?}", card[0]);
    // // }
    //
    // const CARD_MASK: Simd<u8, 64> = u8x64::splat(0b00111111);
    //
    // let mut seed = Xoshiro256PlusPlusX8Seed::default();
    // rand::rng().fill_bytes(&mut *seed);
    // let mut rng = Xoshiro256PlusPlusX8::from_seed(seed);
    //
    // let board_cards_needed = 5 - board.len();
    // let opp_cards_needed = 2 * num_random_opponents;
    // let cards_to_draw = (board_cards_needed + opp_cards_needed) * 1_000_000;
    // println!("Cards needed: {}", cards_to_draw);
    // let mut random_cards: Vec<u8> = Vec::with_capacity(cards_to_draw);
    // let mut current_len = 0;

    let start_time = Instant::now(); // begin timer to benchmark equity calcs

    // /*
        let equity = calculate_equity_unified_mc_avx512(
            hole,
            &board,
            &specific_opponents,
            num_random_opponents,
            actual_cards,
        );
    // */
    //
    // while current_len < cards_to_draw {
    //     let mut vector: u8x64 = unsafe { transmute(rng.next_u64x8()) };
    //     vector &= CARD_MASK;
    //
    //     for card in &forbidden_cards_splats {
    //         let repeat_catch_check: mask8x64 = vector.simd_eq(*card);
    //         vector = repeat_catch_check.select(u8x64::splat(0), vector);
    //     }
    //
    //     let res = extract_non_zeros(vector);
    //     random_cards.extend_from_slice(&res);
    //     current_len += res.len();
    // }

    let duration = start_time.elapsed();

    // println!(
    //     "Time elapsed: {:?}, Cards generated: {:?}",
    //     duration, current_len
    // );

    // for card in res {
    //     println!("{:08b}", card);
    // }

    println!(
        "Estimated Equity: {:.2}%, Time elapsed: {:?}",
        equity * 100.0,
        duration
    );
}
