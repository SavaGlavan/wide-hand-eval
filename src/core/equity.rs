use crate::core::card::{Card, RANK_MASK, Rank, Suit};
use crate::core::hand::Hand;
use std::simd::prelude::*;
use std::simd::{Simd, mask8x64, u8x64, u64x16};

use rand::SeedableRng;
use rayon::prelude::*;
use std::mem::transmute;
use simd_rand::portable::*;

const U64_RANDOM_BOTTOM_BIT_INVERSE: u64 = 0xFFFFFFFFFFFFFFFE;
const CARD_MASK: Simd<u8, 64> = u8x64::splat(0b00111111);

#[inline(always)]
pub fn hand_arr_to_u64(cards: &[Card; 7]) -> u64 {
    let mut hand_u64 = 0u64;

    for i in cards.iter().take(7) {
        let card_val = i.0;
        let is_ace = ((card_val & RANK_MASK) == 14) as u64;
        let low_ace_shift = card_val ^ RANK_MASK;
        hand_u64 |= (1u64 << card_val) | (is_ace << low_ace_shift);
    }

    hand_u64 & U64_RANDOM_BOTTOM_BIT_INVERSE
}

pub fn calculate_equity_unified_mc_avx512(
    hole: [Card; 2],
    board: &[Card],
    opponents: &[[Card; 2]],
    num_random_opponents: usize,
    used_cards_bitmask: u64,
) -> f64 {
    let num_opponents = opponents.len() + num_random_opponents;
    let board_cards_needed = 5 - board.len();
    let opp_cards_needed = 2 * num_random_opponents;
    let cards_to_draw = board_cards_needed + opp_cards_needed;

    const FULL_DECK: u64 = 0x7FFC_7FFC_7FFC_7FFC;
    let available_cards_bitmask = FULL_DECK & (!used_cards_bitmask);

    // Pre-calculate forbidden splats using the bitmask to avoid allocating in the hot loop
    let mut forbidden_cards_splats = Vec::with_capacity(64);
    for i in 0..64 {
        // If a card is NOT available, add it to the forbidden splat mask
        if available_cards_bitmask & (1u64 << i) == 0 {
            forbidden_cards_splats.push(u8x64::splat(i as u8));
        }
    }

    // Create a fast lookup table to instantly convert `u8` back to `Card` without `try_from` overhead
    let mut card_lookup = [hole[0]; 64];
    for count_one in 0..64 {
        if available_cards_bitmask & (1u64 << count_one) != 0 {
            card_lookup[count_one] = Card::new(
                Rank::try_from(count_one as u8).unwrap(),
                Suit::try_from(count_one as u8).unwrap(),
            );
        }
    }

    // Force batches to align perfectly with the x86 AVX lanes
    const TOTAL_BATCHES: u32 = 1_000_000;

    // We utilize Rayon's Work Stealing via par_iter
    let (total_equity, total_games) = (0..TOTAL_BATCHES)
        .into_par_iter()
        .map_init(
            || {
                // Thread-local state initialization
                // Seeds the high-performance SIMD RNG directly from the thread RNG
                let rng = Xoshiro256PlusPlusX8::from_rng(&mut rand::rng());
                let hands_buffer = vec![0u64; 16 * (1 + num_opponents)];
                let scores_buffer = vec![0u32; 16 * (1 + num_opponents)];
                (rng, hands_buffer, scores_buffer)
            },
            |(rng, hands, scores), _batch_idx| {
                let mut local_equity = 0.0;
                let local_games = 16;
                let mut counter = 0;

                // 1. GENERATION STAGE
                for _ in 0..16 {
                    let mut drawn_cards = [0u8; 64];
                    let mut current_len = 0;
                    let mut drawn_cards_mask = 0u64; // Prevents intra-vector duplicates

                    // SIMD RANDOM CARD GENERATION
                    while current_len < cards_to_draw {
                        let mut vector: u8x64 = unsafe { transmute(rng.next_u64x8()) };
                        vector &= CARD_MASK;

                        // Apply forbidden masks
                        for card in forbidden_cards_splats.iter() {
                            let repeat_catch_check: mask8x64 = vector.simd_eq(*card);
                            vector = repeat_catch_check.select(u8x64::splat(0), vector);
                        }

                        let mut bitmask = vector.simd_ne(u8x64::splat(0)).to_bitmask();
                        let array = vector.to_array();

                        while bitmask != 0 && current_len < cards_to_draw {
                            let index = bitmask.trailing_zeros(); // Hardware tzcnt
                            let card_val = array[index as usize];

                            // Ensure we haven't already drawn this card in this specific game
                            if (drawn_cards_mask & (1u64 << card_val)) == 0 {
                                drawn_cards_mask |= 1u64 << card_val;
                                drawn_cards[current_len] = card_val;
                                current_len += 1;
                            }

                            bitmask &= bitmask - 1; // Clear lowest set bit
                        }
                    }

                    // Pack the completed board (Replaces the unsafe MaybeUninit block)
                    let mut completed_board = [hole[0]; 5];
                    let board_len = board.len();
                    if board_len > 0 {
                        completed_board[..board_len].copy_from_slice(board);
                    }
                    for i in 0..board_cards_needed {
                        completed_board[board_len + i] = card_lookup[drawn_cards[i] as usize];
                    }

                    // Pack Hero's Hand
                    hands[counter] = hand_arr_to_u64(&[
                        completed_board[0],
                        completed_board[1],
                        completed_board[2],
                        completed_board[3],
                        completed_board[4],
                        hole[0],
                        hole[1],
                    ]);

                    // Pack Known Opponents
                    for j in opponents {
                        counter += 1;
                        hands[counter] = hand_arr_to_u64(&[
                            completed_board[0],
                            completed_board[1],
                            completed_board[2],
                            completed_board[3],
                            completed_board[4],
                            j[0],
                            j[1],
                        ]);
                    }

                    // Pack Random Opponents
                    for j in 0..num_random_opponents {
                        counter += 1;
                        hands[counter] = hand_arr_to_u64(&[
                            completed_board[0],
                            completed_board[1],
                            completed_board[2],
                            completed_board[3],
                            completed_board[4],
                            card_lookup[drawn_cards[board_cards_needed + 2 * j] as usize],
                            card_lookup[drawn_cards[board_cards_needed + 2 * j + 1] as usize],
                        ]);
                    }
                    counter += 1;
                }

                // 2. SIMD EVALUATION STAGE
                let mut out_idx = 0;
                for chunk in hands.chunks_exact(16) {
                    let in_buffer: [u64; 16] = chunk.try_into().unwrap();
                    let a: u64x16 = u64x16::from_array(in_buffer);

                    let c = Hand::new(a);
                    let out_buffer: [u32; 16] = c.to_array();

                    scores[out_idx..out_idx + 16].copy_from_slice(&out_buffer);
                    out_idx += 16;
                }

                // 3. TALLY STAGE (Accounting for Fractional Pot / Ties)
                for i in 0..16 {
                    let hero_score_idx = i * (1 + num_opponents);
                    let hero_score = scores[hero_score_idx];

                    let mut max_score = hero_score;
                    let mut winners_count = 1;

                    for j in 1..=(num_opponents) {
                        let opp_score = scores[hero_score_idx + j];
                        if opp_score > max_score {
                            max_score = opp_score;
                            winners_count = 1;
                        } else if opp_score == max_score {
                            winners_count += 1;
                        }
                    }

                    // Add proportional equity. A clean win adds 1.0. A 2-way tie adds 0.5.
                    if hero_score == max_score {
                        local_equity += 1.0 / (winners_count as f64);
                    }
                }

                (local_equity, local_games)
            },
        )
        // 4. REDUCTION STAGE
        .reduce(
            || (0.0, 0),
            |(eq_a, games_a), (eq_b, games_b)| (eq_a + eq_b, games_a + games_b),
        );

    total_equity / total_games as f64
}