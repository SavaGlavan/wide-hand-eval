use crate::core::card::{Card, RANK_MASK, Rank, Suit};
use crate::core::hand::Hand;
use std::simd::u64x16;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::ptr;

const U64_RANDOM_BOTTOM_BIT_INVERSE: u64 = 0xFFFFFFFFFFFFFFFE;

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

    let mut base_rem_deck: Vec<Card> =
        Vec::with_capacity((52 - used_cards_bitmask.count_ones()) as usize);
    const FULL_DECK: u64 = 0x7FFC_7FFC_7FFC_7FFC;
    let available_cards_bitmask = FULL_DECK & (!used_cards_bitmask);

    for count_one in 0..64 {
        if available_cards_bitmask & (1u64 << count_one) != 0 {
            base_rem_deck.push(Card::new(
                Rank::try_from(count_one as u8).unwrap(),
                Suit::try_from(count_one as u8).unwrap(),
            ));
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
                // Prevents heap allocation overhead inside the inner Monte Carlo loop
                let rng: SmallRng = rand::make_rng();
                let local_deck = base_rem_deck.clone();
                let hands_buffer = vec![0u64; 16 * (1 + num_opponents)];
                let scores_buffer = vec![0u32; 16 * (1 + num_opponents)];
                (rng, local_deck, hands_buffer, scores_buffer)
            },
            |(rng, rem_deck, hands, scores), _batch_idx| {
                let mut local_equity = 0.0;
                let local_games = 16;
                let mut counter = 0;

                // 1. GENERATION STAGE
                for _ in 0..16 {
                    // NATIVE PARTIAL SHUFFLE
                    // Shuffles exactly `cards_to_draw` elements uniformly at random and
                    // moves them to the front of the `rem_deck` slice.
                    let _ = rem_deck.partial_shuffle(rng, cards_to_draw);

                    let completed_board: [Card; 5] = unsafe {
                        let mut arr = MaybeUninit::<[Card; 5]>::uninit();
                        let ptr = arr.as_mut_ptr() as *mut Card;
                        let board_len = board.len();

                        if board_len > 0 {
                            ptr::copy_nonoverlapping(board.as_ptr(), ptr, board_len);
                        }

                        // This remains perfectly safe because `partial_shuffle` leaves
                        // the drawn cards neatly packed at the beginning of `rem_deck`.
                        ptr::copy_nonoverlapping(
                            rem_deck.as_ptr(),
                            ptr.add(board_len),
                            board_cards_needed,
                        );
                        arr.assume_init()
                    };

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
                            rem_deck[board_cards_needed + 2 * j],
                            rem_deck[board_cards_needed + 2 * j + 1],
                        ]);
                    }
                    counter += 1;
                }

                // 2. SIMD EVALUATION STAGE
                // Extract into chunks of exactly 8. This guarantees no remainder and
                // tells LLVM it can unroll and vectorize without bounds checks.
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
