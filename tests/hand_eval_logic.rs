#![feature(portable_simd)]
use wide_hand_eval::core::card::*;
use wide_hand_eval::core::equity::hand_arr_to_u64;
use wide_hand_eval::core::hand::*;

use std::simd::u64x16;

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to adapt the old array-based test cases to the new SIMD architecture.
    /// It converts a 7-card array to a u64, loads it into a u64x8 SIMD vector,
    /// evaluates the hand, and extracts the scalar u32 score for standard comparison.
    fn eval_hand(cards: [Card; 7]) -> u32 {
        let hand_u64 = hand_arr_to_u64(&cards);
        let pack = u64x16::from_array([hand_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let res_hand = Hand::new(pack);
        res_hand.to_array()[0]
    }

    mod hierarchy {
        use super::*;
        #[test]
        fn straight_flush_beats_four_of_a_kind() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Diamonds),
                    Card::new(Rank::Five, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Three, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Hearts),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn four_of_a_kind_beats_full_house() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn full_house_beats_flush() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Seven, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn flush_beats_straight() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Seven, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn straight_beats_three_of_a_kind() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Five, Suit::Clubs),
                    Card::new(Rank::Four, Suit::Spades),
                    Card::new(Rank::Three, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn three_of_a_kind_beats_two_pair() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn two_pair_beats_pair() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn pair_beats_high_card() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Clubs),
                    Card::new(Rank::Eight, Suit::Spades),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod straight_flushes {
        use super::*;
        #[test]
        fn high_beats_low() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn low_ace() {
            assert!(
                eval_hand([
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Three, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Three, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kickers() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod quads {
        use super::*;
        #[test]
        fn high_beats_low() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::King, Suit::Clubs),
                    Card::new(Rank::King, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_second_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ace, Suit::Spades),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod full_houses {
        use super::*;
        #[test]
        fn higher_triple_wins() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::King, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn equal_triple_higher_pair_wins() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::King, Suit::Spades),
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::King, Suit::Spades),
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Nine, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::King, Suit::Spades),
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod flushes {
        use super::*;
        #[test]
        fn high_beats_low_high() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn high_beats_low_mid() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn high_beats_low_low() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test] // this test should not matter cause multiple flushes are impossible
        fn suits_equal() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Diamonds),
                    Card::new(Rank::Six, Suit::Diamonds),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Hearts),
                ])
            )
        }
        #[test]
        fn only_5_cards() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kickers() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod straights {
        use super::*;
        #[test]
        fn high_beats_low() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Hearts),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn low_ace() {
            assert!(
                eval_hand([
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Three, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Four, Suit::Hearts),
                    Card::new(Rank::Three, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kickers() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod three_of_a_kinds {
        use super::*;
        #[test]
        fn higher_triple_wins() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::King, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Nine, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn second_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Seven, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ace, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod two_pairs {
        use super::*;
        #[test]
        fn higher_pair_wins() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn higher_pair_equal_lower_pair_wins() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod pairs {
        use super::*;
        #[test]
        fn higher_pair_wins() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Ten, Suit::Clubs),
                    Card::new(Rank::Nine, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::King, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Nine, Suit::Spades),
                    Card::new(Rank::Eight, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Jack, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn second_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Ten, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn third_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn no_kicker() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Six, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Ace, Suit::Diamonds),
                    Card::new(Rank::Queen, Suit::Clubs),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Hearts),
                    Card::new(Rank::Five, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }

    mod high_cards {
        use super::*;
        #[test]
        fn equal() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) == eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn first() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::King, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn second() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Jack, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn third() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Nine, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn fourth() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Seven, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
        #[test]
        fn fifth() {
            assert!(
                eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Six, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ]) > eval_hand([
                    Card::new(Rank::Ace, Suit::Hearts),
                    Card::new(Rank::Queen, Suit::Spades),
                    Card::new(Rank::Ten, Suit::Diamonds),
                    Card::new(Rank::Eight, Suit::Clubs),
                    Card::new(Rank::Five, Suit::Hearts),
                    Card::new(Rank::Four, Suit::Diamonds),
                    Card::new(Rank::Two, Suit::Diamonds),
                ])
            )
        }
    }
}
