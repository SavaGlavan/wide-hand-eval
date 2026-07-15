use std::fmt;
use std::mem::transmute;
use std::simd::prelude::*;
use std::simd::{simd_swizzle, u16x64, u32x16, u64x16};

const TRIP_MASK: Simd<u32, 16> = u32x16::splat(0x30000000);
const STRAIGHT_MASK: Simd<u32, 16> = u32x16::splat(0x40000000);
const FLUSH_MASK: Simd<u32, 16> = u32x16::splat(0x50000000);
const FULL_HOUSE_MASK: Simd<u32, 16> = u32x16::splat(0x60000000);
const QUAD_MASK: Simd<u32, 16> = u32x16::splat(0xE0000000);
const STRAIGHT_FLUSH_MASK: Simd<u32, 16> = u32x16::splat(0xF0000000);
const U16X64_LOW_ACE_MASK_INVERSE: Simd<u16, 64> = u16x64::splat(0xFFFD);
const U32X16_LOW_ACE_MASK_INVERSE: Simd<u32, 16> = u32x16::splat(0xFFFFFFFD);
const U32X16_BOTTOM_HALF: Simd<u32, 16> = u32x16::splat(0x0000FFFF);
const U64X16_BOTTOM_HALF: Simd<u64, 16> = u64x16::splat(0xFFFF);
const U32X16_OVERFLOW_CATCH: Simd<u32, 16> = u32x16::splat(0xF);

#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Hand(u32x16);

impl Hand {
    pub fn new(cards: u64x16) -> Hand {
        let suit_split: u16x64 = unsafe { transmute(cards) };
        let suit_split_clean: u16x64 = suit_split & U16X64_LOW_ACE_MASK_INVERSE;

        let spades: u16x16 = simd_swizzle!(
            suit_split,
            [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
        );
        let diamonds: u16x16 = simd_swizzle!(
            suit_split,
            [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]
        );
        let clubs: u16x16 = simd_swizzle!(
            suit_split,
            [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]
        );
        let hearts: u16x16 = simd_swizzle!(
            suit_split,
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
        );

        let folded_dirty: u32x16 = (spades | diamonds | clubs | hearts).cast();
        let folded_clean: u32x16 = folded_dirty ^ u32x16::splat(2);

        let mut res: u32x16 = folded_dirty;

        let ab_and: u16x16 = spades & diamonds;
        let ab_or: u16x16 = spades | diamonds;
        let cd_and: u16x16 = clubs & hearts;
        let cd_or: u16x16 = clubs | hearts;

        // PAIR & TWO PAIR
        let pair_reduced: u32x16 = (ab_and | cd_and | (ab_or & cd_or)).cast()
            & U32X16_LOW_ACE_MASK_INVERSE
            & U32X16_BOTTOM_HALF;
        let pair_check: mask32x16 = pair_reduced.simd_ne(u32x16::splat(0));
        let two_pair_check: mask32x16 = pair_reduced.count_ones().simd_gt(u32x16::splat(1));

        let first: u32x16 = pair_check.select(
            pair_reduced.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH,
            u32x16::splat(0),
        );
        let second: u32x16 = two_pair_check.select(
            (pair_reduced ^ (u32x16::splat(1) << first)).leading_zeros() & U32X16_OVERFLOW_CATCH
                ^ U32X16_OVERFLOW_CATCH,
            u32x16::splat(0),
        );
        let kickers: u32x16 = two_pair_check.select(
            folded_clean ^ (u32x16::splat(1) << first) ^ (u32x16::splat(1) << second),
            folded_clean ^ (u32x16::splat(1) << first),
        );

        let two_pair_kicker: u32x16 = u32x16::splat(1)
            << (kickers.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);
        let two_pair_res: u32x16 = first << 24 | second << 20 | two_pair_kicker;

        let k1: u32x16 = u32x16::splat(1)
            << (kickers.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);
        let k_rem1: u32x16 = kickers ^ k1;
        let k2: u32x16 = u32x16::splat(1)
            << (k_rem1.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);
        let k_rem2: u32x16 = k_rem1 ^ k2;
        let k3: u32x16 = u32x16::splat(1)
            << (k_rem2.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);
        let pair_res: u32x16 = first << 24 | (k1 | k2 | k3);

        res = pair_check.select(pair_res, res);
        res = two_pair_check.select(two_pair_res, res);

        // THREE OF A KIND
        let triple_reduced: u32x16 =
            ((ab_and & cd_or) | (cd_and & ab_or)).cast() & U32X16_LOW_ACE_MASK_INVERSE;
        let trip_check: mask32x16 = triple_reduced.simd_ne(u32x16::splat(0));

        let trip_rank_idx: u32x16 =
            triple_reduced.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH;
        let trip_kickers: u32x16 = folded_clean ^ (u32x16::splat(1) << trip_rank_idx);

        let trip_k1: u32x16 = u32x16::splat(1)
            << (trip_kickers.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);
        let trip_k_rem1: u32x16 = trip_kickers ^ trip_k1;
        let trip_k2: u32x16 = u32x16::splat(1)
            << (trip_k_rem1.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);

        let triple_res: u32x16 = TRIP_MASK | (triple_reduced << 12) | (trip_k1 | trip_k2);
        res = trip_check.select(triple_res, res);

        // STRAIGHT
        let mut straight_calc: u32x16 = folded_dirty & (folded_dirty << 1);
        straight_calc &= straight_calc << 2;
        straight_calc &= folded_dirty << 4;
        let straight_check: mask32x16 = straight_calc.simd_ne(u32x16::splat(0));

        let straight_res: u32x16 = STRAIGHT_MASK
            | (straight_calc.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH);
        res = straight_check.select(straight_res, res);

        // FLUSH
        let mut flush_calc: u16x64 = suit_split_clean.count_ones();
        let flush_check: mask16x64 = flush_calc.simd_ge(u16x64::splat(5));
        flush_calc = flush_check.select(suit_split_clean, u16x64::splat(0));

        let recast_flush_calc: u64x16 = unsafe { transmute(flush_calc) };
        let first_fold: u64x16 = recast_flush_calc | (recast_flush_calc >> 32);
        let second_fold: u64x16 = first_fold | (first_fold >> 16);

        let mut flush_res: u32x16 = (second_fold & U64X16_BOTTOM_HALF).cast();
        let flush_res_calc: u32x16 = flush_res.count_ones();
        let reduced_flush_check: mask32x16 = flush_res_calc.simd_ge(u32x16::splat(5));
        let reduced_flush_check_6_card: mask32x16 = flush_res_calc.simd_ge(u32x16::splat(6));
        let reduced_flush_check_7_card: mask32x16 = flush_res_calc.simd_ge(u32x16::splat(7));

        let flush_res_minus_1: u32x16 = flush_res - u32x16::splat(1);
        let dropped_1: u32x16 = flush_res & flush_res_minus_1;
        flush_res = reduced_flush_check_6_card.select(dropped_1, flush_res);

        let flush_res_minus_1_again: u32x16 = flush_res - u32x16::splat(1);
        let dropped_2: u32x16 = flush_res & flush_res_minus_1_again;
        flush_res = reduced_flush_check_7_card.select(dropped_2, flush_res);

        res = reduced_flush_check.select(FLUSH_MASK | flush_res, res);

        // FULL HOUSE
        let full_house_check: mask32x16 = trip_check & two_pair_check;
        let fh_pair: u32x16 = pair_reduced ^ (u32x16::splat(1) << trip_rank_idx);
        res = full_house_check.select(FULL_HOUSE_MASK | (triple_reduced << 13) | fh_pair, res);

        // QUADS
        let quad_calc: u32x16 = (ab_and & cd_and).cast();
        let quad_check: mask32x16 = quad_calc.simd_ne(u32x16::splat(0));

        let quad_rank_idx: u32x16 =
            quad_calc.leading_zeros() & U32X16_OVERFLOW_CATCH ^ U32X16_OVERFLOW_CATCH;
        let quad_res: u32x16 = QUAD_MASK
            | (quad_rank_idx << 24)
            | (u32x16::splat(32) - (folded_dirty ^ quad_calc).leading_zeros());
        res = quad_check.select(quad_res, res);

        // STRAIGHT FLUSH
        let mut straight_flush_calc: u64x16 = cards & (cards << 1);
        straight_flush_calc &= straight_flush_calc << 2;
        straight_flush_calc &= cards << 4;
        let cleaned_straight_flush_calc: u32x16 =
            (straight_flush_calc | (straight_flush_calc >> 32)).cast();
        let straight_flush_check: mask32x16 = cleaned_straight_flush_calc.simd_ne(u32x16::splat(0));

        let straight_flush_res: u32x16 = STRAIGHT_FLUSH_MASK
            | (cleaned_straight_flush_calc.leading_zeros() & U32X16_OVERFLOW_CATCH
                ^ U32X16_OVERFLOW_CATCH);
        res = straight_flush_check.select(straight_flush_res, res);

        Hand(res)
    }

    pub fn to_array(&self) -> [u32; 16] {
        self.0.to_array()
    }
}

impl fmt::Binary for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f,
               "\n0: {:032b}\n1: {:032b}\n2: {:032b}\n3: {:032b}\n4: {:032b}\n5: {:032b}\n6: {:032b}\n7: {:032b}
               \n8: {:032b}\n9: {:032b}\n10: {:032b}\n11: {:032b}\n12: {:032b}\n13: {:032b}\n14: {:032b}\n15: {:032b}
               ",
               a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
    }
}
