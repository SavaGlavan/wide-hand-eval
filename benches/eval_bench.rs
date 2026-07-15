#![feature(portable_simd)]
use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::SmallRng;
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::simd::u64x16;

use wide_hand_eval::core::card::{Card, Rank, Suit};
use wide_hand_eval::core::equity::hand_arr_to_u64;
use wide_hand_eval::core::hand::Hand;

fn bench_hand_new(c: &mut Criterion) {
    const FULL_DECK: u64 = 0x7FFC_7FFC_7FFC_7FFC;
    let mut base_deck: Vec<Card> = Vec::with_capacity(52);

    for count_one in 0..64 {
        if FULL_DECK & (1u64 << count_one) != 0 {
            base_deck.push(Card::new(
                Rank::try_from(count_one as u8).unwrap(),
                Suit::try_from(count_one as u8).unwrap(),
            ));
        }
    }

    let mut rng: SmallRng = rand::make_rng();
    let mut hands_u64 = [0u64; 16];

    // 2. Generate 16 independent random 7-card hands
    for i in &mut hands_u64 {
        let mut local_deck = base_deck.clone();
        let (drawn_cards, _) = local_deck.partial_shuffle(&mut rng, 7);
        let hand_arr: [Card; 7] = drawn_cards.try_into().unwrap();
        *i = hand_arr_to_u64(&hand_arr);
    }

    let my_pack: u64x16 = u64x16::from_array(hands_u64);

    c.bench_function("Hand::new", |b| {
        b.iter(|| {
            let hand = Hand::new(black_box(my_pack));
            black_box(hand);
        })
    });
}

criterion_group!(benches, bench_hand_new);
criterion_main!(benches);
