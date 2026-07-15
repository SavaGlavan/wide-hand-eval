use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

const SUIT_MASK: u8 = 0b0011_0000; // Second 2 bits
pub const RANK_MASK: u8 = 0b0000_1111; // Last 4 bits

#[repr(u8)] // Tells the compiler to store the enum discriminants as u8 (1 byte)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum Suit {
    Hearts = 0b0000_0000,
    Clubs = 0b0001_0000,
    Diamonds = 0b0010_0000,
    Spades = 0b0011_0000,
}
impl TryFrom<u8> for Suit {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & SUIT_MASK {
            0b0000_0000 => Ok(Suit::Hearts),
            0b0001_0000 => Ok(Suit::Clubs),
            0b0010_0000 => Ok(Suit::Diamonds),
            0b0011_0000 => Ok(Suit::Spades),
            _ => Err("Invalid rank value"),
        }
    }
}

impl FromStr for Suit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "H" => Ok(Suit::Hearts),
            "C" => Ok(Suit::Clubs),
            "D" => Ok(Suit::Diamonds),
            "S" => Ok(Suit::Spades),
            _ => Err(format!("Invalid suit '{}'", s)),
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Suit::Hearts => "H",
                Suit::Clubs => "C",
                Suit::Diamonds => "D",
                Suit::Spades => "S",
            }
        )
    }
}

impl fmt::Binary for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&(*self as u8), f)
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum Rank {
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Jack = 11,
    Queen = 12,
    King = 13,
    Ace = 14,
}

impl TryFrom<u8> for Rank {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & RANK_MASK {
            2 => Ok(Rank::Two),
            3 => Ok(Rank::Three),
            4 => Ok(Rank::Four),
            5 => Ok(Rank::Five),
            6 => Ok(Rank::Six),
            7 => Ok(Rank::Seven),
            8 => Ok(Rank::Eight),
            9 => Ok(Rank::Nine),
            10 => Ok(Rank::Ten),
            11 => Ok(Rank::Jack),
            12 => Ok(Rank::Queen),
            13 => Ok(Rank::King),
            14 => Ok(Rank::Ace),
            _ => Err("Invalid rank value"),
        }
    }
}
impl FromStr for Rank {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "2" => Ok(Rank::Two),
            "3" => Ok(Rank::Three),
            "4" => Ok(Rank::Four),
            "5" => Ok(Rank::Five),
            "6" => Ok(Rank::Six),
            "7" => Ok(Rank::Seven),
            "8" => Ok(Rank::Eight),
            "9" => Ok(Rank::Nine),
            "10" | "T" => Ok(Rank::Ten),
            "J" => Ok(Rank::Jack),
            "Q" => Ok(Rank::Queen),
            "K" => Ok(Rank::King),
            "A" => Ok(Rank::Ace),
            _ => Err(format!("Invalid rank '{}'", s)),
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Rank::Two => '2',
                Rank::Three => '3',
                Rank::Four => '4',
                Rank::Five => '5',
                Rank::Six => '6',
                Rank::Seven => '7',
                Rank::Eight => '8',
                Rank::Nine => '9',
                Rank::Ten => 'T',
                Rank::Jack => 'J',
                Rank::Queen => 'Q',
                Rank::King => 'K',
                Rank::Ace => 'A',
            }
        )
    }
}

impl fmt::Binary for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&(*self as u8), f)
    }
}

#[repr(transparent)] // Guarantees the struct has the same layout as u8
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Card(pub(crate) u8);

impl Card {
    pub const fn new(rank: Rank, suit: Suit) -> Self {
        Self(rank as u8 ^ suit as u8)
    }
    pub fn suit(&self) -> Suit {
        match self.0 & SUIT_MASK {
            0b0000_0000 => Suit::Hearts,
            0b0001_0000 => Suit::Clubs,
            0b0010_0000 => Suit::Diamonds,
            0b0011_0000 => Suit::Spades,
            _ => unreachable!(), // or handle invalid state
        }
    }

    pub fn rank(&self) -> Rank {
        match self.0 & RANK_MASK {
            2 => Rank::Two,
            3 => Rank::Three,
            4 => Rank::Four,
            5 => Rank::Five,
            6 => Rank::Six,
            7 => Rank::Seven,
            8 => Rank::Eight,
            9 => Rank::Nine,
            10 => Rank::Ten,
            11 => Rank::Jack,
            12 => Rank::Queen,
            13 => Rank::King,
            14 => Rank::Ace,
            _ => unreachable!(),
        }
    }
}

impl FromStr for Card {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_uppercase();

        if s.len() < 2 || s.len() > 3 {
            return Err(format!("Invalid length for card string: '{}'", s));
        }

        let (rank_str, suit_str) = s.split_at(s.len() - 1);

        let rank = rank_str.parse::<Rank>()?;
        let suit = suit_str.parse::<Suit>()?;

        Ok(Card::new(rank, suit))
    }
}

impl PartialOrd<Self> for Card {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Card {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

// should look into this if I want to make it really nice / safe: https://en.wikipedia.org/wiki/Playing_cards_in_Unicode
impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.rank(), self.suit())
    }
}

impl fmt::Binary for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}
