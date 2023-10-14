use std::collections::HashSet;
use std::fmt::Display;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, UNIX_EPOCH};

use crate::card::{CardCache, SavedCard};
use crate::paths::get_cards_path;
use crate::Id;
use std::io::{self, ErrorKind};

use std::time::SystemTime;

pub fn duration_to_days(dur: &Duration) -> f32 {
    dur.as_secs_f32() / 86400.
}

type Filter = (String, Box<dyn FnMut(&SavedCard) -> bool>);

#[derive(Default)]
struct Filters {
    positive: Vec<Filter>,
    negative: Vec<Filter>,
}

impl Filters {
    fn run<'a>(&mut self, cards: HashSet<&'a Id>, cache: &mut CardCache) -> HashSet<&'a Id> {
        let mut accepted_cards = HashSet::new();
        for card_id in cards {
            let card = cache.get_ref(card_id);
            if self.positive.iter_mut().all(|filter| filter.1(&card))
                && !self.negative.iter_mut().any(|filter| filter.1(&card))
            {
                accepted_cards.insert(card_id);
            }
        }
        accepted_cards
    }

    fn insert_positive(&mut self, filter: Filter) {
        self.positive.push(filter);
    }
    fn insert_negative(&mut self, filter: Filter) {
        self.negative.push(filter);
    }

    fn is_pending(&self) -> Filter {
        let closure = move |card: &SavedCard| -> bool { card.is_pending() };

        ("is pending".to_string(), Box::new(closure))
    }

    fn is_finished(&self) -> Filter {
        let closure = move |card: &SavedCard| -> bool { card.is_finished() };

        ("is finished".to_string(), Box::new(closure))
    }

    fn is_suspended(&self) -> Filter {
        let closure = move |card: &SavedCard| -> bool { card.is_suspended() };

        ("is suspended".to_string(), Box::new(closure))
    }

    fn has_tag(&self, tag: String) -> Filter {
        let s = format!("includes tag:  {}", &tag);
        let closure = move |card: &SavedCard| -> bool { card.contains_tag(&tag) };
        (s, Box::new(closure))
    }

    fn max_strength(&self, max_strength: Duration) -> Filter {
        let closure = move |card: &SavedCard| -> bool {
            if let Some(strength_rate) = card.strength() {
                strength_rate > max_strength
            } else {
                false
            }
        };
        let s = format!("strength < {}", duration_to_days(&max_strength));
        (s, Box::new(closure))
    }

    fn max_stability(&self, max_stability: Duration) -> Filter {
        let closure = move |card: &SavedCard| -> bool {
            if let Some(stability_rate) = card.stability() {
                stability_rate > max_stability
            } else {
                false
            }
        };
        let s = format!("stability < {}", duration_to_days(&max_stability));
        (s, Box::new(closure))
    }

    fn max_recall(&self, max_recall: f32) -> Filter {
        let closure = move |card: &SavedCard| -> bool {
            if let Some(recall_rate) = card.recall_rate() {
                recall_rate > max_recall
            } else {
                false
            }
        };
        let s = format!("recall < {max_recall}");
        (s, Box::new(closure))
    }
}

pub fn current_time() -> Duration {
    system_time_as_unix_time(SystemTime::now())
}

pub fn system_time_as_unix_time(time: SystemTime) -> Duration {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .expect("Time went backwards")
}

/// Safe way to truncate string.
pub fn truncate_string(input: String, max_len: usize) -> String {
    let mut graphemes = input.chars();
    let mut result = String::new();

    for _ in 0..max_len {
        if let Some(c) = graphemes.next() {
            result.push(c);
        } else {
            break;
        }
    }

    result
}

pub mod serde_duration_as_float_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let secs = duration.as_secs_f32();
        serializer.serialize_f32(secs)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f32::deserialize(deserializer)?;
        Ok(Duration::from_secs_f32(secs))
    }
}

pub mod serde_duration_as_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let secs = duration.as_secs();
        serializer.serialize_u64(secs)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

pub fn open_file_with_vim(path: &Path) -> io::Result<()> {
    let status = Command::new("vim").arg(path).status()?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(
            ErrorKind::Other,
            "Failed to open file with vim",
        ))
    }
}

pub trait MenuItem: Display {
    fn action(&self) -> Box<dyn FnMut() -> bool>;
}

/// Randomizing a vector.
/// Not importing rand cause im trying to keep dependency-count low.
pub fn randvec<T>(mut v: Vec<T>) -> Vec<T> {
    let veclen = v.len();

    for i in 0..veclen {
        let other_index = rand_int(veclen as u32);
        v.swap(i, other_index as usize);
    }
    v
}

pub fn view_cards_in_explorer() {
    open_folder_in_explorer(&get_cards_path()).unwrap()
}

fn open_folder_in_explorer(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer").arg(path).status()?;
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(path).status()?;
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(path).status()?;
    }

    Ok(())
}

/// will generate a number between 0 and 100 and check that it's below the given percentage.
/// so if you input '10', then ofc, 10% of the times it will return true as the number will be below 10
pub fn within_percentage(percentage: u32) -> bool {
    rand_int(100) < percentage
}

pub fn rand_int(max: u32) -> u32 {
    let time = current_time();
    (time.as_micros() ^ time.as_nanos() ^ time.as_millis()) as u32 % max
}

pub fn get_last_modified(path: PathBuf) -> Duration {
    let metadata = std::fs::metadata(path).unwrap();
    let modified_time = metadata.modified().unwrap();
    let secs = modified_time
        .duration_since(UNIX_EPOCH)
        .map(|s| s.as_secs())
        .unwrap();
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_within_percentage() {
        for _ in 0..20 {
            //std::thread::sleep(Duration::from_secs_f32(0.03258));
            dbg!(within_percentage(50));
        }
    }

    #[test]
    fn test_rand_vec() {
        let vec = vec![1, 2, 3, 4, 5, 6, 7];
        dbg!(randvec(vec));
    }

    #[test]
    fn foo() {
        let input_vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let randomized = randvec(input_vec);
        dbg!(randomized);
    }
}
