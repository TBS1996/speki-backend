use serde::{de, Deserialize, Serialize, Serializer};
use toml::Value;

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::fs::read_to_string;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use std::time::Duration;
use uuid::Uuid;

use std::sync::Arc;

use crate::categories::Category;
use crate::media::AudioSource;
use crate::ml::{log_transform, make_prediction};
use crate::{common::current_time, Id};

pub type RecallRate = f32;

#[derive(Default)]
pub struct CardCache(HashMap<Id, Arc<SavedCard>>);

impl CardCache {
    /// Checks that the card in the cache is up to date, and fixes it if it's not.
    /// There's three possibilities:
    ///     1. its up to date, no need to do anything.
    ///     2. It's outdated, we simply deserialize the card from the same path, so that its updated.
    ///     3. card isn't even found in the location, we search through all the cards to find it. Panicking if it's not found.
    fn maybe_update(&mut self, id: &Id) {
        let card_needs_update = match self.0.get(id) {
            Some(cached_card) => {
                let path = cached_card.as_path();
                if path.exists() {
                    // Get the file's last_modified time
                    let metadata = std::fs::metadata(path.as_path()).unwrap();
                    let last_modified_time = system_time_as_unix_time(metadata.modified().unwrap());

                    // Check if the file has been modified since we cached it
                    Some(last_modified_time > cached_card.last_modified)
                } else {
                    None
                }
            }
            None => None, // If card isn't in the cache, then we definitely need to update
        };

        match card_needs_update {
            Some(true) => {
                let path = self.0.get(id).unwrap().as_path();
                let updated_card = SavedCard::from_path(path.as_path());
                self.0.insert(*id, updated_card.into());
            }
            // if you find the card, and it's up to date, then no need to do anything.
            Some(false) => {}
            None => {
                // Read the card from the disk
                // expensive! it'll comb through all the cards linearly.
                let card = SavedCard::from_id(id).unwrap();
                self.0.insert(*id, card.into());
            }
        };
    }

    /// gets all the Ids (keys) sorted by recent modified
    pub fn all_ids(&self) -> Vec<Id> {
        let mut pairs: Vec<_> = self.0.iter().collect();
        pairs.sort_by_key(|&(_, v)| {
            if v.is_outdated() {
                get_last_modified(v.as_path())
            } else {
                v.last_modified().to_owned()
            }
        });
        pairs.reverse();
        pairs.into_iter().map(|(k, _)| k.to_owned()).collect()
    }

    pub fn exists(&self, id: &Id) -> bool {
        self.0.get(id).is_some()
    }

    pub fn insert(&mut self, card: SavedCard) {
        let id = card.id();
        self.0.insert(*id, card.into());
    }

    pub fn remove(&mut self, id: &Id) {
        self.0.remove(id);
    }

    pub fn dependencies(&mut self, id: &Id) -> BTreeSet<Id> {
        self.get_ref(id)
            .dependency_ids()
            .iter()
            .map(ToOwned::to_owned)
            .collect()
    }

    pub fn dependents(&mut self, id: &Id) -> BTreeSet<Id> {
        self.get_ref(id)
            .dependent_ids()
            .iter()
            .map(ToOwned::to_owned)
            .collect()
    }

    pub fn recursive_dependencies(&mut self, id: &Id) -> BTreeSet<Id> {
        let mut dependencies = BTreeSet::new();
        let mut stack = VecDeque::new();
        stack.push_back(*id);

        while let Some(card) = stack.pop_back() {
            if !dependencies.contains(&card) {
                dependencies.insert(card);

                let card_dependencies = self.dependencies(&card);

                for dependency in card_dependencies {
                    stack.push_back(dependency);
                }
            }
        }

        dependencies.remove(id);
        dependencies
    }

    pub fn recursive_dependents(&mut self, id: &Id) -> BTreeSet<Id> {
        let mut dependencies = BTreeSet::new();
        let mut stack = VecDeque::new();
        stack.push_back(*id);

        while let Some(card) = stack.pop_back() {
            if !dependencies.contains(&card) {
                dependencies.insert(card);

                let card_dependencies = self.dependents(&card);

                for dependency in card_dependencies {
                    stack.push_back(dependency);
                }
            }
        }

        dependencies.remove(id);
        dependencies
    }

    pub fn get_owned(&mut self, id: &Id) -> SavedCard {
        (*self.get_ref(id)).clone()
    }

    pub fn get_ref(&mut self, id: &Id) -> Arc<SavedCard> {
        self.maybe_update(id);
        self.0.get(id).unwrap().to_owned()
    }

    pub fn new() -> Self {
        let mut cache = Self::default();
        cache.cache_all();
        cache
    }

    pub fn refresh(&mut self) {
        *self = Self::new();
    }

    fn cache_all(&mut self) {
        let all_cards = SavedCard::load_all_cards();
        for card in all_cards {
            self.cache_one(card);
        }
    }

    pub fn cache_one(&mut self, card: SavedCard) {
        self.0.insert(card.card.meta.id, card.into());
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Clone, Debug)]
pub struct CardLocation {
    file_name: OsString,
    category: Category,
}

impl CardLocation {
    pub fn new(path: &Path) -> Self {
        let file_name = path.file_name().unwrap().to_owned();
        let category = Category::from_card_path(path);
        Self {
            file_name,
            category,
        }
    }

    fn as_path(&self) -> PathBuf {
        let mut path = self.category.as_path().join(self.file_name.clone());
        path.set_extension("toml");
        path
    }
}

impl std::fmt::Display for SavedCard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.front_text())
    }
}

impl From<SavedCard> for Card {
    fn from(value: SavedCard) -> Self {
        value.card
    }
}

pub enum ReviewType {
    Normal,
    Pending,
    Unfinished,
}

#[derive(Debug, Default)]
pub struct CardInfo {
    pub recall_rate: f32,
    pub strength: f32,
    pub stability: f32,
    pub resolved: bool,
    pub suspended: bool,
    pub finished: bool,
    pub change: f32,
}

impl CardInfo {
    fn new(card: &SavedCard, cache: &mut CardCache) -> Option<Self> {
        Self {
            recall_rate: card.recall_rate()?,
            strength: card.strength()?.as_secs_f32() / 86400.,
            stability: card.stability()?.as_secs_f32() / 86400.,
            change: card.card.history.expected_gain()?,
            resolved: card.is_resolved(cache),
            suspended: card.is_suspended(),
            finished: card.is_finished(),
        }
        .into()
    }
}

/// Represents a card that has been saved as a toml file, which is basically anywhere in the codebase
/// except for when youre constructing a new card.
/// Don't save this in containers or pass to functions, rather use the Id, and get new instances of SavedCard from the cache.
/// Also, every time you mutate it, call the persist() method.
#[derive(Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Debug)]
pub struct SavedCard {
    card: Card,
    location: CardLocation,
    last_modified: Duration,
}

impl SavedCard {
    pub fn set_priority(&mut self, priority: Priority) {
        self.card.meta.priority = priority;
        self.persist();
    }

    #[allow(dead_code)]
    pub fn priority(&self) -> &Priority {
        &self.card.meta.priority
    }

    pub fn expected_gain(&self) -> Option<f32> {
        self.card.history.expected_gain()
    }

    pub fn ml_expected_gain_debug(&self, time: Duration) -> Option<(Duration, Duration, Duration)> {
        let current_strength = self.strength_at_time(time)?;
        print!("A");

        let failed_review_card = {
            let mut card = self.clone();
            card.fake_new_review(Grade::Late, Duration::from_secs(5), time);
            card
        };

        let success_review_card = {
            let mut card = self.clone();
            card.fake_new_review(Grade::Some, Duration::from_secs(5), time);
            card
        };

        let time = time + Duration::from_secs(30);

        let fail_strength = (failed_review_card.strength_at_time(time))?;
        print!("B");
        let win_strength = success_review_card.strength_at_time(time)?;
        print!("C");

        Some((current_strength, win_strength, fail_strength))
    }

    pub fn pure_ml_expected_gain(
        card: SavedCard,
        win_card: SavedCard,
        fail_card: SavedCard,
        time: Duration,
    ) -> f32 {
        let recall_rate = card.ml_recall_rate_at_time(time).unwrap();
        let current_strength = card.strength_at_time(time).unwrap();
        let win_strength = win_card.strength_at_time(time).unwrap();
        let fail_strength = fail_card.strength_at_time(time).unwrap();

        Reviews::pure_expected_gain(recall_rate, current_strength, fail_strength, win_strength)
    }

    pub fn ml_expected_gain(&self, time: Duration, og_time: Duration) -> Option<f32> {
        let recall_rate = self.ml_recall_rate_at_time(time)?;

        // assert!((time.as_secs()) >= (current_time().as_secs()));

        let current_strength = self.strength_at_time(time)?;
        print!("A");

        let failed_review_card = {
            let mut card = self.clone();
            card.fake_new_review(Grade::Late, Duration::from_secs(5), og_time);
            card
        };

        let success_review_card = {
            let mut card = self.clone();
            card.fake_new_review(Grade::Some, Duration::from_secs(5), og_time);
            card
        };

        let time = time + Duration::from_secs(30);

        let fail_strength = (failed_review_card.strength_at_time(time))?;
        print!("B");
        let win_strength = success_review_card.strength_at_time(time)?;
        print!("C");

        Some(Reviews::pure_expected_gain(
            recall_rate,
            current_strength,
            fail_strength,
            win_strength,
        ))
    }

    pub fn get_info(&self, cache: &mut CardCache) -> Option<CardInfo> {
        CardInfo::new(self, cache)
    }

    pub fn reviews(&self) -> &Vec<Review> {
        &self.card.history.0
    }

    pub fn raw_reviews(&self) -> &Reviews {
        &self.card.history
    }

    pub fn new(card: Card, location: CardLocation, last_modified: Duration) -> Self {
        Self {
            card,
            location,
            last_modified,
        }
    }

    pub fn get_unfinished_dependent_qty(&self, cache: &mut CardCache) -> usize {
        let dependents = cache.recursive_dependents(self.id());
        let mut unfinished = 0;
        for dependent in dependents {
            unfinished += cache.get_ref(&dependent).is_finished() as usize;
        }
        unfinished
    }

    pub fn last_modified(&self) -> &Duration {
        &self.last_modified
    }

    pub fn category(&self) -> &Category {
        &self.location.category
    }

    pub fn front_text(&self) -> &str {
        &self.card.front.text
    }

    #[allow(dead_code)]
    pub fn is_pending(&self) -> bool {
        self.card.history.is_empty()
    }

    pub fn is_suspended(&self) -> bool {
        self.card.meta.suspended.is_suspended()
    }

    pub fn is_finished(&self) -> bool {
        self.card.meta.finished
    }

    pub fn set_front_text(&mut self, text: &str) {
        self.card.front.text = text.to_string();
        self.persist();
    }

    pub fn set_back_text(&mut self, text: &str) {
        self.card.back.text = text.to_string();
        self.persist();
    }

    /// If you accidentally made two identical cards, this will merge the stuff
    /// like depencies, reviews etc.. and delete the other card
    pub fn merge_with(&mut self, cache: &mut CardCache, other_card: Id) {
        let other_card = cache.get_owned(&other_card);

        // we cant just add them directly, we have to check for loops in the DAG
        for dependency in other_card.dependency_ids() {
            self.set_dependency(dependency, cache);
        }
        for dependent in other_card.dependent_ids() {
            self.set_dependent(dependent, cache);
        }

        self.card
            .history
            .0
            .extend(other_card.reviews().iter().map(|review| review.to_owned()));
        self.card
            .meta
            .tags
            .extend(other_card.card.meta.tags.iter().map(|s| s.to_owned()));
        other_card.delete(cache);
        self.persist();
    }

    pub fn ml_recall_rate_at_time(&self, current_time: Duration) -> Option<f32> {
        let revlen = self.reviews().len();

        let input_values =
            MlArgs::get_prediction_data_at_time(current_time, self.raw_reviews().to_owned());

        let model_name = match revlen {
            0 => return None,
            1 => "2data",
            2 => "3data",
            _ => "4data",
        };
        make_prediction(model_name, input_values).unwrap().into()
    }

    pub fn stability(&self) -> Option<Duration> {
        self.card.history.stability()
    }

    pub fn numerical_integration(
        &self,
        from_time: Duration,
        right_bound: Duration,
        steps: usize,
    ) -> Duration {
        let dt = right_bound.div_f32(steps as f32);
        let mut sum = Duration::default();

        for i in 0..steps {
            let t1 = dt.mul_f32(i as f32) + from_time;
            let t2 = dt.mul_f32((i + 1) as f32) + from_time;
            let avg_rate = (self.ml_recall_rate_at_time(t1).unwrap()
                + self.ml_recall_rate_at_time(t2).unwrap())
                / 2.0;

            sum += dt.mul_f32(avg_rate);
        }
        sum
    }

    pub fn recall_rate(&self) -> Option<f32> {
        self.card.history.recall_rate()
    }

    pub fn trad_recall_rate_at_time(&self, time: Duration) -> Option<f32> {
        let stability = self.stability()?;
        let time_passed = time.checked_sub(self.reviews().last().unwrap().timestamp)?;
        Reviews::calculate_recall_rate(&time_passed, &stability).into()
    }

    pub fn xstrength(&self) -> Option<Duration> {
        if self.reviews().is_empty() {
            return None;
        }
        self.numerical_integration(current_time(), Duration::from_secs(86400 * 100), 100)
            .into()
    }

    pub fn strength_at_time(&self, time: Duration) -> Option<Duration> {
        if self.reviews().is_empty() {
            return None;
        }
        self.numerical_integration(time, Duration::from_secs(86400 * 100), 10)
            .into()
    }

    pub fn strength(&self) -> Option<Duration> {
        self.card.history.strength()
    }

    pub fn time_since_last_review(&self) -> Option<Duration> {
        self.card.time_passed_since_last_review()
    }

    pub fn back_text(&self) -> &str {
        &self.card.back.text
    }

    pub fn contains_tag(&self, tag: &str) -> bool {
        self.card.meta.tags.contains(tag)
    }

    pub fn id(&self) -> &Id {
        &self.card.meta.id
    }

    pub fn dependent_ids(&self) -> &BTreeSet<Id> {
        &self.card.meta.dependents
    }

    pub fn dependency_ids(&self) -> &BTreeSet<Id> {
        &self.card.meta.dependencies
    }

    pub fn set_suspended(&mut self, suspended: IsSuspended) {
        self.card.meta.suspended = suspended;
        self.persist();
    }

    pub fn set_finished(&mut self, finished: bool) {
        self.card.meta.finished = finished;
        self.persist();
    }

    pub fn insert_tag(&mut self, tag: String) {
        self.card.meta.tags.insert(tag);
        self.persist();
    }

    pub fn check_cycle(cache: &mut CardCache, id: &Id, is_dependent: bool) -> Option<Vec<Id>> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        Self::has_cycle_helper(cache, id, is_dependent, &mut visited, &mut path)
    }

    fn has_cycle_helper(
        cache: &mut CardCache,
        id: &Id,
        is_dependent: bool,
        visited: &mut HashSet<Id>,
        path: &mut Vec<Id>,
    ) -> Option<Vec<Id>> {
        if path.contains(id) {
            return Some(path.clone());
        }

        path.push(*id);

        let dependencies = if is_dependent {
            cache.dependencies(id)
        } else {
            cache.dependents(id)
        };

        for dependency in dependencies {
            if !visited.contains(&dependency) {
                if let Some(cycle) =
                    Self::has_cycle_helper(cache, &dependency, is_dependent, visited, path)
                {
                    return Some(cycle);
                }
            }
        }

        visited.insert(*id);
        path.pop();

        None
    }

    pub fn set_dependency(&mut self, id: &Id, cache: &mut CardCache) -> Option<String> {
        self.card.meta.dependencies.insert(*id);
        self.persist();

        let mut other_card = cache.get_owned(id);
        other_card.card.meta.dependents.insert(self.card.meta.id);
        other_card.persist();
        if let Some(mut path) = Self::check_cycle(cache, id, false) {
            path.reverse();

            while path[0] != *self.id() {
                path.rotate_right(1);
            }
            path.push(*self.id());
            let veclen = path.len();

            let textvec = path
                .into_iter()
                .map(|id| cache.get_ref(&id).front_text().to_string())
                .collect::<Vec<_>>();

            let mut s = String::new();

            for i in 0..veclen {
                s.push_str(&format!("'{}'", &textvec[i]));

                if i == 0 {
                    s.push_str(" depends on ")
                } else if i < (veclen - 2) {
                    s.push_str(" which depends on ")
                } else if i != (veclen - 1) {
                    s.push_str(" which creates a cycle, as it depends on ")
                }
            }

            self.card.meta.dependencies.remove(id);
            self.persist();

            other_card.card.meta.dependents.remove(self.id());
            other_card.persist();
            return Some(s);
        }
        None
    }

    pub fn set_dependent(&mut self, id: &Id, cache: &mut CardCache) -> Option<String> {
        self.card.meta.dependents.insert(*id);
        self.persist();

        let mut other_card = cache.get_owned(id);
        other_card.card.meta.dependencies.insert(self.card.meta.id);
        other_card.persist();

        if let Some(mut path) = Self::check_cycle(cache, id, true) {
            while path[0] != *self.id() {
                path.rotate_right(1);
            }
            path.push(*self.id());
            let veclen = path.len();

            let textvec = path
                .into_iter()
                .map(|id| cache.get_ref(&id).front_text().to_string())
                .collect::<Vec<_>>();

            let mut s = String::new();

            for i in 0..veclen {
                s.push_str(&format!("'{}'", &textvec[i]));

                if i == 0 {
                    s.push_str(" depends on ")
                } else if i < (veclen - 2) {
                    s.push_str(" which depends on ")
                } else if i != (veclen - 1) {
                    s.push_str(" which creates a cycle, as it depends on ")
                }
            }

            self.card.meta.dependents.remove(id);
            self.persist();

            other_card.card.meta.dependencies.remove(self.id());
            other_card.persist();

            return Some(s);
        }
        None
    }

    /// a = span means foo
    /// b = change span desc by..
    /// inserted: c = what is a span desc?

    pub fn _insert_dependency_raw(
        dependent_id: &Id,
        dependency_id: &Id,
        insertion_id: &Id,
        cache: &mut CardCache,
    ) {
        let mut dependent = Self::from_id(dependent_id).unwrap();
        let _insertion = Self::from_id(insertion_id).unwrap();

        dependent.remove_dependency(dependency_id, cache);
        //dependent.set_dependency(insertion_id);
        //insertion.set_dependency(dependency_id);
    }

    pub fn remove_dependency(&mut self, id: &Id, _cache: &mut CardCache) {
        self.card.meta.dependencies.remove(id);
        self.persist();

        if let Some(mut other_card) = Self::from_id(id) {
            other_card.card.meta.dependents.remove(self.id());
            other_card.persist();
        }
    }

    pub fn remove_dependent(&mut self, id: &Id, _cache: &mut CardCache) {
        self.card.meta.dependencies.remove(id);
        self.persist();

        if let Some(mut other_card) = Self::from_id(id) {
            other_card.card.meta.dependents.remove(self.id());
            other_card.persist();
        }
    }

    pub fn as_path(&self) -> PathBuf {
        self.location.as_path()
    }

    pub fn pending_filter(card: &Id, cache: &mut CardCache) -> bool {
        let card = cache.get_ref(card);
        card.card.history.is_empty()
            && !card.is_suspended()
            && card.is_finished()
            && card.is_confidently_resolved(cache)
    }

    pub fn random_filter(card: &Id, cache: &mut CardCache) -> bool {
        let card = cache.get_ref(card);
        card.is_finished() && !card.is_suspended() && card.is_resolved(cache)
    }

    pub fn unfinished_filter(card: &Id, cache: &mut CardCache) -> bool {
        let card = cache.get_ref(card);
        !card.is_finished() && !card.is_suspended() && card.is_resolved(cache)
    }

    pub fn review_filter(card: &Id, cache: &mut CardCache) -> bool {
        let card = cache.get_ref(card);
        let (Some(stability), Some(last_review_time)) =
            (card.stability(), card.card.history.time_since_last_review())
        else {
            return false;
        };

        card.is_finished()
            && !card.is_suspended()
            && stability < last_review_time
            && card.is_confidently_resolved(cache)
    }

    /// Checks if corresponding file has been modified after this type got deserialized from the file.
    pub fn is_outdated(&self) -> bool {
        let file_last_modified = {
            let path = self.as_path();
            let system_time = std::fs::metadata(path).unwrap().modified().unwrap();
            system_time_as_unix_time(system_time)
        };

        let in_memory_last_modified = self.last_modified;

        match in_memory_last_modified.cmp(&file_last_modified) {
            Ordering::Less => true,
            Ordering::Equal => false,
            Ordering::Greater => panic!("Card in-memory shouldn't have a last_modified more recent than its corresponding file"),
        }
    }

    pub fn is_resolved(&self, cache: &mut CardCache) -> bool {
        cache
            .recursive_dependencies(self.id())
            .iter()
            .all(|id| cache.get_ref(id).is_finished())
    }

    /// Checks that its dependencies are not only marked finished, but they're also strong memories.
    pub fn is_confidently_resolved(&self, cache: &mut CardCache) -> bool {
        let min_stability = Duration::from_secs(86400 * 2);
        let min_recall: f32 = 0.95;
        let dependencies = cache.recursive_dependencies(self.id());
        //  dbg!("##############", self.front_text());
        for _dep in dependencies {
            //   dbgshit.push(format!("{}: {}\t", dep.front_text(), dep.is_finished()));
        }
        //   dbg!(dbgshit);

        let x = cache.recursive_dependencies(self.id()).iter().all(|id| {
            let card = cache.get_ref(id);
            let (Some(stability), Some(recall)) =
                (card.stability(), card.card.history.recall_rate())
            else {
                return false;
            };

            card.card.meta.finished && stability > min_stability && recall > min_recall
        });
        //    dbg!("$$", &x, "$$");
        x
    }

    /// Moves card by deleting it and then creating it again in a new location
    /// warning: will refresh file name
    pub fn move_card(self, destination: &Category, cache: &mut CardCache) -> Self {
        if self.location.category == *destination {
            return self;
        }
        assert!(self.as_path().exists());
        std::fs::remove_file(self.as_path()).unwrap();
        assert!(!self.as_path().exists());
        self.into_card().save_new_card(destination, cache)
    }

    pub fn get_review_type(&self) -> ReviewType {
        match (self.card.history.is_empty(), self.is_finished()) {
            (_, false) => ReviewType::Unfinished,
            (false, true) => ReviewType::Normal,
            (true, true) => ReviewType::Pending,
        }
    }

    pub fn delete(self, cache: &mut CardCache) {
        let path = self.as_path();
        std::fs::remove_file(&path).unwrap();
        assert!(!path.exists());

        let self_id = self.card.meta.id;

        for dependency in self.card.meta.dependencies {
            let mut dependency = cache.get_owned(&dependency);
            dependency.card.meta.dependents.remove(&self_id);
            dependency.persist();
        }

        for dependent in self.card.meta.dependents {
            let mut dependent = cache.get_owned(&dependent);
            dependent.card.meta.dependencies.remove(&self_id);
            dependent.persist();
        }
        cache.remove(&self_id);
    }

    pub fn get_cards_from_category_recursively(category: &Category) -> HashSet<SavedCard> {
        let mut cards = HashSet::new();
        let cats = category.get_following_categories();
        for cat in cats {
            cards.extend(cat.get_containing_cards());
        }
        cards
    }

    pub fn search_in_cards<'a>(
        input: &'a str,
        cards: &'a HashSet<SavedCard>,
        excluded_cards: &'a HashSet<Id>,
    ) -> Vec<&'a SavedCard> {
        cards
            .iter()
            .filter(|card| {
                (card
                    .card
                    .front
                    .text
                    .to_ascii_lowercase()
                    .contains(&input.to_ascii_lowercase())
                    || card
                        .card
                        .back
                        .text
                        .to_ascii_lowercase()
                        .contains(&input.to_ascii_lowercase()))
                    && !excluded_cards.contains(card.id())
            })
            .collect()
    }

    // expensive function!
    pub fn from_id(id: &Id) -> Option<Self> {
        Self::load_all_cards()
            .into_iter()
            .find(|card| &card.card.meta.id == id)
    }

    pub fn load_all_cards() -> HashSet<SavedCard> {
        Self::get_cards_from_category_recursively(&Category::root())
    }

    pub fn edit_with_vim(&self) -> Self {
        let path = self.as_path();
        open_file_with_vim(path.as_path()).unwrap();
        Self::from_path(path.as_path())
    }

    pub fn from_path(path: &Path) -> Self {
        let content = read_to_string(path).expect("Could not read the TOML file");
        let card: Card = toml::from_str(&content).unwrap();
        let location = CardLocation::new(path);

        let last_modified = {
            let system_time = std::fs::metadata(path).unwrap().modified().unwrap();
            system_time_as_unix_time(system_time)
        };

        Self {
            card,
            location,
            last_modified,
        }
    }

    pub fn into_card(self) -> Card {
        self.card
    }

    pub fn card_as_ref(&self) -> &Card {
        &self.card
    }

    // Call this function every time SavedCard is mutated.
    fn persist(&mut self) {
        if self.is_outdated() {
            // When you persist, the last_modified in the card should match the ones from the file.
            // This shouldn't be possible, as this function mutates itself to get a fresh copy, so
            // i'll panic here to alert me of the logic bug.
            let _x = format!("{:?}", self);
            // panic!("{}", x);
        }

        let path = self.as_path();
        if !path.exists() {
            let msg = format!("following path doesn't really exist: {}", path.display());
            panic!("{msg}");
        }

        let toml = toml::to_string(self.card_as_ref()).unwrap();

        std::fs::write(&path, toml).unwrap();
        *self = SavedCard::from_path(path.as_path())
    }

    pub fn new_review(&mut self, grade: Grade, time: Duration) {
        let review = Review::new(grade, time);
        self.card.history.add_review(review);
        self.persist();
    }

    pub fn fake_new_review(&mut self, grade: Grade, time: Duration, at_time: Duration) {
        let review = Review {
            timestamp: at_time,
            grade,
            time_spent: time,
        };
        self.card.history.add_review(review);
    }
}

#[derive(Ord, PartialOrd, Eq, Hash, PartialEq, Deserialize, Serialize, Debug, Default, Clone)]
pub struct Card {
    pub front: Side,
    pub back: Side,
    pub meta: Meta,
    #[serde(default, skip_serializing_if = "Reviews::is_empty")]
    pub history: Reviews,
}

// public
impl Card {
    pub fn new(front: Side, back: Side, meta: Meta) -> Self {
        Card {
            front,
            back,
            meta,
            history: Reviews::default(),
        }
    }

    pub fn import_cards(filename: &Path) -> Option<Vec<Self>> {
        let mut lines = std::io::BufReader::new(std::fs::File::open(filename).ok()?).lines();
        let mut cards = vec![];

        while let Some(Ok(question)) = lines.next() {
            if let Some(Ok(answer)) = lines.next() {
                cards.push(Self::new_simple(question, answer));
            }
        }
        cards.into()
    }

    pub fn new_simple(front: String, back: String) -> Self {
        Card {
            front: Side {
                text: front,
                ..Default::default()
            },
            back: Side {
                text: back,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn _review_priority(&self) -> Option<f32> {
        let recall_rate = self.history.recall_rate()?;
        let priority = &self.meta.priority;
        Some((recall_rate - 1.0) * priority.as_float())
    }

    pub fn save_new_card(self, category: &Category, cache: &mut CardCache) -> SavedCard {
        let toml = toml::to_string(&self).unwrap();
        std::fs::create_dir_all(category.as_path()).unwrap();
        let max_char_len = 40;
        let front_text = self
            .front
            .text
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || c.is_ascii_whitespace())
            .collect::<String>()
            .replace(' ', "_");
        let mut file_name = PathBuf::from(truncate_string(front_text, max_char_len));

        if file_name.exists() {
            file_name = PathBuf::from(self.meta.id.to_string());
        }

        let mut path = category.as_path().join(file_name).with_extension("toml");

        if path.exists() {
            file_name = PathBuf::from(self.meta.id.to_string());
            path = category.as_path().join(file_name).with_extension("toml");
        }

        std::fs::write(&path, toml).unwrap();

        let full_card = SavedCard::from_path(path.as_path());
        cache.insert(full_card.clone());
        full_card
    }

    fn time_passed_since_last_review(&self) -> Option<Duration> {
        if current_time() < self.history.0.last()?.timestamp {
            return Duration::default().into();
        }

        Some(current_time() - self.history.0.last()?.timestamp)
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Serialize, Debug, Default, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Grade {
    // No recall, not even when you saw the answer.
    #[default]
    None,
    // No recall, but you remember the answer when you read it.
    Late,
    // Struggled but you got the answer right or somewhat right.
    Some,
    // No hesitation, perfect recall.
    Perfect,
}

impl Grade {
    pub fn get_factor(&self) -> f32 {
        match self {
            Grade::None => 0.1,
            Grade::Late => 0.25,
            Grade::Some => 2.,
            Grade::Perfect => 3.,
        }
        //factor * Self::randomize_factor()
    }

    pub fn ml_normalized(&self) -> String {
        match self {
            Grade::None => 0.,
            Grade::Late => 0.2,
            Grade::Some => 0.8,
            Grade::Perfect => 1.,
        }
        .to_string()
    }

    pub fn ml_binary(&self) -> String {
        (self.is_succesful_recall() as u32).to_string()
    }

    pub fn is_succesful_recall(&self) -> bool {
        match self {
            Grade::None | Grade::Late => false,
            Grade::Some | Grade::Perfect => true,
        }
    }
}

impl std::str::FromStr for Grade {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1" => Ok(Self::None),
            "2" => Ok(Self::Late),
            "3" => Ok(Self::Some),
            "4" => Ok(Self::Perfect),
            _ => Err(()),
        }
    }
}

use crate::common::{
    get_last_modified, open_file_with_vim, rand_int, serde_duration_as_float_secs,
    serde_duration_as_secs, system_time_as_unix_time, truncate_string, within_percentage,
};

#[derive(Debug, Clone)]
pub struct MlArgs {
    first_grade: Grade,
    next_ones: Vec<MlPoint>,
}

#[derive(Clone, Debug)]
struct MlPoint {
    grade: Grade,
    time_passed: Duration,
}

impl MlArgs {
    pub fn new_truncated(reviews: Reviews, len: usize) -> Option<Self> {
        if reviews.is_empty() {
            return None;
        }
        let mut obj = MlArgs::new(&reviews);

        if obj.next_ones.len() < (len - 1) {
            return None;
        }
        obj.next_ones.truncate(len - 1);

        Some(obj)
    }

    pub fn tail_prediction_data(reviews: Reviews, len: usize) -> Vec<String> {
        Self::tail_prediction_data_at_time(current_time(), reviews, len)
    }

    pub fn get_prediction_data(reviews: Reviews) -> Vec<String> {
        Self::get_prediction_data_at_time(current_time(), reviews)
    }

    fn get_prediction_data_at_time(time: Duration, reviews: Reviews) -> Vec<String> {
        let revlen = reviews.len();

        match revlen {
            0 => panic!(),
            1 => Self::full_prediction_data_at_time(time, reviews),
            2 => Self::full_prediction_data_at_time(time, reviews),
            _ => Self::tail_prediction_data_at_time(time, reviews, 3),
        }
    }

    fn full_prediction_data_at_time(time: Duration, reviews: Reviews) -> Vec<String> {
        let revlen = reviews.len();
        Self::tail_prediction_data_at_time(time, reviews, revlen)
    }

    // for unit testing
    fn tail_prediction_data_at_time(
        time: Duration,
        mut reviews: Reviews,
        len: usize,
    ) -> Vec<String> {
        let revlen = reviews.len();
        if len < revlen {
            let raw = reviews.into_inner();
            let stripped = &raw[(revlen - len)..];
            reviews = Reviews::from_raw(stripped.to_owned());
        }

        let ml = MlArgs::new(&reviews);
        let mut output = ml.as_training_data();

        let prev_timestamp = reviews.0.last().unwrap().timestamp;

        let Some(time_passed) = time.checked_sub(prev_timestamp) else {
            dbg!(time, prev_timestamp, len, reviews);
            panic!()
        };

        let time_passed = log_transform(time_passed).to_string();
        output.push(time_passed);
        output
    }

    pub fn as_training_data(&self) -> Vec<String> {
        let mut output = vec![];

        let grade = (self.first_grade.is_succesful_recall() as u32).to_string();

        output.push(grade);
        //let mut pred_grade = Grade::None;

        for x in &self.next_ones {
            // let grade = x.grade.ml_normalized();
            let grade = x.grade.ml_binary();
            let time_passed = log_transform(x.time_passed).to_string();
            output.push(time_passed);
            output.push(grade);
            //pred_grade = x.grade.clone();
        }
        //output.pop();
        //output.push((pred_grade.is_succesful_recall() as u32).to_string());

        output
    }

    pub fn get_windowed(&self, len: usize) -> Vec<Self> {
        let mut output: Vec<Self> = vec![];

        let windows = self.next_ones.windows(len);

        for slice in windows {
            let grade = slice[0].grade.clone();
            let new = Self {
                first_grade: grade,
                next_ones: slice[1..].into(),
            };
            output.push(new);
        }

        output
    }

    pub fn new(reviews: &Reviews) -> Self {
        if reviews.is_empty() {
            panic!("ohno");
        }

        let mut slice_iter = reviews.0.iter();
        let first_review = slice_iter.next().unwrap();

        let mut myself = Self {
            first_grade: first_review.grade.clone(),
            next_ones: vec![],
        };

        let mut previous_timestamp = first_review.timestamp;

        for review in slice_iter {
            let time_passed = review.timestamp - previous_timestamp;
            let point = MlPoint {
                grade: review.grade.clone(),
                time_passed,
            };

            previous_timestamp = review.timestamp;
            myself.next_ones.push(point);
        }

        myself
    }
}

#[derive(Ord, PartialOrd, Eq, Hash, PartialEq, Debug, Default, Clone)]
pub struct Reviews(pub Vec<Review>);

impl Reviews {
    // Basically, you get
    //pub fn get_all_training_args(&self) -> Vec<String> {}

    /*
    pub fn get_rolling_args(&self, len: usize, rolling: bool) -> Vec<Vec<String>> {
        let data = self.get_all_training_args();
        let window = data.windows(len).into_iter();
        let mut output = vec![];

        for slice in window {
            let mut inner_output = vec![];
        }
        todo!()
    }
    */

    pub fn strength(&self) -> Option<Duration> {
        let days_passed = self.time_since_last_review()?;
        let stability = self.stability()?;
        let strength = calculate_memory_strength(0.9, days_passed, stability);
        //dbg!(days_passed.as_secs_f32() / 86400., stability.as_secs_f32() / 86400., strength);
        Duration::from_secs_f32(strength * 86400.).into()
    }

    pub fn next_strength(&self, grade: Grade) -> Duration {
        let mut myself = self.clone();
        let new_review = Review::new(grade, Duration::default());
        myself.add_review(new_review);
        myself.strength().unwrap_or_default()
    }

    // Expected gain in memory strength after a review.
    pub fn expected_gain(&self) -> Option<f32> {
        let recall_rate = self.recall_rate()?;

        let current_strength = self.strength()?;

        // The estimated strength if you fail at this point in time.
        let fail_strength = self.next_strength(Grade::Late);

        // The estimated strength if you succeed at this point in time.
        let win_strength = self.next_strength(Grade::Some);

        Some(Self::pure_expected_gain(
            recall_rate,
            current_strength,
            fail_strength,
            win_strength,
        ))
    }

    /// trying to extract more pure functions
    pub fn pure_expected_gain(
        recall_rate: f32,
        current_strength: Duration,
        fail_strength: Duration,
        win_strength: Duration,
    ) -> f32 {
        let current_strength = current_strength.as_secs_f32() / 86400.;
        let winstrength = win_strength.as_secs_f32() / 86400.;
        let failstrength = fail_strength.as_secs_f32() / 86400.;

        let expected_win = (winstrength) * recall_rate;
        let expected_loss = (failstrength) * (1. - recall_rate);

        let expected_strength = expected_win + expected_loss;

        expected_strength - current_strength
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn into_inner(self) -> Vec<Review> {
        self.0
    }

    pub fn from_raw(reviews: Vec<Review>) -> Self {
        Self(reviews)
    }

    pub fn add_review(&mut self, review: Review) {
        self.0.push(review);
    }

    pub fn new_stability(
        grade: &Grade,
        time_passed: Option<Duration>,
        current_stability: Duration,
    ) -> Duration {
        let grade_factor = grade.get_factor();
        let time_passed = time_passed.unwrap_or(Duration::from_secs(86400));

        if grade_factor < 1.0 {
            // the grade is wrong
            time_passed.min(current_stability).mul_f32(grade_factor)
        } else {
            // the grade is correct
            let alternative_stability = time_passed.mul_f32(grade_factor);
            if alternative_stability > current_stability {
                alternative_stability
            } else {
                let interpolation_ratio =
                    time_passed.as_secs_f32() / current_stability.as_secs_f32() * grade_factor;
                current_stability
                    + Duration::from_secs_f32(current_stability.as_secs_f32() * interpolation_ratio)
            }
        }
    }

    pub fn stability(&self) -> Option<Duration> {
        let reviews = &self.0;
        if reviews.is_empty() {
            return None;
        }

        let mut stability =
            Self::new_stability(&reviews[0].grade, None, Duration::from_secs(86400));
        let mut prev_timestamp = reviews[0].timestamp;

        for review in &reviews[1..] {
            if prev_timestamp > review.timestamp {
                return None;
            }
            let time_passed = review.timestamp - prev_timestamp; // Calculate the time passed since the previous review
            stability = Self::new_stability(&review.grade, Some(time_passed), stability);
            prev_timestamp = review.timestamp; // Update the timestamp for the next iteration
        }

        Some(stability)
    }

    pub fn recall_rate(&self) -> Option<RecallRate> {
        let days_passed = self.time_since_last_review()?;
        let stability = self.stability()?;
        Some(Self::calculate_recall_rate(&days_passed, &stability))
    }

    pub fn calculate_recall_rate(days_passed: &Duration, stability: &Duration) -> RecallRate {
        let base: f32 = 0.9;
        let ratio = days_passed.as_secs_f32() / stability.as_secs_f32();
        (base.ln() * ratio).exp()
    }

    pub fn time_since_last_review(&self) -> Option<Duration> {
        self.0.last().map(Review::time_passed)
    }
}

impl Serialize for Reviews {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Reviews {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut reviews = Vec::<Review>::deserialize(deserializer)?;
        reviews.sort_by_key(|review| review.timestamp);
        Ok(Reviews(reviews))
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Clone, Serialize, Debug, Default)]
pub struct Review {
    // When (unix time) did the review take place?
    #[serde(with = "serde_duration_as_secs")]
    pub timestamp: Duration,
    // Recall grade.
    pub grade: Grade,
    // How long you spent before attempting recall.
    #[serde(with = "serde_duration_as_float_secs")]
    pub time_spent: Duration,
}

impl Review {
    fn new(grade: Grade, time_spent: Duration) -> Self {
        Self {
            timestamp: current_time(),
            grade,
            time_spent,
        }
    }

    fn time_passed(&self) -> Duration {
        let unix = self.timestamp;
        let current_unix = current_time();
        current_unix.checked_sub(unix).unwrap_or_default()
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Clone, Serialize, Debug, Default)]
pub struct Side {
    pub text: String,
    #[serde(flatten)]
    pub audio: AudioSource,
    //#[serde(deserialize_with = "deserialize_image_path")]
    //pub image: ImagePath,
}

use serde::de::Deserializer;

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Clone)]
pub enum IsSuspended {
    False,
    True,
    // Card is temporarily suspended, until contained unix time has passed.
    TrueUntil(Duration),
}

impl From<bool> for IsSuspended {
    fn from(value: bool) -> Self {
        match value {
            true => Self::True,
            false => Self::False,
        }
    }
}

impl Default for IsSuspended {
    fn default() -> Self {
        Self::False
    }
}

impl IsSuspended {
    fn verify_time(self) -> Self {
        if let Self::TrueUntil(dur) = self {
            if dur < current_time() {
                return Self::False;
            }
        }
        self
    }

    pub fn is_suspended(&self) -> bool {
        !matches!(self, IsSuspended::False)
    }
}

impl Serialize for IsSuspended {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        match self.clone().verify_time() {
            IsSuspended::False => serializer.serialize_bool(false),
            IsSuspended::True => serializer.serialize_bool(true),
            IsSuspended::TrueUntil(duration) => serializer.serialize_u64(duration.as_secs()),
        }
    }
}

impl<'de> Deserialize<'de> for IsSuspended {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Value = Deserialize::deserialize(deserializer)?;

        match value {
            Value::Boolean(b) => Ok(b.into()),
            Value::Integer(i) => {
                if let Ok(secs) = std::convert::TryInto::<u64>::try_into(i) {
                    Ok(IsSuspended::TrueUntil(Duration::from_secs(secs)).verify_time())
                } else {
                    Err(de::Error::custom("Invalid duration format"))
                }
            }

            _ => Err(serde::de::Error::custom("Invalid value for IsDisabled")),
        }
    }
}

fn default_finished() -> bool {
    true
}

/// How important a given card is, where 0 is the least important, 100 is most important.
#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Clone)]
pub struct Priority(u32);

impl Priority {
    pub fn as_float(&self) -> f32 {
        self.to_owned().into()
    }

    pub fn is_default(&self) -> bool {
        Self::default() == *self
    }
}

impl TryFrom<char> for Priority {
    type Error = ();

    fn try_from(value: char) -> Result<Self, Self::Error> {
        let pri = match value {
            '1' => 16,
            '2' => 33,
            '3' => 66,
            '4' => 83,
            _ => return Err(()),
        };
        Ok(Self(pri))
    }
}

impl From<u32> for Priority {
    fn from(value: u32) -> Self {
        Self(value.clamp(0, 100))
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self(50)
    }
}

impl From<Priority> for f32 {
    fn from(value: Priority) -> Self {
        value.0 as f32 / 100.
    }
}

impl Serialize for Priority {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Priority {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        if value > 100 {
            Err(serde::de::Error::custom("Invalid priority value"))
        } else {
            Ok(Priority(value))
        }
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Serialize, Debug, Clone)]
pub struct Meta {
    pub id: Id,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub dependencies: BTreeSet<Id>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub dependents: BTreeSet<Id>,
    // this card is needed to understand the current card. useful so that you won't need to write a bunch of boilerplate context
    #[serde(default)]
    pub context: Option<Id>,
    #[serde(default)]
    pub suspended: IsSuspended,
    #[serde(default = "default_finished")]
    pub finished: bool,
    #[serde(default, skip_serializing_if = "Priority::is_default")]
    pub priority: Priority,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub tags: BTreeSet<String>,
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            dependencies: BTreeSet::new(),
            dependents: BTreeSet::new(),
            suspended: IsSuspended::False,
            context: None,
            finished: default_finished(),
            priority: Priority::default(),
            tags: BTreeSet::new(),
        }
    }
}

pub fn calculate_memory_strength(base: f64, time_passed: Duration, stability: Duration) -> f32 {
    let t = time_passed.as_secs_f64() / 86400.;
    let base = base.ln();
    let value = -1.0 / base + (1.0 - f64::exp(t * base)) / base;
    value as f32 * stability.as_secs_f32() / 86400.
}

#[allow(dead_code)]
pub fn generate_random_reviews(qty: usize) -> Reviews {
    let mut reviews = Reviews::default();
    let mut prev_time = current_time();
    reviews.add_review(Review {
        timestamp: prev_time,
        grade: Grade::Late,
        time_spent: Duration::default(),
    });

    for _ in 0..(qty - 1) {
        let stability = reviews.stability().unwrap();
        let new = stability.mul_f32((rand_int(100) as f32 / 100.) + 1.);
        let added_time = stability + new;
        let recall_rate = Reviews::calculate_recall_rate(&added_time, &stability);
        let grade = if within_percentage((recall_rate * 100.) as u32) {
            Grade::Some
        } else {
            Grade::Late
        };

        let new_time = prev_time + added_time;
        reviews.add_review(Review {
            timestamp: new_time,
            grade,
            time_spent: Duration::default(),
        });
        prev_time = new_time;
    }
    reviews
}

fn duration_from_days(days: f32) -> Duration {
    Duration::from_secs_f32(days * 86400.)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_pure_expected_gain() {
        let recall_rate = 0.1;
        let current_strength = duration_from_days(10.);
        let fail_strength = duration_from_days(1.);
        let win_strength = duration_from_days(20.);

        let result =
            Reviews::pure_expected_gain(recall_rate, current_strength, fail_strength, win_strength);
        dbg!(result);
    }

    #[test]
    fn test_get_prediction_data() {
        let reviews = generate_random_reviews(9);
        dbg!(&reviews);
        let prev_review = reviews.0.last().unwrap().timestamp;
        let time = prev_review + Duration::from_secs(86400);
        let x = MlArgs::get_prediction_data_at_time(time, reviews);
        dbg!(x);
    }

    #[test]
    fn wtf() {
        let reviews = generate_random_reviews(3);

        let x = MlArgs::new(&reviews);
        dbg!(&x);

        let y = x.get_windowed(3);
        dbg!(y);

        let reviews = generate_random_reviews(7);
        let x = MlArgs::new_truncated(reviews, 4).unwrap();

        let y = x.as_training_data().join(",");
        dbg!(y);

        let reviews = generate_random_reviews(7);
        let time = reviews.0.last().unwrap().timestamp + Duration::from_secs(4 * 86400);
        let x = MlArgs::tail_prediction_data_at_time(time, reviews, 4);
        dbg!(x);
    }

    #[test]
    fn test_truncated() {
        let reviews = generate_random_reviews(4);
        let time = reviews.0.last().unwrap().timestamp + Duration::from_secs(86400);
        let x = MlArgs::tail_prediction_data_at_time(time, reviews, 1);
        dbg!(x);
    }

    #[test]
    fn test_stability() {
        let reviews = vec![
            Review {
                timestamp: Duration::from_secs(1687124756),
                grade: Grade::None,
                time_spent: Duration::default(),
            },
            Review {
                timestamp: Duration::from_secs(1687158818),
                grade: Grade::Some,
                time_spent: Duration::default(),
            },
            Review {
                timestamp: Duration::from_secs(1687248985),
                grade: Grade::Some,
                time_spent: Duration::default(),
            },
            Review {
                timestamp: Duration::from_secs(1687439802),
                grade: Grade::Some,
                time_spent: Duration::default(),
            },
            Review {
                timestamp: Duration::from_secs(1687853599),
                grade: Grade::Late,
                time_spent: Duration::default(),
            },
            Review {
                timestamp: Duration::from_secs(1687853599),
                grade: Grade::Some,
                time_spent: Duration::default(),
            },
        ];

        //reviews.pop();

        let reviews = Reviews(reviews);
        let x = reviews.stability().unwrap().as_secs_f32() / 86400.;
        dbg!(x);
    }

    fn debug_review(passed: f32, success: bool) -> Review {
        Review {
            timestamp: Duration::default() + Duration::from_secs_f32(passed * 86400.),
            grade: if success { Grade::Some } else { Grade::Late },
            time_spent: Duration::default(),
        }
    }

    #[test]
    fn test_expected_gain() {
        let reviews = vec![
            debug_review(0., true),
            debug_review(1., false),
            debug_review(10., false),
        ];
        let reviews = Reviews(reviews);
        dbg!(reviews.expected_gain());
    }

    #[test]
    fn foobar() {
        let vec: Vec<i32> = vec![];
        dbg!(vec.iter().all(|x| x == &0));
    }

    #[test]
    fn test_load_cards_from_folder() {
        let _category = Category(vec!["maths".into(), "calculus".into()]);

        //let cards = Card::load_cards_from_folder(&category);
        //insta::assert_debug_snapshot!(cards);
    }

    #[test]
    fn debug_strength() {
        let stability = Duration::from_secs_f32(100. * 86400.);
        let time_passed = Duration::from_secs_f32(0. * 86400.);
        let x = calculate_memory_strength(0.9, time_passed, stability);
        dbg!(x);
    }

    #[test]
    fn test_strength() {
        let stability = Duration::from_secs(86400);
        let days_passed = Duration::default();
        let recall_rate = Reviews::calculate_recall_rate(&days_passed, &stability);
        assert_eq!(recall_rate, 1.0);

        let days_passed = Duration::from_secs(86400);
        let recall_rate = Reviews::calculate_recall_rate(&days_passed, &stability);
        assert_eq!(recall_rate, 0.9);
    }
}
