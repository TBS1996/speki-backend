use crate::card::{CardCache, CardLocation, SavedCard};
use crate::common::get_last_modified;
use crate::paths::{self, get_cards_path};
use crate::Id;
use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::Path;
use std::path::PathBuf;

pub type CardFilter = Box<dyn FnMut(&Id, &mut CardCache) -> bool>;

// Represent the category that a card is in, can be nested
#[derive(Ord, PartialOrd, Eq, Hash, Debug, Clone, Default, PartialEq)]
pub struct Category(pub Vec<String>);

fn read_lines<P>(filename: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    reader.lines().collect::<Result<_, _>>()
}

impl Category {
    pub fn get_all_tags() -> BTreeSet<String> {
        let cats = Category::get_following_categories(&Category::root());
        let mut tags = BTreeSet::new();

        for cat in cats {
            let path = cat.as_path().join("tags");
            if let Ok(lines) = read_lines(path) {
                tags.extend(lines);
            }
        }
        tags.remove("");
        tags
    }

    pub fn get_tags(&self) -> BTreeSet<String> {
        let mut tags = BTreeSet::new();
        let mut cat = self.clone();

        loop {
            let path = cat.as_path().join("tags");
            if let Ok(lines) = read_lines(path) {
                tags.extend(lines);
            }
            if cat.0.is_empty() {
                break;
            }
            cat.0.pop();
        }
        tags
    }

    pub fn root() -> Self {
        Self::default()
    }

    pub fn joined(&self) -> String {
        self.0.join("/")
    }
    pub fn from_card_path(path: &Path) -> Self {
        let without_prefix = path.strip_prefix(paths::get_cards_path()).unwrap();
        let folder = without_prefix.parent().unwrap();

        let components: Vec<String> = Path::new(folder)
            .components()
            .filter_map(|component| component.as_os_str().to_str().map(String::from))
            .collect();

        let categories = Self(components);

        if categories.as_path().exists() {
            categories
        } else {
            panic!();
        }
    }

    pub fn get_containing_cards(&self) -> HashSet<SavedCard> {
        let directory = self.as_path();
        let mut cards = HashSet::new();

        for entry in std::fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("toml") {
                let card = SavedCard::from_path(path.as_path()).into_card();
                let location = CardLocation::new(&path);
                let last_modified = get_last_modified(path);

                let full_card = SavedCard::new(card, location, last_modified);
                cards.insert(full_card);
            }
        }
        cards
    }

    pub fn get_containing_card_ids(&self) -> HashSet<Id> {
        self.get_containing_cards()
            .into_iter()
            .map(|card| card.id().to_owned())
            .collect()
    }

    pub fn sort_categories(categories: &mut [Category]) {
        categories.sort_by(|a, b| {
            let a_str = a.0.join("/");
            let b_str = b.0.join("/");
            a_str.cmp(&b_str)
        });
    }
    pub fn get_following_categories(&self) -> HashSet<Self> {
        let categories = Category::load_all().unwrap();
        let catlen = self.0.len();
        categories
            .into_iter()
            .filter(|cat| cat.0.len() >= catlen && cat.0[0..catlen] == self.0[0..catlen])
            .collect()
    }

    pub fn print_it(&self) -> String {
        self.0.last().unwrap_or(&"root".to_string()).clone()
    }

    pub fn print_full(&self) -> String {
        let mut s = "/".to_string();
        s.push_str(&self.joined());
        s
    }

    pub fn print_it_with_depth(&self) -> String {
        let mut s = String::new();
        for _ in 0..self.0.len() {
            s.push_str("  ");
        }
        format!("{}{}", s, self.print_it())
    }

    pub fn import_category() -> Self {
        let cat = Self(vec!["imports".into()]);
        std::fs::create_dir_all(cat.as_path()).unwrap();
        dbg!(cat.as_path());
        cat
    }

    pub fn load_all() -> io::Result<Vec<Self>> {
        let root = get_cards_path();
        let root = root.as_path();
        let mut folders = Vec::new();
        Self::collect_folders_inner(root, root, &mut folders)?;
        folders.push(Category::default());
        Category::sort_categories(&mut folders);
        Ok(folders)
    }

    pub fn _append(mut self, category: &str) -> Self {
        self.0.push(category.into());
        self
    }

    fn collect_folders_inner(
        root: &Path,
        current: &Path,
        folders: &mut Vec<Category>,
    ) -> io::Result<()> {
        if current.is_dir() {
            for entry in fs::read_dir(current)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    // Compute the relative path from root to the current directory
                    let rel_path = path
                        .strip_prefix(root)
                        .expect("Failed to compute relative path")
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<String>>();
                    if !rel_path.last().unwrap().starts_with('_') {
                        folders.push(Self(rel_path));
                        Self::collect_folders_inner(root, &path, folders)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn as_path(&self) -> PathBuf {
        let categories = self.0.join("/");
        let path = format!("{}/{}", get_cards_path().to_string_lossy(), categories);
        PathBuf::from(path)
    }

    fn get_cards_with_filter(&self, mut filter: CardFilter, cache: &mut CardCache) -> Vec<Id> {
        self.get_containing_card_ids()
            .into_iter()
            .filter(|card| filter(card, cache))
            .collect()
    }

    pub fn get_unfinished_cards(&self, cache: &mut CardCache) -> Vec<Id> {
        self.get_cards_with_filter(Box::new(SavedCard::unfinished_filter), cache)
    }

    pub fn get_pending_cards(&self, cache: &mut CardCache) -> Vec<Id> {
        self.get_cards_with_filter(Box::new(SavedCard::pending_filter), cache)
    }

    pub fn get_random_review_cards(&self, cache: &mut CardCache) -> Vec<Id> {
        self.get_cards_with_filter(Box::new(SavedCard::random_filter), cache)
    }

    pub fn get_review_cards(&self, cache: &mut CardCache) -> Vec<Id> {
        self.get_cards_with_filter(Box::new(SavedCard::review_filter), cache)
    }
}

#[cfg(test)]
mod tests {

    use crate::categories::Category;

    use super::*;

    #[test]
    fn test_load_all() {
        let root = Path::new("./testing");
        let mut folders = vec![];
        Category::collect_folders_inner(root, root, &mut folders).unwrap();

        insta::assert_debug_snapshot!(folders);
    }

    #[test]
    fn test_joined() {
        let category = Category(vec!["foo".into(), "bar".into()]);
        let joined = category.joined();
        insta::assert_debug_snapshot!(joined);
    }

    #[test]
    fn test_as_card_path() {
        let cards_path = paths::get_cards_path()
            .join("foo")
            .join("bar")
            .join("guten tag.toml");
        Category::from_card_path(cards_path.as_path());
    }

    /*
    #[test]
    fn test_from_card_path() {
        let card_path = "./testing/maths/calculus/491f8b92-c943-4c4b-b7bf-f7d483208eb0.toml";
        let card_path = Path::new(card_path);
        let x = Category::from_card_path(card_path);
        insta::assert_debug_snapshot!(x);
    }

    #[test]
    fn test_as_path() {
        let category = Category(vec!["foo".into(), "bar".into()]);
        let x = category.as_path();
        insta::assert_debug_snapshot!(x);
    }

    #[test]
    fn test_as_path_with_id() {
        let id = uuid!("8bc35fe2-f02b-4633-8f1b-306eb4e09cd2");
        let category = Category(vec!["foo".into(), "bar".into()]);
        let x = category.as_path_with_id(id);
        insta::assert_debug_snapshot!(x);
    }
    */
}
